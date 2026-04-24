import os
import time
import sys
import io
import traceback
import argparse

if sys.platform == 'win32':
    # 强制将标准输出重定向为 UTF-8，防止 Windows 终端在输出 Emoji 时崩溃
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 添加项目根目录到路径
# (Removed hardcoded sys.path.append to comply with standard project struct)

from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs
import pandas as pd
from baselines.heuristic.baseline_ga import GeneticAlgorithmScheduler
from utils.visualization import plot_gantt
import random
from sil_buffer import SILBuffer

# 设置全局随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# 经验回放缓冲区 (Memory Buffer)
# ---------------------------------------------------------------------------
class Memory:
    """
    存储 PPO 训练所需的轨迹数据。
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = [] # (task_mask, station_mask, worker_mask)
        self.values = [] # (state_value)
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]
        del self.values[:]
        
        # 显式释放残余对象，防 OOM 内存泄漏
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------
def evaluate_model(env, agent, num_runs=1, temperature=None):
    """
    使用包含温度平滑的定制定向策略评估当前模型性能。
    因当前评估环境与种子已完全固定（Deterministic），单次执行即可获得稳定基线。
    
    Returns:
        makespan (float): 多次运行均值最大完工时间 
        balance_std (float): 多次运行均值站位负载的标准差
        total_reward (float): 多次运行均值有效奖励总和
    """
    if temperature is None:
        temperature = getattr(configs, 'eval_temperature', 0.0)
        
    # 评估期间必须关闭 Dropout 等机制
    agent.policy.eval()
        
    makespans = []
    balances = []
    rewards = []
    schedules = []
    durations = []
    
    for _ in range(num_runs):
        # 验证场景绝对不可以使用任何数据扰乱！保证评估基线的绝对公平。
        state = env.reset(randomize_duration=False)
        done = False
        total_reward = 0
        device = agent.device
        
        start_time = time.time()
        while not done:
            task_mask, station_mask, worker_mask = env.get_masks()
            if task_mask.all():
                if env.try_wait_for_resources():
                     continue
                print(f"[Eval] FATAL DEADLOCK detected! Network topology restricts all remaining tasks and no future events.")
                done = True
                break
                
            # 引入验证温度的动作选择
            action_ret = agent.select_action(
                state.to(device), 
                mask_task=task_mask.to(device), 
                mask_station_matrix=station_mask.to(device),
                mask_worker=worker_mask.to(device),
                deterministic=(temperature == 0.0),
                temperature=temperature,
                is_eval=True
            )
            
            if action_ret[0] is None:
                print(f"[Eval] Agent failed to form a valid team (Worker Deadlock). Returning max penalty.")
                task_mask = torch.ones_like(task_mask) # 强制触发下方的 deadlock 终局结算
                break
                
            action, _, _, _, is_invalid = action_ret
            
            if getattr(configs, 'ablation_no_mask', False) and is_invalid:
                task_mask = torch.ones_like(task_mask) 
                break
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
        end_time = time.time()
        
        if len(env.assigned_tasks) != env.num_tasks:
            # 强化评估时的死锁反馈
            makespans.append(99999.0) 
            balances.append(9999.0)
            # 评估时的奖励衰减也与配置挂钩，设为 4 倍死锁惩罚以示严厉
            rewards.append(total_reward - (configs.deadlock_penalty_makespan * configs.r_coef_makespan * configs.reward_scale * 4))
            schedules.append([])
            durations.append(end_time - start_time)
        else:
            makespans.append(np.max(env.station_wall_clock)) # [CRITICAL] True physical completion time!
            balances.append(np.std(env.station_loads))     # [Maintain] Use workloads for labor distribution stats
            rewards.append(total_reward)
            schedules.append(env.assigned_tasks)
            durations.append(end_time - start_time)
        
    best_idx = np.argmin(makespans)
    return np.mean(makespans), np.mean(balances), np.mean(rewards), schedules[best_idx], np.mean(durations)

# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def train(args):
    try:
        print("--- 开始训练 (Starting Training) ---")
        
        # 接管顶层强化学习伪随机核心
        seed_cfg = configs.seed
        set_seed(seed_cfg)
        
        # 1. 初始化环境
        data_path = str(configs.data_file_path) if configs.data_file_path else "3000.csv"
        # 转换为绝对路径以防万一
        if not os.path.exists(data_path) and os.path.exists(os.path.join(os.getcwd(), data_path)):
             data_path = os.path.join(os.getcwd(), data_path)
             
        print(f"数据路径: {data_path}")
        # 固定种子以保证训练环境的一致性 (Determinism)
        env = AirLineEnv_Graph(data_path=data_path, seed=42)
        
        # [Validation] 单独开辟一个确定性验证环境，防止其污染训练轨迹状态
        eval_env = AirLineEnv_Graph(data_path=data_path, seed=2026)
        print("环境初始化完成.")
        
        # 2. 初始化设备与模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        model = HBGATPN(configs).to(device)
        print("模型已加载至设备.")
        
        # Init Agent
        # Calculate Total Timesteps for Scheduler
        total_updates = int(configs.max_episodes / configs.update_every_episodes)

        agent = PPOAgent(
            model=model,
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=configs.k_epochs,
            eps_clip=configs.eps_clip,
            device=device,
            batch_size=configs.batch_size,
            total_timesteps=total_updates
        )

        

        print(f"Agent Initialized. Total Scheduled Updates: {total_updates}")
        
        # 3. 断点续训 (Resume Training)
        start_episode = 1
        model_dir = "checkpoints"
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, "latest_checkpoint.pth")
        
        if args.resume and os.path.exists(checkpoint_path):
            print(f"正在从 {checkpoint_path} 恢复训练...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    agent.policy.load_state_dict(checkpoint['model_state_dict'])
                else: 
                     # Fallback if checkpoint is just a model_state_dict
                    agent.policy.load_state_dict(checkpoint)

                if 'optimizer_state_dict' in checkpoint:
                     try:
                         # 检查当前是否开启了SF，以及检查点中是否含有SF特有的 train_mode 标识
                         current_is_sf = getattr(configs, 'use_schedule_free', False)
                         param_groups = checkpoint['optimizer_state_dict'].get('param_groups', [])
                         is_sf_checkpoint = len(param_groups) > 0 and 'train_mode' in param_groups[0]
                         
                         if current_is_sf and not is_sf_checkpoint:
                             print("⚠️ 检查点中为普通 AdamW，当前开启了 ScheduleFree，已跳过优化器状态加载以防止崩溃。")
                         elif not current_is_sf and is_sf_checkpoint:
                             print("⚠️ 检查点中为 ScheduleFree，当前未开启 SF，跳过优化器状态加载。")
                         else:
                             agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                     except Exception as opt_e:
                         print(f"⚠️ 无法恢复优化器状态: {opt_e}")
                if 'optimizer_adam_state_dict' in checkpoint and hasattr(agent, 'optimizer_adam'):
                    agent.optimizer_adam.load_state_dict(checkpoint['optimizer_adam_state_dict'])
                
                if 'ema_model_state_dict' in checkpoint and hasattr(agent, 'ema_policy'):
                    agent.ema_policy.load_state_dict(checkpoint['ema_model_state_dict'])
                
                start_episode = checkpoint.get('episode', 0) + 1 if isinstance(checkpoint, dict) and 'episode' in checkpoint else 1
                print(f"恢复成功. 起始 Episode: {start_episode}")
            except Exception as e:
                print(f"⚠️ 恢复失败: 模型结构不匹配或缺少键值 (可能是 configs 修改了层数/维度). 跳过恢复。\n报错信息截取: {str(e)[:100]}...")
        
        # 最佳模型记录
        best_makespan = float('inf')
        best_model_dir = os.path.join(model_dir, "bestmodel")
        os.makedirs(best_model_dir, exist_ok=True)
        best_model_path = os.path.join(best_model_dir, "best_model.pth")
        
        # 4. TensorBoard 设置
        run_name = f"ALB_PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(configs.log_dir, run_name)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard 日志目录: {log_dir}")
        
        memory = Memory()
        
        # 初始化 SIL 名人堂 Buffer
        if getattr(configs, 'use_sil', False):
            sil_buffer = SILBuffer(capacity=getattr(configs, 'sil_capacity', 10))
            print(f"SIL (自我模仿学习) 已开启. 名人堂容量: {sil_buffer.capacity}")
        else:
            sil_buffer = None
        
        # 5. 训练循环参数
        max_episodes = configs.max_episodes 
        update_every_episodes = configs.update_every_episodes
        eval_freq = configs.eval_freq
        
        print(f"开始 Episode 循环 (Max: {max_episodes})...")
        
        for ep in range(start_episode, configs.max_episodes + 1):
            
            # 每轮迭代开始前强制恢复训练模式，开启 Dropout 等机制
            agent.policy.train()
            
            # 采样温度：训练时必须使用 configs.sample_temperature (通常为 1.0) 开启随机探索
            # 只有评估时才使用 configs.eval_temperature (通常为 0.0) 追求稳健
            current_temp = configs.sample_temperature
            
            # 课程式学习 (缓解初期 Critic 震荡)
            # 训练前期关闭随机时间与技能波动，给网络一个稳定的静态拟合期
            curr_curriculum_episodes = configs.curriculum_episodes
            apply_noise = configs.randomize_durations
            if ep <= curr_curriculum_episodes:
                apply_noise = False
                
            state = env.reset(randomize_duration=apply_noise, randomize_workers=apply_noise)
            
            done = False
            ep_reward = 0
            
            # 动态设置最大步数 (防止无限循环，通常设为任务数的2倍)
            max_steps = env.num_tasks * 2 
            
            for t in range(max_steps):
                # 获取 Mask
                task_mask, station_mask, worker_mask = env.get_masks()
                t_mask = task_mask.to(device)
                s_mask = station_mask.to(device)
                w_mask = worker_mask.to(device)
                if t_mask.all():
                     # 尝试跳跃时间等待未来事件解锁资源
                     if env.try_wait_for_resources():
                         continue

                     print(f"REAL DEADLOCK (Step {t}): 没有任何合法的任务派发（可能是前置任务全卡死），且未来无待完成事件。")
                     # 从 configs 读取死锁惩罚时长，计算公式为：惩罚小时数 * 系数 * 缩放
                     reward = -configs.deadlock_penalty_makespan * configs.r_coef_makespan * configs.reward_scale 
                     done = True
                     # 将死锁终局的极度惩罚，追加给上一个做出决策的动作
                     if len(memory.rewards) > 0:
                         memory.rewards[-1] += reward
                         memory.is_terminals[-1] = True
                     ep_reward += reward
                     break
                     
                if w_mask.all():
                     # 所有工人都在忙，理论上 _advance_time 会跳过这段时间，
                     # 但如果出现这种情况，说明时间推进逻辑可能需要检查。
                     # 这里仅作警告。
                     pass
                
                # 选择动作 (Stochastic with Annealed Temperature)
                action, logprob, val, _, is_invalid = agent.select_action(
                    state.to(device), 
                    mask_task=t_mask, 
                    mask_station_matrix=s_mask,
                    mask_worker=w_mask,
                    deterministic=False,
                    temperature=current_temp
                )
                
                # 无效动作的软惩罚
                if configs.ablation_no_mask and is_invalid:
                     reward = -configs.deadlock_penalty_makespan * configs.r_coef_makespan * configs.reward_scale 
                     done = True      # Terminate episode immediately to prevent infinite loops of illegal actions
                     
                     memory.states.append(env.get_state_snapshot()) 
                     memory.actions.append(action)
                     memory.logprobs.append(torch.tensor(logprob).to(device) if not isinstance(logprob, torch.Tensor) else logprob)
                     memory.rewards.append(reward)
                     memory.is_terminals.append(done)
                     memory.masks.append((task_mask.cpu(), station_mask.cpu(), worker_mask.cpu()))
                     memory.values.append(torch.tensor(val).to(device) if not isinstance(val, torch.Tensor) else val)
                     ep_reward += reward
                     break
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储轨迹 (取代原有几兆的完整异构图对象，只存极小的快照)
                memory.states.append(env.get_state_snapshot()) 
                memory.actions.append(action)
                memory.logprobs.append(logprob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                memory.masks.append((task_mask.cpu(), station_mask.cpu(), worker_mask.cpu()))
                memory.values.append(val)
                
                state = next_state
                ep_reward += reward
                
                if done:
                    break
            
            # 提取每次训练结束时的实时完工耗时
            ep_makespan = np.max(env.station_wall_clock) if len(env.assigned_tasks) > 0 else 0.0
            is_deadlock = len(env.assigned_tasks) < env.num_tasks
            status_str = "[DEADLOCK]" if is_deadlock else "[COMPLETED]"
            
            # 记录日志
            writer.add_scalar('Reward/Episode', ep_reward, ep)
            writer.add_scalar('Train/WallClock_Makespan', ep_makespan, ep)
            
            print(f"Episode {ep} {status_str} | Reward: {ep_reward:.2f} | Steps: {t+1} | Makespan: {ep_makespan:.1f}")
            
            # --- [新增] SIL 记录绝佳比赛 ---
            if sil_buffer is not None and not is_deadlock:
                # 假设我们只保留 makespan < sil_threshold 的，或者直接全丢给 Buffer 让它自己排
                if ep_makespan <= getattr(configs, 'sil_threshold', 9999.0):
                    added = sil_buffer.add_episode(ep_makespan, memory, configs.gamma)
                    if added:
                        print(f"      🌟 发现高光操作 (Makespan: {ep_makespan:.1f})! 已录入 SIL 名人堂。")
            
            # PPO 更新
            if ep % update_every_episodes == 0:
                try:
                    metrics = agent.update(memory, env)
                    
                    # --- [新增] 手动线性学习率衰减 (Linear LR Decay) ---
                    # 迫使 PPO 在末期走非常细微的精准步伐 (如果不使用 ScheduleFree)
                    if not getattr(configs, 'use_schedule_free', False):
                        progress = min(1.0, ep / configs.max_episodes)
                        min_lr = 1e-6
                        current_lr = configs.lr - progress * (configs.lr - min_lr)
                        for param_group in agent.optimizer.param_groups:
                            param_group['lr'] = current_lr
                    # ----------------------------------------------------
                    
                    for k, v in metrics.items():
                        writer.add_scalar(k, v, ep)
                        
                    # --- [新增] SIL 参数迭代 ---
                    if sil_buffer is not None:
                        sil_metrics = agent.update_sil(sil_buffer, env)
                        for k, v in sil_metrics.items():
                            writer.add_scalar(k, v, ep)
                            
                except RuntimeError as e:
                    if "out of memory" in str(e) or "OOM" in str(e):
                        print(f"\n⚠️ [OOM 防护] 显存不足，自动清理缓存并跳过本轮更新 (Episode {ep})")
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e
                finally:
                    memory.clear()
                
            # 定期评估与保存
              # [Validation Strategy]
            if ep % configs.eval_freq == 0:
                makespan, balance, eval_reward, best_sch, eval_duration = evaluate_model(eval_env, agent, num_runs=1, temperature=configs.eval_temperature)
                
                print(f"Epoch {ep:04d} [EVAL] | Makespan: {makespan:.2f} \t| Balance Std: {balance:.2f} \t| Eval Reward: {eval_reward:.2f} \t| Latency: {eval_duration:.4f}s")
                
                # 记录 Station Attention Weights
                # 监控 Critic 的注意力分布 (Gaze Variance)
                if configs.use_attention_critic:
                     s_var = getattr(agent.policy, 'last_s_var', 0.0)
                     writer.add_scalar('Critic/Gaze_Variance', s_var, ep)
                     print(f"      -> [Critic Gaze Variance]: {s_var:.6f}")
                
                writer.add_scalar('Eval/WallClock_Makespan', makespan, ep)
                writer.add_scalar('Eval/Workload_Balance_Std', balance, ep)
                writer.add_scalar('Eval/Average_Return', eval_reward, ep)
                writer.add_scalar('Eval/Inference_Time_sec', eval_duration, ep)
                
                # Save Latest
                save_dict = {
                    'episode': ep,
                    'model_state_dict': agent.policy.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict()
                }
                if hasattr(agent, 'optimizer_adam'):
                    save_dict['optimizer_adam_state_dict'] = agent.optimizer_adam.state_dict()
                if hasattr(agent, 'ema_policy'):
                    save_dict['ema_model_state_dict'] = agent.ema_policy.state_dict()
                
                torch.save(save_dict, checkpoint_path)
                
                # Save Best
                if makespan < best_makespan:
                    best_makespan = makespan
                    torch.save(agent.policy.state_dict(), best_model_path)
                    print(f"NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNew Best Model Saved! Makespan: {best_makespan}")
                    
                    # [Real-time Tracer] 实时快照抓拍最好成绩的排单策略
                    trace_dir = "checkpoints/eval_traces"
                    os.makedirs(trace_dir, exist_ok=True)
                    if best_sch:
                        tasks_data = []
                        for (tid, sid, team, start, end) in best_sch:
                             tasks_data.append({
                                 'TaskID': tid,
                                 'StationID': sid + 1,
                                 'Team': str(team),
                                 'Start': start,
                                 'End': end,
                                 'Duration': end - start
                             })
                        df = pd.DataFrame(tasks_data)
                        df.to_csv(os.path.join(trace_dir, f"Ep_{ep}_Best_Schedule.csv"), index=False)
                        plot_gantt(best_sch, os.path.join(trace_dir, f"Ep_{ep}_Gantt.png"))
                        print(f"📸 Real-time Schedule Trace exported to {trace_dir}/Ep_{ep}_Gantt.png")
                        
                    
        # =======================================================================
        # 6. 训练结束 - 终局性能测评与基线对比 (End of Training Evaluation)
        # =======================================================================
        print("\n" + "="*50)
        print("🎉 强化学习训练循环已结束！开始获取最强方案对比基线。")
        print("="*50)
        
        # 加载最好验证参数
        if os.path.exists(best_model_path):
             print(f"加载训练历史上最好的验证模型用于最终推演: {best_model_path}")
             try:
                 model.load_state_dict(torch.load(best_model_path, map_location=device))
             except RuntimeError as e:
                 print(f"⚠️ 警告: 历史最佳模型 ({best_model_path}) 的结构与当前配置不匹配，无法加载。将继续使用当前最新的训练结果进行推演！")
             
        # 配置 PPO 最终推演
        print("\n>>> [1/2] 开始执行 PPO Agent 的终局推演...")
        # 重新实例环境，避免脏数据
        eval_env = AirLineEnv_Graph(data_path=data_path, seed=2026)
        ppo_makespan, ppo_balance, _, ppo_assigned, ppo_duration = evaluate_model(eval_env, agent, num_runs=1, temperature=configs.eval_temperature)
        
        # 配置 GA 基准对抗
        print("\n>>> [2/2] 开始执行 Genetic Algorithm (GA) 基线推演...")
        ga_env = AirLineEnv_Graph(data_path=data_path, seed=2026)
        ga_scheduler = GeneticAlgorithmScheduler(ga_env, pop_size=30, max_gen=20)
        ga_start = time.time()
        ga_makespan, ga_balance, ga_assigned = ga_scheduler.run()
        ga_duration = time.time() - ga_start
        
        # --- 报表总结生成 ---
        print("\n" + "#"*60)
        print("🚀 终局对比结果报告 (PPO vs GA) 🚀")
        print(f"指标说明：Makespan/Balance (越小越好), 推理耗时 (越快越好)")
        print("-" * 60)
        print(f"| 模型算法类型          | Makespan (h) | Balance Std | 推理耗时 (秒) |")
        print(f"|-----------------------|--------------|-------------|---------------|")
        print(f"| 经典运筹学: (GA 基线) | {ga_makespan:12.2f} | {ga_balance:11.2f} | {ga_duration:13.4f} |")
        print(f"| 强化学习: (HB-GAT-PN) | {ppo_makespan:12.2f} | {ppo_balance:11.2f} | {ppo_duration:13.4f} |")
        print("#"*60 + "\n")
        
        # 导出最佳 PPO 与 GA 细节到各自的文件夹及画图
        output_dir_ppo = os.path.join("results", "PPO")
        output_dir_ga = os.path.join("results", "GA")
        os.makedirs(output_dir_ppo, exist_ok=True)
        os.makedirs(output_dir_ga, exist_ok=True)
        
        def save_schedule(tasks, prefix_name, target_dir):
            if not tasks: return
            tasks_data = []
            for (tid, sid, team, start, end) in tasks:
                 tasks_data.append({
                     'TaskID': tid,
                     'StationID': sid + 1,
                     'Team': str(team),
                     'Start': start,
                     'End': end,
                     'Duration': end - start
                 })
            df = pd.DataFrame(tasks_data)
            df.to_csv(os.path.join(target_dir, f"{prefix_name}_schedule.csv"), index=False)
            plot_gantt(tasks, os.path.join(target_dir, f"{prefix_name}_gantt.png"))
            
        print(f"正在向目录 ./results/PPO 与 ./results/GA 保存排程细节与甘特图...")
        save_schedule(ppo_assigned, "PPO_Final", output_dir_ppo)
        save_schedule(ga_assigned, "GA_Baseline", output_dir_ga)
        print("所有流程圆满结束！")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    from args_parser import get_base_parser
    parser = get_base_parser()
    args = parser.parse_args()
    
    # 动态写入 configs 对象，由于各处都会 import configs，可实现全局透传
    setattr(configs, 'ablation_no_gat', args.ablation_no_gat)
    setattr(configs, 'ablation_no_pointer', args.ablation_no_pointer)
    setattr(configs, 'ablation_no_mask', args.ablation_no_mask)
    setattr(configs, 'data_file_path', args.data_path)
    setattr(configs, 'seed', args.seed)
    setattr(configs, 'max_episodes', args.max_episodes)
    
    train(args)
