import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import math
import copy
from configs import configs
try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    AdamWScheduleFree = None

class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 智能体。
    负责与 Environment 交互，收集轨迹，并更新 Strategy Network。
    """
    def __init__(self, model, lr, gamma, k_epochs, eps_clip, device, batch_size=4, total_timesteps=0):
        self.policy = model.to(device)
        
        self.use_schedule_free = getattr(configs, 'use_schedule_free', False)
        
        # [SF Enhancement] 自动覆盖 EMA 防止控制权冲突与冗余计算
        if self.use_schedule_free:
            configs.use_ema = False
            print("INFO: ScheduleFree 优化器已开启，自动禁用传统的 EMA 同步机制。")
            
        if self.use_schedule_free and AdamWScheduleFree is not None:
            # [SF Enhancement] 动态调整预热期，设定为总更新步数的 5% (最小 100)
            warmup = getattr(configs, 'sf_warmup_steps', max(100, int(max(1, total_timesteps) * 0.05)))
            self.optimizer = AdamWScheduleFree(self.policy.parameters(), lr=lr, weight_decay=1e-4, warmup_steps=warmup)
        else:
            self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
            
        self.use_ema = getattr(configs, 'use_ema', False)
        self.ema_decay = getattr(configs, 'ema_decay', 0.995)
        if self.use_schedule_free and self.use_ema:
            print("WARNING: EMA and ScheduleFree are both enabled. This runs the risk of instability.")
            
        if self.use_ema:
            self.ema_policy = copy.deepcopy(self.policy).to(device)
            # 冻结 EMA 模型的梯度运算，它只作为旁观者接收指派
            for param in self.ema_policy.parameters():
                param.requires_grad = False
                
        self.lr = lr
        self.gamma = gamma          # 折扣因子
        self.k_epochs = k_epochs    # 每次 Update 的迭代轮数
        self.eps_clip = eps_clip    # PPO Clip参数 (e.g., 0.2)
        self.device = device
        self.batch_size = batch_size
        self.accumulation_steps = configs.accumulation_steps
        self.gae_lambda = configs.gae_lambda
        
        self.MseLoss = nn.MSELoss() 
        
        self.kl_early_stop = configs.kl_early_stop
        
        self.initial_lr = lr
        
        self.total_timesteps = max(1, total_timesteps)
        self.current_step = 0
        
        # 自适应评估新旧策略差距 (KL散度) 的方法在 update 尾部变动 LR。
    def select_action(self, obs, mask_task=None, mask_station_matrix=None, mask_worker=None, deterministic=False, temperature=1.0, is_eval=False):
        """
        选择动作 (Select Action)。
        
        Args:
            obs: 异构图观测数据 (HeteroData)
            mask_task: [N] Bool Tensor, True=Invalid
            mask_station_matrix: [N, S] Bool Tensor, True=Invalid
            mask_worker: [W] Bool Tensor, True=Invalid (Global)
            deterministic: 是否确定性选择 (ArgMax vs Sampling)
            temperature: 采样温度，T越小越贪婪，T越大越随机，忽略当 deterministic=True 时
            
        Returns:
            action_tuple: (task_id, station_id, team_indices_list)
            action_logprob: float
            state_value: float
            specific_station_mask: 用于 Memory 记录
        """
        no_mask = configs.ablation_no_mask
        
        if self.use_schedule_free:
            if is_eval:
                self.optimizer.eval()
            else:
                self.optimizer.train()
        
        # 决定当前激活的大脑 (Eval+EMA 时使用影子网络)
        if is_eval and getattr(self, 'use_ema', False) and hasattr(self, 'ema_policy'):
            active_policy = self.ema_policy
        else:
            active_policy = self.policy

        with torch.no_grad():
            x_dict, global_context = active_policy(obs)
            
            # 获取动态适配的 dtype 极小值（防止 FP16 下 -1e9 溢出）
            mask_value = torch.finfo(x_dict['task'].dtype).min / 2.0
            
            # ------------------
            # 1. 选择工序 (Select Task)
            # ------------------
            task_logits = active_policy.task_head(x_dict['task'], global_context, mask=mask_task if not no_mask else None)
            
            # [Robustness] 检查并处理 NaN
            if torch.isnan(task_logits).any():
                task_logits = torch.nan_to_num(task_logits, nan=mask_value)
            
            if deterministic:
                if mask_task is not None and not no_mask:
                    task_logits = task_logits.masked_fill(mask_task, mask_value)
                task_action = torch.argmax(task_logits)
                task_logprob = torch.tensor(0.0).to(self.device)
            else:
                if mask_task is not None and not no_mask:
                     task_logits = task_logits.masked_fill(mask_task, mask_value)
                
                # Check for all -inf
                if (task_logits <= mask_value * 0.99).all():
                     print("WARNING: All Task Logits -inf in select_action. Force picking 0.")
                     task_action = torch.tensor(0).to(self.device)
                     task_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        task_logits = task_logits / max(temperature, 1e-5)
                    task_dist = Categorical(logits=task_logits)
                    task_action = task_dist.sample()
                    task_logprob = task_dist.log_prob(task_action)
            
            t_idx = task_action.item()
            selected_task_emb = x_dict['task'][t_idx].unsqueeze(0) # [1, H]
            
            # 获取任务的人数需求
            raw_demand = obs['task'].x[t_idx, -1].item()
            demand = int(raw_demand)
            if demand < 1: demand = 1 # Safety clamp
            
            # ------------------
            # 2. 选择站位 (Select Station)
            # ------------------
            specific_station_mask = None
            if mask_station_matrix is not None:
                # [N, S] -> [1, S]
                specific_station_mask = mask_station_matrix[t_idx].unsqueeze(0)
            
            station_embs = x_dict['station'].unsqueeze(0)
            station_logits = active_policy.station_head(selected_task_emb, station_embs, mask=specific_station_mask if not no_mask else None)
            
            if torch.isnan(station_logits).any():
                station_logits = torch.nan_to_num(station_logits, nan=mask_value)
            
            if deterministic:
                if specific_station_mask is not None and not no_mask:
                     station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                station_action = torch.argmax(station_logits)
                station_logprob = torch.tensor(0.0).to(self.device)
            else:
                if specific_station_mask is not None and not no_mask:
                     station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                
                if (station_logits <= mask_value * 0.99).all():
                     print("WARNING: All Station Logits -inf. Force picking 0.")
                     station_action = torch.tensor(0).to(self.device)
                     station_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        station_logits = station_logits / max(temperature, 1e-5)
                    station_dist = Categorical(logits=station_logits)
                    station_action = station_dist.sample()
                    station_logprob = station_dist.log_prob(station_action)
                
            # ------------------
            # 3. 选择工人 (Select Workers) - 自回归
            # ------------------
            team_indices = []
            worker_logprobs = []
            
            # 动态 Mask: 初始 Mask + 技能 Mask
            current_worker_mask = mask_worker.clone() if mask_worker is not None else torch.zeros(obs['worker'].num_nodes, dtype=torch.bool).to(self.device)
            
            worker_feats = obs['worker'].x
            worker_skills = worker_feats[:, 1:11] # 10 dim
            
            task_type_idx = torch.argmax(obs['task'].x[t_idx, 5:15]).item() 
            
            has_skill = worker_skills[:, task_type_idx] > 0.5
            skill_mask = ~has_skill 
            
            s_act = station_action.item() + 1
            worker_locks = torch.argmax(worker_feats[:, 13:21], dim=1)
            lock_mask = (worker_locks != 0) & (worker_locks != s_act)
            
            if no_mask:
                current_worker_mask = skill_mask.to(self.device)
            else:
                current_worker_mask = current_worker_mask | skill_mask.to(self.device) | lock_mask.to(self.device)

            worker_embs = x_dict['worker'].unsqueeze(0)
            
            # 加入迭代阈值和 Fallback 防止因掩码过度重叠发生死循环
            max_iter = demand * 2
            iter_cnt = 0
            
            # 初始化团队记忆
            current_team_emb = None 
            
            while len(team_indices) < demand and iter_cnt < max_iter:
                iter_cnt += 1
                
                # 还有可选工人吗?
                if current_worker_mask.all():
                    break
                
                worker_logits = active_policy.worker_head.forward_choice(selected_task_emb, worker_embs, mask=current_worker_mask, current_team_emb=current_team_emb)
                
                if torch.isnan(worker_logits).any():
                    worker_logits = torch.nan_to_num(worker_logits, nan=mask_value)
                
                if deterministic:
                     if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                     if (worker_logits <= mask_value * 0.99).all(): break
                     
                     w_action = torch.argmax(worker_logits)
                     w_lp = torch.tensor(0.0).to(self.device)
                else:
                     if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                     
                     if (worker_logits <= mask_value * 0.99).all():
                         break # 无法继续选人
                     
                     if temperature != 1.0:
                         worker_logits = worker_logits / max(temperature, 1e-5)
                         
                     w_dist = Categorical(logits=worker_logits)
                     w_action = w_dist.sample()
                     w_lp = w_dist.log_prob(w_action)
                
                w_idx = w_action.item()
                team_indices.append(w_idx)
                worker_logprobs.append(w_lp)
                
                # 刷新已选团队表征记忆
                selected_worker_feats = worker_embs[0, team_indices, :]
                current_team_emb = selected_worker_feats.mean(dim=0, keepdim=True) # [1, H]
                
                # 更新 Mask (选过的人不能再选)
                current_worker_mask = current_worker_mask.clone() # 确保不 原地修改 影响下一轮
                current_worker_mask[w_idx] = True
            
            # [兜底逻辑] 若因过度竞争或死锁选不够人选
            if len(team_indices) < demand:
                if is_eval:
                    # [Evaluation Strict Mode] 验证期间绝对不允许兜底作弊！
                    # 如果选不够人，说明策略出现断层死锁，直接将失败上传以施加真实的验证集惩罚。
                    return None, 0.0, 0.0, None, True
                    
                # [Zero-Fallback Enforcement] 原有的兜底机制已被彻底移除。
                # 由于环境的 get_masks() 已经在物理和拓扑层面上保证了只有当满足 demand 人数（且技能、工位锁定状态都符合要求）时，
                # 站位和任务才是合法的。如果在这里选不出足够的人，说明前置掩码与内层选人掩码存在逻辑脱节，或出现了未知的计算漏洞。
                # 此时绝不可再凑数塞入假人或存入假概率，这会导致后期 update 产生爆炸的虚假 KL 并诱发一连串的崩溃！
                raise RuntimeError(
                    f"FATAL DEADLOCK: Failed to select enough valid workers (needed {demand}, got {len(team_indices)}).\n"
                    f"The masking logic in environment get_masks() strictly guarantees worker sufficiency.\n"
                    f"No manual fallback is ever allowed to preserve the KL purity. Please inspect the mask consistency!"
                )
            
            
            total_worker_logprob = sum(worker_logprobs) if worker_logprobs else torch.tensor(0.0).to(self.device)
            
            action_logprob = task_logprob + station_logprob + total_worker_logprob
            # 物理隔离 Critic 防止其巨大的 Value Error 梯度捣毁底层共享 GAT 拓扑特征
            # 传入完整的 state (batch_data)，由于处于 with torch.no_grad() 下，此处无需 detach，直接前向提取价值。
            state_value = active_policy.get_value(obs)
            
            action_tuple = (t_idx, station_action.item(), team_indices)
            
            # 检查 action 对于 soft penalty 的有效性
            is_invalid_action = False
            if mask_task is not None and mask_task[t_idx].item():
                is_invalid_action = True
            if specific_station_mask is not None and specific_station_mask[0, station_action.item()].item():
                is_invalid_action = True
            if mask_worker is not None:
                for w_idx in team_indices:
                    if mask_worker[w_idx].item():
                        is_invalid_action = True
            
        return action_tuple, action_logprob.item(), state_value.item(), specific_station_mask, is_invalid_action

    def update(self, memory, env=None):
        """
        PPO 更新逻辑。
        
        Args:
            memory: 存储轨迹的 Buffer
            
        Returns:
            metrics: dict, 用于 TensorBoard 记录
        """
        # 1. 计算广义优势估计 (GAE - Generalized Advantage Estimation)
        rewards = []
        advantages = []
        gae = 0
        
        # 将 rewards 与 values 张量化以进行 GAE 计算
        mem_rewards = memory.rewards
        mem_is_terminals = memory.is_terminals
        
        # 提取存储在 states 中的 state_values
        # (这需要在 select_action 之后被记录下来，如果没有记录，回退为普通的 MC 回报加基线)
        if hasattr(memory, 'values') and len(memory.values) == len(mem_rewards):
            mem_values = memory.values
            next_value = 0 # 终止状态后的 value 为 0
            
            for step in reversed(range(len(mem_rewards))):
                if mem_is_terminals[step]:
                    next_value = 0
                    gae = 0
                
                delta = mem_rewards[step] + self.gamma * next_value - mem_values[step]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages.insert(0, gae)
                next_value = mem_values[step]
                
            advantages = torch.tensor(advantages, dtype=torch.float32)
            rewards = advantages + torch.tensor(mem_values, dtype=torch.float32)
        else:
            # Fallback 到 Monte-Carlo + Advantage (如果缺少 Value 记录)
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(mem_rewards), reversed(mem_is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
                
            rewards = torch.tensor(rewards, dtype=torch.float32)
            # 兼容处理
            advantages = rewards.clone()
            
        # 归一化 Advantages 与 Returns (有助于长期负反馈环境的训练稳定性)
        if advantages.std() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        else:
            advantages = advantages - advantages.mean()
            
        # [CRITICAL FIX: Removed Return Normalization]
        # 绝不应对 Critic 的 Target Returns 进行动态批次标准化，
        # 否则每一轮 Update 的均值和方差都在变（移动靶），导致 Critic 永远无法收敛，产生巨大的梯度震荡。
        # 我们改用配置中的静态系数缩小全局 reward。
        
        # 2. 准备 Batch 数据
        old_actions = memory.actions 
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        
        # Pad Team List (变长 -> 定长 Tensor)
        max_team_size = max(len(a[2]) for a in old_actions) if old_actions else 1
        
        b_task = torch.tensor([a[0] for a in old_actions], dtype=torch.long)
        b_station = torch.tensor([a[1] for a in old_actions], dtype=torch.long)
        
        team_list = []
        for a in old_actions:
            t = a[2]
            pad = [-1] * (max_team_size - len(t))
            team_list.append(t + pad)
        b_team = torch.tensor(team_list, dtype=torch.long)
        
        # Attach targets to Data objects for Batching
        rebuilt_states = []
        if env is not None:
             for snap in memory.states:
                 rebuilt_states.append(env.rebuild_state_from_snapshot(snap))
        else:
             rebuilt_states = memory.states
             
        for i, state in enumerate(rebuilt_states):
            state.y_task = b_task[i].unsqueeze(0)
            state.y_station = b_station[i].unsqueeze(0)
            state.y_team = b_team[i].unsqueeze(0) 
            state.y_logprob = old_logprobs[i].unsqueeze(0)
            state.y_reward = rewards[i].unsqueeze(0)
            state.y_advantage = advantages[i].unsqueeze(0)
            
            # [Added] Load original state values for PPO Value Clipping
            if len(memory.values) > i:
                 state.y_value = torch.tensor([memory.values[i]], dtype=torch.float32)
            
            if i < len(memory.masks):
                t_mask, s_mask, w_mask = memory.masks[i]
                state.y_task_mask = t_mask
                state.y_station_mask = s_mask
                state.y_worker_mask = w_mask
        
        loader = DataLoader(rebuilt_states, batch_size=self.batch_size, shuffle=True)
        
        # 3. PPO Optimization Loop
        print(f"PPO Update: BatchSize={self.batch_size}, Total Batches={len(loader)}")
        
        avg_loss = 0
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_loss = 0
        update_counts = 0
        approx_kls = []
        
        if self.use_schedule_free:
             self.optimizer.train()
        
        self.optimizer.zero_grad()
            
        final_epoch = self.k_epochs - 1
        kl_meltdown_occurred = False 
        total_batches_diagnosed = 0 # 用于消除归零毛刺的统计口径
        for i_epoch in range(self.k_epochs):
            epoch_kls = []
            for step_idx, batch in enumerate(loader):
                batch = batch.to(self.device)
                
                # 当前策略的前向传播
                x_dict, global_context = self.policy(batch)
                
                # 独立骨干评估 state_values
                state_values = self.policy.get_value(batch).view(-1)
                
                # --- Re-evaluate LogProbs ---
                # A. Task LogProb
                from torch_geometric.utils import to_dense_batch
                task_x, p_mask = to_dense_batch(x_dict['task'], batch['task'].batch)
                
                # 恢复 Mask
                if hasattr(batch, 'y_task_mask'):
                    logical_task_mask, _ = to_dense_batch(batch.y_task_mask, batch['task'].batch)
                    combined_task_mask = logical_task_mask | (~p_mask)
                else:
                    combined_task_mask = ~p_mask
                    
                task_logits = self.policy.task_head(task_x, global_context, mask=combined_task_mask)
                if torch.isnan(task_logits).any(): task_logits = torch.nan_to_num(task_logits, nan=-1e9)
                
                task_dist = Categorical(logits=task_logits)
                task_lp = task_dist.log_prob(batch.y_task)
                entropy = task_dist.entropy()
                
                # B. Station LogProb
                batch_indices = torch.arange(batch.y_task.size(0)).to(self.device)
                sel_task_emb = task_x[batch_indices, batch.y_task] 
                
                station_x, s_p_mask = to_dense_batch(x_dict['station'], batch['station'].batch)
                
                if hasattr(batch, 'y_station_mask'):
                    dense_s_mask, _ = to_dense_batch(batch.y_station_mask, batch['task'].batch)
                    specific_station_mask = dense_s_mask[batch_indices, batch.y_task]
                    curr_s_mask = specific_station_mask | (~s_p_mask)
                else:
                    curr_s_mask = ~s_p_mask
                
                station_logits = self.policy.station_head(sel_task_emb, station_x, mask=curr_s_mask)
                if torch.isnan(station_logits).any(): station_logits = torch.nan_to_num(station_logits, nan=-1e9)
                
                station_dist = Categorical(logits=station_logits)
                station_lp = station_dist.log_prob(batch.y_station)
                entropy += station_dist.entropy()
                
                # C. Worker Team LogProb
                worker_x, w_p_mask = to_dense_batch(x_dict['worker'], batch['worker'].batch)
                team_lp = torch.zeros_like(task_lp)
                
                if hasattr(batch, 'y_worker_mask'):
                     d_w_mask, _ = to_dense_batch(batch.y_worker_mask.float(), batch['worker'].batch)
                     curr_mask = (d_w_mask > 0.5) | (~w_p_mask)
                else:
                     curr_mask = (~w_p_mask)
                
                # Add Skill Mask based on the selected task
                task_raw, _ = to_dense_batch(batch['task'].x, batch['task'].batch)
                sel_task_raw = task_raw[batch_indices, batch.y_task]
                task_type_idx = torch.argmax(sel_task_raw[:, 5:15], dim=1) # [B]
                
                worker_raw, _ = to_dense_batch(batch['worker'].x, batch['worker'].batch)
                worker_skills = worker_raw[:, :, 1:11] # [B, Max_W, 10]
                
                B_size, Max_W_size = worker_skills.shape[0], worker_skills.shape[1]
                b_indices_expanded = torch.arange(B_size).view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                w_indices_expanded = torch.arange(Max_W_size).view(1, -1).expand(B_size, -1).reshape(-1)
                t_indices_expanded = task_type_idx.view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                
                has_skill_flat = worker_skills[b_indices_expanded, w_indices_expanded, t_indices_expanded] > 0.5
                skill_mask = (~has_skill_flat).view(B_size, Max_W_size).to(self.device)
                
                s_act = batch.y_station + 1 # [B]
                worker_locks = torch.argmax(worker_raw[:, :, 13:21], dim=2) # [B, Max_W]
                s_act_expanded = s_act.view(B_size, 1).expand(B_size, Max_W_size).to(self.device)
                lock_mask = (worker_locks != 0) & (worker_locks != s_act_expanded)
                
                curr_mask = curr_mask | skill_mask | lock_mask.to(self.device)
                
                current_team_emb = None # [B, H]
                team_emb_sum = torch.zeros(B_size, worker_x.size(-1)).to(self.device)
                team_cnt = torch.zeros(B_size, 1).to(self.device)
                
                for k in range(batch.y_team.size(1)):
                    target = batch.y_team[:, k] 
                    valid_step = (target != -1)
                    if not valid_step.any(): continue
                    
                    logits = self.policy.worker_head.forward_choice(sel_task_emb, worker_x, mask=curr_mask, current_team_emb=current_team_emb)
                    if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=-1e9)
                    
                    dist = Categorical(logits=logits)
                    step_lp = dist.log_prob(torch.clamp(target, min=0)) 
                    team_lp[valid_step] += step_lp[valid_step]
                    entropy[valid_step] += dist.entropy()[valid_step]
                    
                    # Update current_team_emb
                    valid_b_indices = torch.nonzero(valid_step).squeeze(-1)
                    valid_targets = target[valid_step]
                    
                    selected_feats = worker_x[valid_b_indices, valid_targets]
                    
                    # 使用 clone() 保障 PyTorch 自动求导机制的连续性 (Gradient Preservation)
                    next_team_emb_sum = team_emb_sum.clone()
                    next_team_cnt = team_cnt.clone()
                    
                    next_team_emb_sum[valid_b_indices] += selected_feats
                    next_team_cnt[valid_b_indices] += 1
                    
                    team_emb_sum = next_team_emb_sum
                    team_cnt = next_team_cnt
                    
                    current_team_emb = team_emb_sum / torch.clamp(team_cnt, min=1.0)
                    
                    # Update mask for next worker in team
                    curr_mask = curr_mask.clone()
                    curr_mask[valid_b_indices, target[valid_step]] = True
                            
                total_lp = task_lp + station_lp + team_lp
                
                # LogProb Clipping, 防止后续的 torch.exp 发生指数散度爆炸
                total_lp = torch.clamp(total_lp, min=-20.0, max=2.0)
                
                # --- PPO Loss Calculation ---
                with torch.no_grad():
                    log_ratio = total_lp - batch.y_logprob.view(-1)
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    epoch_kls.append(approx_kl.item())

                # 极简 KL 熔断机制 (Meltdown Protection)
                loss_scale = 1.0
                hard_limit = self.kl_early_stop
                
                if approx_kl.item() > hard_limit:
                    print(f"      [KL MELTDOWN] Batch {step_idx}: KL={approx_kl.item():.4f} > Limit {hard_limit}. Applying extreme braking (0.01x loss).")
                    loss_scale = 0.01

                ratios = torch.exp(total_lp - batch.y_logprob.view(-1))
                
                # Use GAE advantages if available, else batch.y_reward - state_values (MC fallback)
                b_adv = batch.y_advantage.view(-1) if hasattr(batch, 'y_advantage') else (batch.y_reward.view(-1) - state_values.detach())
                
                # 动态衰减探索上限
                progress = min(1.0, self.current_step / max(1, self.total_timesteps))
                curr_eps_clip = self.eps_clip - progress * (self.eps_clip - 0.05)
                
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1-curr_eps_clip, 1+curr_eps_clip) * b_adv
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                c_val = configs.c_value
                decay_eps = configs.entropy_decay_episodes
                update_freq = configs.update_every_episodes
                decay_updates = max(1, decay_eps // update_freq)  # 将 Episode 跨度转换为 Update 次数跨度
                
                ent_progress = min(1.0, self.current_step / decay_updates)
                
                c_ent_base = configs.c_entropy
                c_ent_end = configs.c_entropy_end
                import math
                c_ent = c_ent_end + (c_ent_base - c_ent_end) * math.exp(-3.0 * ent_progress)
                
                c_pol = configs.c_policy
                
                b_reward = batch.y_reward.view(-1)
                value_loss = c_val * F.smooth_l1_loss(state_values, b_reward, beta=5.0)
                entropy_loss = -c_ent * entropy.mean()
                
                loss = c_pol * policy_loss + value_loss + entropy_loss
                
                # 应用软熔断缩放
                loss = loss * loss_scale
                
                # Backprop
                loss = loss / self.accumulation_steps # 归一化 Gradient
                loss.backward()
                
                if ((step_idx + 1) % self.accumulation_steps == 0) or (step_idx + 1 == len(loader)):
                    # 独立参数梯度裁剪
                    actor_params = [p for n, p in self.policy.named_parameters() if 'critic' not in n and 'attn' not in n]
                    critic_params = [p for n, p in self.policy.named_parameters() if 'critic' in n or 'attn' in n]
                    
                    torch.nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
                    # 给 Critic 挂装远比 Actor 更薄弱的装甲，防止局部脉冲带崩全盘
                    torch.nn.utils.clip_grad_norm_(critic_params, max_norm=configs.clip_v_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    update_counts += 1
                
                # Log Stats (记录原始未缩放的 loss 用于诊断)
                avg_loss += (loss.item() / max(1e-8, loss_scale)) * self.accumulation_steps
                avg_policy_loss += policy_loss.item()
                avg_value_loss += F.smooth_l1_loss(state_values, b_reward, beta=5.0).item()
                avg_entropy_loss += entropy.mean().item()
                total_batches_diagnosed += 1
            
            # 尽早触发 early stopping，防止退化
            # 计算当前 epoch 的平均 KL
            curr_epoch_kl = sum(epoch_kls) / len(epoch_kls) if epoch_kls else 0.0
            
            # 我们始终记录最后一轮未掐断的 KL 作为自适应引擎的参考
            approx_kls = epoch_kls
            
            # 如果偏离超过硬阈值，提前终止本次 Update 循环以保护模型
            if curr_epoch_kl > self.kl_early_stop:
                print(f"      -> Early stopping at epoch {i_epoch+1} due to reaching max KL: {curr_epoch_kl:.4f}")
                break
        
        # (已移除冗杂的学习率下降逻辑，完全交给 Schedule-Free 或恒定 LR)
        mean_kl = sum(approx_kls) / len(approx_kls) if approx_kls else 0.0
        
        self.current_step += 1
        
        # [EMA 更新] 每一轮外部 Update 结束（包括内部 k_epochs）后，由主模型向影子模型进行一次 Exponential Moving Averaging 同步
        if getattr(self, 'use_ema', False) and hasattr(self, 'ema_policy'):
            alpha = self.ema_decay
            with torch.no_grad():
                for ema_param, param in zip(self.ema_policy.parameters(), self.policy.parameters()):
                    ema_param.data.copy_(alpha * ema_param.data + (1.0 - alpha) * param.data)
                
        metrics = {
            'Loss/Total': avg_loss / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Loss/Policy': avg_policy_loss / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Loss/Value': avg_value_loss / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Loss/Entropy': avg_entropy_loss / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Loss/ApproxKL': mean_kl,
            'Train/LearningRate': self.optimizer.param_groups[0]['lr']
        }
        return metrics

    def update_sil(self, sil_buffer, env=None):
        """
        Self-Imitation Learning (SIL) Update.
        Samples trajectories from the expert buffer and trains the agent to clone them,
        using an advantage filter: only train if the historical return is better than current value estimate.
        """
        if len(sil_buffer) == 0:
            return {}
            
        sil_batch_size = getattr(configs, 'sil_batch_size', 8)
        sil_epochs = getattr(configs, 'sil_epochs', 2)
        c_sil = getattr(configs, 'c_sil', 0.1)
        
        avg_sil_loss = 0
        avg_sil_adv = 0
        update_cnt = 0
        
        for _ in range(sil_epochs):
            transitions = sil_buffer.sample_batch(sil_batch_size)
            if not transitions:
                break
                
            rebuilt_states = []
            b_task, b_station, b_team, b_return = [], [], [], []
            
            for t in transitions:
                snap = t['state_snap']
                if env is not None:
                    state = env.rebuild_state_from_snapshot(snap)
                else:
                    state = snap
                    
                b_task.append(t['action'][0])
                b_station.append(t['action'][1])
                b_team.append(t['action'][2])
                b_return.append(t['return'])
                
                state.y_task = torch.tensor([t['action'][0]], dtype=torch.long)
                state.y_station = torch.tensor([t['action'][1]], dtype=torch.long)
                state.y_reward = torch.tensor(t['return'], dtype=torch.float32) # Store directly on state
                
                if t['mask'] is not None:
                    state.y_task_mask, state.y_station_mask, state.y_worker_mask = t['mask']
                
                rebuilt_states.append(state)
                
            # Pad teams
            max_team = max(max((len(team) for team in b_team), default=0), 1)
            for i in range(len(rebuilt_states)):
                t = b_team[i]
                pad = [-1] * (max_team - len(t))
                rebuilt_states[i].y_team = torch.tensor([t + pad], dtype=torch.long)
            
            loader = DataLoader(rebuilt_states, batch_size=sil_batch_size, shuffle=False)
            
            for batch in loader:
                batch = batch.to(self.device)
                x_dict, global_context = self.policy(batch)
                state_values = self.policy.get_value(batch).view(-1)
                
                # --- A. Task LogProb Recalculation ---
                from torch_geometric.utils import to_dense_batch
                task_x, p_mask = to_dense_batch(x_dict['task'], batch['task'].batch)
                
                if hasattr(batch, 'y_task_mask'):
                    logical_task_mask, _ = to_dense_batch(batch.y_task_mask, batch['task'].batch)
                    combined_task_mask = logical_task_mask | (~p_mask)
                else:
                    combined_task_mask = ~p_mask
                    
                task_logits = self.policy.task_head(task_x, global_context, mask=combined_task_mask)
                if torch.isnan(task_logits).any(): task_logits = torch.nan_to_num(task_logits, nan=-1e9)
                task_dist = Categorical(logits=task_logits)
                task_lp = task_dist.log_prob(batch.y_task)
                
                # --- B. Station LogProb ---
                batch_indices = torch.arange(batch.y_task.size(0)).to(self.device)
                sel_task_emb = task_x[batch_indices, batch.y_task] 
                station_x, s_p_mask = to_dense_batch(x_dict['station'], batch['station'].batch)
                
                if hasattr(batch, 'y_station_mask'):
                    dense_s_mask, _ = to_dense_batch(batch.y_station_mask, batch['task'].batch)
                    specific_station_mask = dense_s_mask[batch_indices, batch.y_task]
                    curr_s_mask = specific_station_mask | (~s_p_mask)
                else:
                    curr_s_mask = ~s_p_mask
                    
                station_logits = self.policy.station_head(sel_task_emb, station_x, mask=curr_s_mask)
                if torch.isnan(station_logits).any(): station_logits = torch.nan_to_num(station_logits, nan=-1e9)
                station_dist = Categorical(logits=station_logits)
                station_lp = station_dist.log_prob(batch.y_station)
                
                # --- C. Worker LogProb ---
                worker_x, w_p_mask = to_dense_batch(x_dict['worker'], batch['worker'].batch)
                team_lp = torch.zeros_like(task_lp)
                
                if hasattr(batch, 'y_worker_mask'):
                     d_w_mask, _ = to_dense_batch(batch.y_worker_mask.float(), batch['worker'].batch)
                     curr_mask = (d_w_mask > 0.5) | (~w_p_mask)
                else:
                     curr_mask = (~w_p_mask)
                
                task_raw, _ = to_dense_batch(batch['task'].x, batch['task'].batch)
                sel_task_raw = task_raw[batch_indices, batch.y_task]
                task_type_idx = torch.argmax(sel_task_raw[:, 5:15], dim=1) 
                
                worker_raw, _ = to_dense_batch(batch['worker'].x, batch['worker'].batch)
                worker_skills = worker_raw[:, :, 1:11] 
                
                B_size, Max_W_size = worker_skills.shape[0], worker_skills.shape[1]
                b_indices_expanded = torch.arange(B_size).view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                w_indices_expanded = torch.arange(Max_W_size).view(1, -1).expand(B_size, -1).reshape(-1)
                t_indices_expanded = task_type_idx.view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                
                has_skill_flat = worker_skills[b_indices_expanded, w_indices_expanded, t_indices_expanded] > 0.5
                skill_mask = (~has_skill_flat).view(B_size, Max_W_size).to(self.device)
                
                s_act = batch.y_station + 1 
                worker_locks = torch.argmax(worker_raw[:, :, 13:21], dim=2) 
                s_act_expanded = s_act.view(B_size, 1).expand(B_size, Max_W_size).to(self.device)
                lock_mask = (worker_locks != 0) & (worker_locks != s_act_expanded)
                
                curr_mask = curr_mask | skill_mask | lock_mask.to(self.device)
                
                current_team_emb = None 
                team_emb_sum = torch.zeros(B_size, worker_x.size(-1)).to(self.device)
                team_cnt = torch.zeros(B_size, 1).to(self.device)
                
                for k in range(batch.y_team.size(1)):
                    target = batch.y_team[:, k] 
                    valid_step = (target != -1)
                    if not valid_step.any(): continue
                    
                    logits = self.policy.worker_head.forward_choice(sel_task_emb, worker_x, mask=curr_mask, current_team_emb=current_team_emb)
                    if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=-1e9)
                    
                    dist = Categorical(logits=logits)
                    step_lp = dist.log_prob(torch.clamp(target, min=0)) 
                    team_lp[valid_step] += step_lp[valid_step]
                    
                    valid_b_indices = torch.nonzero(valid_step).squeeze(-1)
                    valid_targets = target[valid_step]
                    selected_feats = worker_x[valid_b_indices, valid_targets]
                    
                    next_team_emb_sum = team_emb_sum.clone()
                    next_team_cnt = team_cnt.clone()
                    next_team_emb_sum[valid_b_indices] += selected_feats
                    next_team_cnt[valid_b_indices] += 1
                    
                    team_emb_sum = next_team_emb_sum
                    team_cnt = next_team_cnt
                    current_team_emb = team_emb_sum / torch.clamp(team_cnt, min=1.0)
                    
                    curr_mask = curr_mask.clone()
                    curr_mask[valid_b_indices, target[valid_step]] = True
                            
                total_lp = task_lp + station_lp + team_lp
                total_lp = torch.clamp(total_lp, min=-20.0, max=2.0)
                
                # --- SIL Advantage Filter ---
                # Retrieve returns stored in data
                b_reward = batch.y_reward.view(-1).to(self.device)
                # Advantage = Return - Value. Only positive advantage is considered (Better than expected).
                sil_advantage = torch.clamp(b_reward - state_values, min=0.0)
                
                valid_mask = sil_advantage > 1e-4
                if valid_mask.any():
                    # SIL Policy Loss equates to off-policy behavior cloning with advantage weighting
                    sil_policy_loss = -(total_lp[valid_mask] * sil_advantage[valid_mask].detach()).mean()
                    # SIL Value Loss updates the critic towards the higher empirical return
                    sil_value_loss = 0.5 * (sil_advantage[valid_mask] ** 2).mean()
                    
                    loss_sil = c_sil * (sil_policy_loss + configs.c_value * sil_value_loss)
                    
                    self.optimizer.zero_grad()
                    loss_sil.backward()
                    
                    # Optional: independent actor/critic gradient clipping for safety
                    actor_params = [p for n, p in self.policy.named_parameters() if 'critic' not in n and 'attn' not in n]
                    critic_params = [p for n, p in self.policy.named_parameters() if 'critic' in n or 'attn' in n]
                    torch.nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(critic_params, max_norm=configs.clip_v_grad_norm)
                    
                    self.optimizer.step()
                    
                    avg_sil_loss += loss_sil.item()
                    avg_sil_adv += sil_advantage[valid_mask].mean().item()
                    update_cnt += 1
                    
        metrics = {}
        if update_cnt > 0:
            metrics['Loss/SIL_Total'] = avg_sil_loss / update_cnt
            metrics['SIL/Advantage'] = avg_sil_adv / update_cnt
        return metrics

