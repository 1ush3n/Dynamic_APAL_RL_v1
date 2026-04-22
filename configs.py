
class configs:
    # ------------------
    # 路径配置 (Paths)
    # ------------------
    # 默认数据路径 (如果没有通过命令行参数指定)
    data_dir = "data"
    data_file_path = "data/290.csv" 
    worker_pool_path = "data/worker_pool_fixed.csv"
    
    # ------------------
    # 环境与图相关 (Environment & Graph)
    # ------------------
    n_j = 290                       # 任务(工序)数量估计 (Graph Nodes)
    n_m = 5                         # 站位数量 (Stations)
    n_w_max = 120                   # 工人池总上限 (最大可配置的工人数量，固定池容量)
    n_w_min = 50                    # 每回合训练随机抽取的最小工人数 (Domain Randomization)
    n_w = 100                        # 每回合训练抽取的最大工人数，及验证(Eval)阶段固定的工人数
                                    # 注意：实际任务数由 data_loader 动态加载，此处仅作参考或 Embedding 初始化上界
    max_station_capacity_ratio = 0.4  # [Hybrid Masking] 单个站位最大容许绑定全厂总人数的比例，超过此值则站位被强制 Mask 屏蔽
    max_slots_per_station = 3        # [Slot Model] 每站位同时执行的最大工序数（物理工位槽），满槽后新工序需等待
    
    # ------------------
    # 模型超参数 (Model Hyperparameters)
    # ------------------
    hidden_dim = 128                # 隐藏层维度 (Embedding Size)
    num_gat_layers = 5              # GAT 层数 (Message Passing Depth)
    num_heads = 4                   # 多头注意力头数 (Attention Heads)
    dropout = 0.0                   # Dropout 比率 (在PPO中开启Dropout会导致Rollout和Update时Logits不一致，引发毁灭性的KL散度爆炸，必须设为0)
    
    # 输入特征维度 (根据 environment.py 中的 _get_observation 定义)
    task_feat_dim = 17              # Task Node Input Features [Duration, Status(4), Type(10), Wait(1), Demand(1)]
    worker_feat_dim = 21            # Worker Node Input Features [Efficiency(1), Skills(10), is_free(1), ProjectedWait(1), Lock(8)]
    station_feat_dim = 15           # Station Node Input Features [Load(1), BoundRatio(1), MobileRatio(1), FreeBoundRatio(1), SlotWait(1), RelSumLoad(1), RelMaxLoad(1)]
    
    # ------------------
    # 泛化性与域随机化 (Domain Randomization)
    # ------------------
    randomize_durations = True      # 是否在训练期间开启工时随机扰动
    dur_random_range = 0.2          # 工时扰动幅度 (0.2 表示基础工时的 ±20% 波动)
    curriculum_episodes = 500       # [课程式学习] 训练前 N 轮强制关闭所有随机因子，稳定 Critic 拟合
    
    # ------------------
    # PPO 训练超参数 (PPO Training)
    # ------------------
    lr = 1e-4                       # 初始学习率 (3000节点序列极长，不可轻易放大以免陷入局部最优死坑)
    gamma = 0.999                  # [治病良方] 3000步超级长线，远视能力必须拉满！建议至少0.999（视野1000步）。0.995的视野仅为200步，太过短视。
    k_epochs = 4                    # 每次更新循环次数
    eps_clip = 0.2                  # PPO Clip阈值 (e.g. 0.1 ~ 0.2)
    clip_v_grad_norm = 0.1          # 保护 Value Network 梯度的防破甲护盾
    batch_size = 4                 # 严防 RTX 4060 爆显存
    max_slots_per_station = 3      # 物理环境:每个站位允许的最大并行工序数
    r_coef_std = 0.5                # [核心修正] 解决 T=0 时坍缩到单机台的黑洞效应
    
    estimated_cmax_station_slots = 1.0 # 预估未来最大并行流度 (基于最严酷的1个槽位预估)
    
    c_policy = 1.0                  # Policy Loss 权重
    c_value = 0.5                   # Critic 价值损失权重
    
    r_coef_makespan = 1.0           # 宏观目标：Makespan 下班时间推移惩罚 (极其容易稀疏，因为只看瓶颈)
    deadlock_penalty_makespan = 400.0 # [修正] 死锁惩罚项（2500过大容易引起V-Loss爆炸，收缩回 400）
    reward_scale    = 0.005         # 全局奖励缩放乘数：在 environment 层面将上几千分差的值域缩放到 [-5, 5] 内，极大地稳定 Critic 的方差
    
    # 由于动作空间极大（120工人x5站位），初始熵极高。必须大幅调低系数，防止熵梯度掩盖排单梯度。
    c_entropy = 0.001                
    c_entropy_end = 0.0001            # 终点逼近0，强制模型在后期做出确定性抉择
    entropy_decay_episodes = 1000   # 大幅延长探索期！
    accumulation_steps = 16       # 16 * 4 = 64 (有效Batch大小)
    gae_lambda = 0.95               # GAE 优势函数的衰减因子 (290短序列不需要太大的 Lambda，0.95让信号更明确)
    sgdr_t0 = 150                   # 针对多节点大图大幅延长重启周期 (150 ep 一个深空潜航)
    
    max_episodes = 3000             # 探索万亿级组合的三千大劫
    update_every_episodes = 2       # 多少个 Episode 收集一次数据进行 PPO 更新
    eval_freq = 2                  # 多少个 Episode 进行一次评估
    eval_temperature = 0.0         # 验证环境和测试推演时的采样温度(0.0 = 完全贪婪)
    sample_temperature = 1.0        # [重要] 训练采样时的温度，必须锁定为 1.0 以保证 PPO On-Policy 数学一致性
    
    # 优化器与权重平滑策略
    # 警告: 二者不应同时取得主导地位。由于 PPO 对动量敏感，建议用纯净的 AdamW(关闭SF) 配合 EMA(开启) 最为平稳。
    use_schedule_free = False       # [修改] 默认关闭 Schedule-Free 以避免 PPO 多 Epoch 更新时的 KL 爆炸
    use_ema = True                  # [新增] 决定是否在验证（Eval）与推理时采用指数移动平均(EMA)版本权重
    ema_decay = 0.995               # [新增] EMA 移动衰减因子
    
    # ------------------
    # 自我模仿学习 (SIL)超参数
    # ------------------
    use_sil = True                  # 是否开启 SIL (Self-Imitation Learning)
    sil_capacity = 10               # 保存最佳历史轨迹的条数 (名人堂容量)
    sil_batch_size = 8              # 每次更新时从名人堂抽取并复温的样本数 
    c_sil = 0.1                     # SIL Loss 权重 (辅助 Actor)
    sil_epochs = 2                  # PPO 更新后，专门分配几次 Epoch 供网络回忆名人堂
    sil_threshold = 200.0           # 历史门槛阈值，起步先随便定个值，后来靠优先级队列动态卡阈值
    
    kl_early_stop = 0.02            # 绝对熔断阈值 (收敛以防止破坏信任域)

    
    # [Network Architecture Ablations - 2026-03-14]
    use_autoregressive_worker = True  # 开启自回归选人机制 (优化A)
    use_attention_critic = True       # 开启Critic的全局注意力池化 (优化B)
    
    # 默认消融参数防呆保护 (避免部分模块未经过 args_parser 初始化)
    ablation_no_mask = False          # 禁用硬约束计算
    ablation_no_gat = False           # 禁用GAT特征提取
    ablation_no_pointer = False       # 禁用指针网络
    seed = 42                         # 固定评测种子
    
    # ------------------
    # 日志与监控 (Logging)
    # ------------------
    log_dir = "tf-logs"                # TensorBoard 日志目录
