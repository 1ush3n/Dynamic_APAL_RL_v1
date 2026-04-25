
from dataclasses import dataclass, field
import os

@dataclass
class Config:
    # ------------------
    # 路径配置 (Paths)
    # ------------------
    data_dir: str = "data"
    data_file_path: str = os.path.join("data", "290.csv") # 默认验证集基准图
    worker_pool_path: str = os.path.join("data", "worker_pool_fixed.csv")
    
    # ------------------
    # 环境与图相关 (Environment & Graph)
    # ------------------
    n_j: int = 290                       # 任务(工序)数量估计 (Graph Nodes)
    n_m: int = 5                         # 站位数量 (Stations)
    n_w_max: int = 60                   # 工人池总上限 (最大可配置的工人数量，固定池容量)
    n_w_min: int = 40                    # 每回合训练随机抽取的最小工人数 (Domain Randomization)
    n_w: int = 50                        # 每回合训练抽取的最大工人数，及验证(Eval)阶段固定的工人数
    max_station_capacity_ratio: float = 0.8  # 单个站位最大容许绑定全厂总人数的比例
    max_slots_per_station: int = 3        # 每站位同时执行的最大工序数（物理工位槽）
    
    # ------------------
    # 模型超参数 (Model Hyperparameters)
    # ------------------
    hidden_dim: int = 128                # 隐藏层维度 (Embedding Size)
    num_gat_layers: int = 5              # GAT 层数 (Message Passing Depth)
    num_heads: int = 4                   # 多头注意力头数 (Attention Heads)
    dropout: float = 0.0                 # Dropout 比率 (设为0防止KL散度爆炸)
    
    task_feat_dim: int = 17              # Task Node Input Features
    worker_feat_dim: int = 21            # Worker Node Input Features
    station_feat_dim: int = 15           # Station Node Input Features
    
    # ------------------
    # 泛化性与域随机化 (Domain Randomization)
    # ------------------
    train_data_path_or_dir: str = "data/train_mix"        # 290+715 混合训练目录
    switch_dataset_every_updates: int = 1                 # 频繁切换以增强泛化能力
    randomize_durations: bool = True                      # 开启随机工时扰动
    dur_random_range: float = 0.2                         # 扰动幅度
    curriculum_episodes: int = 0        # 训练前 N 轮强制关闭所有随机因子
    
    # ------------------
    # 动态事件 (Dynamic Events)
    # ------------------
    enable_dynamic_events: bool = True     # 是否在训练期间开启突发动态事件（域随机化的一部分）
    prob_worker_absent_base: float = 0.0   # 工人缺勤的基础概率（验证和推理时的默认值）
    prob_worker_absent_max: float = 0.15   # 训练时最大随机波动的缺勤概率
    absence_duration_min: float = 1.0      # 缺勤的最短时间 (小时)
    absence_duration_max: float = 30.0     # 缺勤的最长时间 (小时)
    # ------------------
    # PPO 训练超参数 (PPO Training)
    # ------------------
    lr: float = 1e-4                       # 初始学习率
    gamma: float = 0.999                   # 折扣因子
    k_epochs: int = 4                      # 每次更新循环次数
    eps_clip: float = 0.2                  # PPO Clip阈值
    clip_v_grad_norm: float = 0.1          # 保护 Value Network 梯度的防破甲护盾
    batch_size: int = 4                    # 严防爆显存
    r_coef_std: float = 0.5                # 解决坍缩效应
    
    estimated_cmax_station_slots: float = 1.0 
    
    c_policy: float = 1.0                  # Policy Loss 权重
    c_value: float = 0.5                   # Critic 价值损失权重
    
    r_coef_makespan: float = 1.0           # 宏观目标：Makespan 下班时间推移惩罚
    deadlock_penalty_multiplier: float = 2.0 # 死锁惩罚项 (相对于理想总完工时间的倍数)
    reward_scale: float = 0.005            # 全局奖励缩放乘数
    
    c_entropy: float = 0.001                
    c_entropy_end: float = 0.0001            
    entropy_decay_episodes: int = 1000   
    accumulation_steps: int = 16       
    gae_lambda: float = 0.95               
    sgdr_t0: int = 150                   
    
    max_episodes: int = 3000             
    update_every_episodes: int = 2       
    eval_freq: int = 2                  
    eval_temperature: float = 0.0         
    sample_temperature: float = 1.0        
    
    use_schedule_free: bool = False       
    use_ema: bool = True                  
    ema_decay: float = 0.995               
    
    # ------------------
    # 自我模仿学习 (SIL)超参数
    # ------------------
    use_sil: bool = True                  
    sil_capacity: int = 10               
    sil_batch_size: int = 8              
    c_sil: float = 0.1                     
    sil_epochs: int = 2                  
    sil_threshold: float = 200.0           
    
    kl_early_stop: float = 0.02            
    
    use_autoregressive_worker: bool = True  
    use_attention_critic: bool = True       
    
    ablation_no_mask: bool = False          
    ablation_no_gat: bool = False           
    ablation_no_pointer: bool = False       
    seed: int = 42                         
    
    # ------------------
    # 日志与监控 (Logging)
    # ------------------
    log_dir: str = "tf-logs"                
    
    def update_from_dict(self, kwargs: dict):
        """支持通过字典动态更新配置"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

# 全局单例实例化，保持向下兼容
configs = Config()

