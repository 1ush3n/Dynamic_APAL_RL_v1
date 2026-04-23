import torch
import numpy as np
from typing import Tuple, Any

class ActionMasker:
    """
    负责计算航空装配线强化学习环境的动作掩码 (Action Mask)。
    抽离自 AirLineEnv_Graph 以保持核心环境模块的整洁。
    """
    def __init__(self, env: Any):
        # 传入 env 实例的弱引用或直接引用
        self.env = env
        
    def get_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成动作掩码 (Action Masking)。
        
        Returns:
            task_mask: [N], True=Invalid (Masked), False=Valid
            station_mask: [N, M], True=Invalid
            worker_mask: [W], True=Invalid
        """
        env = self.env
        
        # 1. Worker Mask (Global)
        worker_mask_np = np.zeros(env.num_workers, dtype=bool)
        worker_mask = torch.tensor(worker_mask_np, dtype=torch.bool)
        
        # 2. Task Mask & Station Mask
        task_mask = torch.ones(env.num_tasks, dtype=torch.bool)
        station_mask = torch.ones((env.num_tasks, env.num_stations), dtype=torch.bool)
        
        ready_indices = np.where(env.task_status == 1)[0]
        
        free_workers_idx = np.arange(env.num_workers)
        free_skills = env.worker_skill_matrix.numpy() 
        free_locks = env.worker_locks
        
        from configs import configs
        max_slots = getattr(configs, 'max_slots_per_station', 15)
                 
        for t in ready_indices:
            # A. 站位前驱约束
            min_station = 0
            for p in env.predecessors[t]:
                p_s = env.task_station_map.get(p, -1)
                if p_s != -1:
                    min_station = max(min_station, p_s)
            
            fixed = env.fixed_stations[t]
            
            # B. 资源约束与站位可行性检查
            req_skill = int(env.task_static_feat[t, 1].item())
            req_demand = int(env.task_static_feat[t, 2].item())
            
            valid_stations = False
            max_station = env.max_allowed_stations[t]
            
            # 构建该任务合法的备选站位域
            station_range = [fixed] if fixed != -1 else list(range(min_station, min(env.num_stations, max_station + 1)))
            
            has_skill = free_skills[:, req_skill] > 0.5
            
            for s in station_range:
                if s < 0 or s >= env.num_stations: continue
                
                # 容量硬限制防拥堵
                if len(env.station_task_finish_times[s]) >= max_slots:
                    continue 

                # 检查能够支持在这个站位 s 工作的空闲人员
                compatible_lock = (free_locks == 0) | (free_locks == s + 1)
                avail = np.sum(compatible_lock & has_skill)
                
                if avail >= req_demand:
                    station_mask[t, s] = False
                    valid_stations = True
            
            if valid_stations:
                task_mask[t] = False
                
        return task_mask, station_mask, worker_mask
