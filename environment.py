import gymnasium as gym
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from gymnasium import spaces
import sys
import os
import pandas as pd
import heapq
import networkx as nx 

# (Removed hardcoded sys.path.append to comply with standard project struct)

from data_loader import load_data
from configs import configs

# Event Definition
# time: Event occur time
# type: 'TASK_FINISH'
# data: {'task_id': int, 'worker_ids': list, 'station_id': int}
# Event Definition
class Event:
    """
    仿真事件类
    Attributes:
        time (float): 事件发生的时间
        type (str): 事件类型 (目前主要使用 'TASK_FINISH')
        data (dict): 事件携带的数据 (如 task_id, worker_ids 等)
    """
    def __init__(self, time, type, data):
        self.time = time
        self.type = type
        self.data = data
        
    def __lt__(self, other):
        # 用于优先队列的排序，时间小的在前
        return self.time < other.time


# ---------------------------------------------------------------------------
# 航空装配线环境 (AirLineEnv_Graph)
# ---------------------------------------------------------------------------
class AirLineEnv_Graph(gym.Env):
    """
    基于图的航空装配线强化学习环境。
    
    核心特性:
    1. 异构图状态: 包含 Task, Worker, Station 三种节点及其相互关系。
    2. 离散事件仿真: 时间推进基于事件(Event-Driven)，而非固定步长。
    3. 复杂约束: 包含工艺优先关系、技能匹配、站位空间约束。
    """
    
    # Gymnasium Metadata
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self, data_path="工序约束_50.xlsx", seed=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 设置随机种子以保证环境复现性 (Determinism)
        # 这对于验证集评估至关重要
        if seed is not None:
            np.random.seed(seed)
            # torch.manual_seed(seed) # 如果涉及 torch 的随机生成，也应设置
        
        # 加载数据
        self.raw_data = load_data(data_path)
        self.num_tasks = self.raw_data['num_tasks']
        self.num_workers = configs.n_w
        self.num_stations = configs.n_m
        
        # 动作空间: Tuple(Task, Station, Worker_List_Leader, Num_Workers)
        # 注意: 标准 Gym 不支持变长动作，这里定义为多离散空间仅作示意。
        # 实际 Agent (PPOAgent) 会处理具体的动作解码。
        self.action_space = spaces.MultiDiscrete([self.num_tasks, self.num_stations, self.num_workers])
        
        # 状态变量初始化
        self.current_time = 0.0
        # 任务状态: 0=不可用(Not Ready), 1=就绪(Ready), 2=已调度(Scheduled)
        self.task_status = np.zeros(self.num_tasks, dtype=int) 
        self.worker_free_time = np.zeros(self.num_workers, dtype=float) 
        # 工人定岗状态: 0=机动人员(未绑定), 1..num_stations=已绑定站位
        self.worker_locks = np.zeros(self.num_workers, dtype=int)
        self.station_loads = np.zeros(self.num_stations, dtype=float)
        self.station_wall_clock = np.zeros(self.num_stations, dtype=float)
        
        # 事件队列 (Priority Queue)
        self.event_queue = []
        
        # 解析固定站位约束 (Fixed Station Constraint)
        # 从原始数据中读取 'fixed_station' 列
        self.fixed_stations = -np.ones(self.num_tasks, dtype=int)
        if 'fixed_station' in self.raw_data['task_df'].columns:
            for idx, val in enumerate(self.raw_data['task_df']['fixed_station']):
                if pd.isna(val): continue
                # 解析逻辑: 支持 "Station 1", "S1", "1" 等格式
                s_idx = -1
                try:
                    val_str = str(val).lower().strip()
                    if val_str.startswith('station'):
                         s_idx = int(float(val_str.split()[-1])) - 1
                    elif val_str.startswith('s'): # S1, S2...
                         s_idx = int(float(val_str[1:])) - 1
                    else:
                         s_idx = int(float(val_str)) - 1 # 假设 Excel 中是 1-based index
                except:
                    pass
                
                if 0 <= s_idx < self.num_stations:
                    self.fixed_stations[idx] = s_idx
        
        # 初始化异构图数据结构
        self.init_hetero_data()
        
        # 计算全局基底工时总量，用于 [站位工时过载掩码]
        # (因为实际运行时有技能加成等，只用 Base duration 估算一个大概的上限)
        self.total_base_workload = torch.sum(self.task_static_feat[:, 0] * self.task_static_feat[:, 2]).item()
        
    def init_hetero_data(self):
        """
        初始化异构图的静态特征 (Task, Worker)。
        包含由 'seed' 控制的随机初始化逻辑。
        """
        data = HeteroData()
        
        # ------------------
        # 1. 任务节点 (Task Nodes)
        # ------------------
        task_df = self.raw_data['task_df']
        # 特征: [Duration, SkillType, DemandWorkers]
        dur = torch.tensor(task_df['duration'].values, dtype=torch.float).unsqueeze(1)
        skill = torch.tensor(task_df['skill_type'].values, dtype=torch.float).unsqueeze(1)
        demand = torch.tensor(task_df['demand_workers'].values, dtype=torch.float).unsqueeze(1)
        # 强制至少需要 1 人
        demand = torch.clamp(demand, min=1.0)
        
        self.task_static_feat = torch.cat([dur, skill, demand], dim=1)
        
        # ------------------
        # 2. 工人节点 (Worker Nodes)
        # ------------------
        # Load full worker pool
        pool_path = configs.worker_pool_path
        if not os.path.exists(pool_path):
             pool_path = os.path.join(os.getcwd(), pool_path)
             
        full_worker_df = pd.read_csv(pool_path)
        self.n_w_max = len(full_worker_df)
        self.full_worker_efficiency = full_worker_df['efficiency'].values
        self.full_worker_skill_matrix = torch.tensor(
            full_worker_df[[f'skill_{i}' for i in range(10)]].values, 
            dtype=torch.float
        )
        
        # Default workers (used in eval)
        self.num_workers = configs.n_w
        
        # Initialize default sizes just to keep base shapes valid before first reset
        self.worker_efficiency = self.full_worker_efficiency[:self.num_workers]
        self.worker_skill_matrix = self.full_worker_skill_matrix[:self.num_workers]
        # 计算每种技能的最大需求人数，确保每种技能至少有这么多工人拥有，
        # 防止出现 "任务需要5人，但全场只有3个合格工人" 的死锁情况。
        self.worker_static_feat = torch.tensor(self.worker_efficiency, dtype=torch.float).unsqueeze(1)
        
        # [再次鲁棒性检查] Check and Clamp Demand
        # 双重保险：如果初始化后发现某技能工人总数仍少于某任务需求，强制降低该任务需求。
        skill_capacity = self.worker_skill_matrix.sum(dim=0) # [10]
        
        clamped_count = 0
        for t in range(self.num_tasks):
            t_skill = int(skill[t].item())
            t_demand = int(demand[t].item())
            
            cap = int(skill_capacity[t_skill].item())
            if cap == 0:
                # 理论上不应发生，除非逻辑错误。兜底处理。
                print(f"CRITICAL: Skill {t_skill} has 0 workers! Force assigning Worker 0.")
                self.worker_skill_matrix[0, t_skill] = 1.0
                skill_capacity[t_skill] += 1
                cap = 1
                
            if t_demand > cap:
                demand[t] = cap
                clamped_count += 1
                
        if clamped_count > 0:
            print(f"[Robustness] Auto-clamped demand for {clamped_count} tasks to match worker availability.")
            
        # 更新被 Clamp 后的特征
        self.task_static_feat = torch.cat([dur, skill, demand], dim=1)
        
        # 预计算图拓扑 (前驱/后继)
        self.predecessors = {i: [] for i in range(self.num_tasks)}
        self.successors = {i: [] for i in range(self.num_tasks)}
        
        edge_index = self.raw_data['precedence_edges'].numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            self.successors[src].append(dst)
            self.predecessors[dst].append(src)
            
        self.num_preds = np.array([len(self.predecessors[i]) for i in range(self.num_tasks)])
        
        # 计算全局的关键路径和最晚允许站位 (持久化静态特征，只计算一次)
        self.is_critical = self._calculate_cpm()
        self.max_allowed_stations = self._calculate_max_allowed_stations()
        
        self.base_data = data
        self.obs_data = None # 将在 reset 中 clone
        
        # 预先分配静态底座张量，避免 step 过程中不断进行内存申请
        self.base_task_x = torch.zeros((self.num_tasks, 17))
        # [Domain Randomization] 备份只读的基础工时分布，用于后续加噪
        self.base_durations = dur.clone() / 100.0  
        self.base_task_x[:, 0:1] = self.base_durations
        
        type_onehot = torch.zeros((self.num_tasks, 10))
        type_indices = skill.long().clamp(0, 9)
        type_onehot.scatter_(1, type_indices, 1)
        self.base_task_x[:, 5:15] = type_onehot
        self.base_task_x[:, 16:17] = demand
        
        # [Feature Upgrade] worker base feat + wait_time slot (10 dims now instead of 9)
        self.base_worker_x = torch.cat([self.worker_static_feat, self.worker_skill_matrix, torch.zeros((self.num_workers, 10))], dim=1)
        # [Feature Upgrade] station base feat + slot_wait_time + relative loads (15 dims)
        self.base_station_x = torch.zeros((self.num_stations, 15))
        
    def reset(self, randomize_duration=False, randomize_workers=False, seed=None, options=None):
        """
        重置环境状态以开始新的 Episode。
        如果在训练阶段开启 randomize_duration，则按 ±range 对静态工时进行伪装修改。
        如果在训练阶段开启 randomize_workers，则动态随机抽取固定工人池的一个子集（领域随机化）。
        """
        # Gymnasium seed forward compatibility
        super().reset(seed=seed, options=options)
        if seed is not None:
             np.random.seed(seed)
             
        # ====================
        # [Domain Randomization] Worker Pool Sampling
        # ====================
        if randomize_workers:
            min_w = configs.n_w_min
            max_w = configs.n_w
            self.num_workers = np.random.randint(min_w, max_w + 1)
            
            # 保障覆盖率的抽样：确保所有必要的技能都有人会
            req_skills = self.task_static_feat[:, 1].unique().long().numpy()
            selected = set()
            for req in req_skills:
                capable_workers = np.where(self.full_worker_skill_matrix[:, req] == 1)[0]
                if len(capable_workers) > 0:
                    selected.add(np.random.choice(capable_workers))
            
            remaining = list(set(range(self.n_w_max)) - selected)
            num_to_add = self.num_workers - len(selected)
            if num_to_add > 0:
                selected.update(np.random.choice(remaining, num_to_add, replace=False))
            else:
                self.num_workers = len(selected)
                
            w_indices = np.array(list(selected))
            np.random.shuffle(w_indices)
        else:
            self.num_workers = configs.n_w
            w_indices = np.arange(self.num_workers)
            
        self.worker_efficiency = self.full_worker_efficiency[w_indices]
        self.worker_skill_matrix = self.full_worker_skill_matrix[w_indices]
        self.worker_static_feat = torch.tensor(self.worker_efficiency, dtype=torch.float).unsqueeze(1)
        
        # 重建动态的 base_worker_x (维度随 num_workers 变化)
        # 1(efficiency) + 10(skills) + 10(Padding: 1 wait, 1 free, 8 locks) = 21 dims
        self.base_worker_x = torch.cat([self.worker_static_feat, self.worker_skill_matrix, torch.zeros((self.num_workers, 10))], dim=1)
        
        # 重置运行状态张量
        self.current_time = 0.0
        self.task_status.fill(0) 
        self.worker_free_time = np.zeros(self.num_workers, dtype=float)
        self.worker_locks = np.zeros(self.num_workers, dtype=int)
        self.station_loads.fill(0.0)
        self.station_wall_clock.fill(0.0)
        self.event_queue = []
        
        # [Slot Model] - 记录每个站位中各并行工序的预计完成时间，用于计算等待延迟
        # 小顶堆：记录每个站位中各并行工序的预计完成时间，用于计算等待延迟
        self.station_task_finish_times = [[] for _ in range(self.num_stations)]
        
        self.assigned_tasks = [] 
        self.task_station_map = {} 
        self.task_end_times = -np.ones(self.num_tasks) 
        
        # 预分配边的内存空间
        MAX_TS_EDGES = self.num_tasks
        MAX_TW_EDGES = self.num_tasks * self.num_workers
        self.edge_ts_mem = torch.zeros((2, MAX_TS_EDGES), dtype=torch.long)
        self.edge_tw_mem = torch.zeros((2, MAX_TW_EDGES), dtype=torch.long)
        self.edge_ts_cnt = 0
        self.edge_tw_cnt = 0
        
        # 预计算图拓扑 (前驱/后继)
        self.predecessors = {i: [] for i in range(self.num_tasks)}
        self.successors = {i: [] for i in range(self.num_tasks)}
        
        edge_index = self.raw_data['precedence_edges'].numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            self.successors[src].append(dst)
            self.predecessors[dst].append(src)
            
        self.num_preds = np.array([len(self.predecessors[i]) for i in range(self.num_tasks)])
        self.completed_preds = np.zeros(self.num_tasks, dtype=int)
        
        # 设定初始任务状态
        # 没有前驱的任务设为 Ready (1)
        for i in range(self.num_tasks):
            if self.num_preds[i] == 0:
                self.task_status[i] = 1 # Ready
            else:
                self.task_status[i] = 0 # Not Ready
                
        # 克隆 Observation 数据并重建稀疏边矩阵 (由于 worker数量波动)
        self.obs_data = self.base_data.clone()
        
        w_indices_edge = torch.arange(self.num_workers).repeat_interleave(self.num_tasks)
        t_indices_edge = torch.arange(self.num_tasks).repeat(self.num_workers)
        task_req_skills = self.task_static_feat[:, 1].squeeze().long()
        has_skill_mask = self.worker_skill_matrix[w_indices_edge, task_req_skills[t_indices_edge]] == 1.0
        
        self.obs_data['worker', 'can_do', 'task'].edge_index = torch.stack([w_indices_edge[has_skill_mask], t_indices_edge[has_skill_mask]])
        
        # 动态篡改工时
        if randomize_duration:
            rnd_range = configs.dur_random_range
            noise = torch.ones_like(self.base_durations).uniform_(1.0 - rnd_range, 1.0 + rnd_range)
            perturbed_durations = self.base_durations * noise
            
            # 刷新模型底层观测到的图静态信息区 (Task_x[0])
            self.base_task_x[:, 0:1] = perturbed_durations
            # 刷新用于仿真计算真实验收时间 (Step duration calculation)
            self.task_static_feat[:, 0] = (perturbed_durations * 100.0).squeeze()
        else:
            # 安全还原成纯净考题卷子
            self.base_task_x[:, 0:1] = self.base_durations
            self.task_static_feat[:, 0] = (self.base_durations * 100.0).squeeze()
            
        # [关键路径计算 (CPM)]
        # 用于后续计算 Blocking Penalty
        self.is_critical = self._calculate_cpm()
        
        return self._get_observation()

    def _calculate_cpm(self):
        """
        关键路径法 (Critical Path Method, CPM)。
        逻辑:
        1. 正向递推 (Forward Pass) -> 计算最早开始时间 (ES)
        2. 反向递推 (Backward Pass) -> 计算最晚开始时间 (LS)
        3. 关键任务判定: 如果 ES == LS (Slack == 0)，则是关键任务。
        """
        durations = self.task_static_feat[:, 0].numpy()
        num_tasks = self.num_tasks
        
        # 1. 拓扑排序 (Kahn's Algorithm)
        in_degree = self.num_preds.copy()
        queue = [i for i in range(num_tasks) if in_degree[i] == 0]
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for v in self.successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # 2. 正向递推 (ES)
        es = np.zeros(num_tasks)
        for u in topo_order:
            my_es = 0
            for p in self.predecessors[u]:
                my_es = max(my_es, es[p] + durations[p])
            es[u] = my_es
            
        max_makespan = 0
        for u in range(num_tasks):
            max_makespan = max(max_makespan, es[u] + durations[u])
            
        # 3. 反向递推 (LS)
        ls = np.full(num_tasks, max_makespan)
        for u in reversed(topo_order):
            my_lf = max_makespan
            if self.successors[u]:
                children_ls = [ls[v] for v in self.successors[u]]
                my_lf = min(children_ls)
            
            ls[u] = my_lf - durations[u]
            
        # 4. 判定关键任务
        slack = ls - es
        is_critical = (slack < 1e-5)
        return is_critical

    def _calculate_max_allowed_stations(self):
        """
        通过反向拓扑遍历计算每个任务被允许部署的“最晚站位”。
        这是为了防止 RL 环境将一个无关任务扔到了非常靠后的工位，
        结果发现其【依赖子任务】在更早的站位是被限死 (Fixed Node) 的，导致永恒死锁。
        """
        num_tasks = self.num_tasks
        max_allowed = np.full(num_tasks, self.num_stations - 1)
        
        # 将 Fixed Stations 初始化进 max_allowed 
        for t in range(num_tasks):
            if self.fixed_stations[t] != -1:
                max_allowed[t] = self.fixed_stations[t]
                
        # 拓扑排序 (Kahn) - 用于获取线性处理顺序
        in_degree = self.num_preds.copy()
        queue = [i for i in range(num_tasks) if in_degree[i] == 0]
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for v in self.successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        # 沿着反向拓扑序更新最晚允许工位: 父节点的最晚工位不能晚于任何子节点的最晚工位
        for u in reversed(topo_order):
            for p in self.predecessors[u]:
                max_allowed[p] = min(max_allowed[p], max_allowed[u])
                
        return max_allowed

    def calculate_duration(self, task_id, team_indices):
        """
        非线性工时计算逻辑:
        T_real = (T_std * N_demand) / (Sum(Eff_i) * Synergy_Factor)
        
        Synergy Factor (协同系数): 
        人数越多，沟通成本越高，效率会有折扣。
        设定: 0.95 ^ (人数 - 1)
        """
        task_info = self.task_static_feat[task_id]
        t_std = task_info[0].item()
        n_demand = int(task_info[2].item())
        
        n_act = len(team_indices)
        if n_act == 0: return float('inf')
        
        # 效率求和
        sum_efficiency = sum(self.worker_efficiency[w] for w in team_indices)
        
        # 协同折扣
        syn_factor = 0.95 ** (n_act - 1)
        
        effective_capacity = sum_efficiency * syn_factor
        
        t_real = (t_std * n_demand) / effective_capacity
        return t_real

    def _get_station_earliest_available_time(self, sid, min_start_time, duration):
        """
        寻找最早的起步时间 T (T >= min_start_time)，使得在 [T, T+duration] 期间，
        站位 sid 的并发任务数严格低于 max_slots (即放入新任务后不超过 max_slots)。
        """
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        intervals = [(at[3], at[4]) for at in self.assigned_tasks if at[1] == sid]
        if len(intervals) < max_slots:
            return min_start_time
            
        candidate_times = [min_start_time] + [ed for (_, ed) in intervals if ed >= min_start_time]
        candidate_times.sort()
        
        for t in candidate_times:
            test_start = t
            test_end = t + duration
            # 扫描在 [test_start, test_end) 期间与现有区间的重叠
            endpoints = []
            for (st, ed) in intervals:
                # 严格重叠条件
                if max(st, test_start) < min(ed, test_end) - 1e-5:
                    endpoints.append((max(st, test_start), 1))
                    endpoints.append((min(ed, test_end), -1))
            
            if not endpoints: return test_start
            
            endpoints.sort(key=lambda x: (x[0], x[1]))
            cur_overlap = 0
            is_valid = True
            for pos, val in endpoints:
                cur_overlap += val
                if cur_overlap >= max_slots:
                    is_valid = False
                    break
            
            if is_valid: return test_start
            
        return candidate_times[-1]

    def _get_estimated_cmax(self):
        """
        [Phase 7: Estimated Cmax]
        计算预估完工期 (Estimated Cmax)，用于指导单步截断的强化学习，防止智能体恶意推迟长耗时任务。
        Cmax_est = max( 当前最大完工期, 各站位平均完工期 + (未排队任务总标准耗时 / 站位数 * 预估槽位) )
        """
        curr_max = np.max(self.station_wall_clock)
        
        # 取未分配的任务：0=Wait, 1=Ready
        unassigned_mask = (self.task_status == 0) | (self.task_status == 1)
        unassigned_sum = self.task_static_feat[unassigned_mask, 0].sum().item()
        
        from configs import configs
        slots = configs.estimated_cmax_station_slots
        
        curr_mean = np.mean(self.station_wall_clock)
        lower_bound = curr_mean + (unassigned_sum / (self.num_stations * slots))
        
        return max(curr_max, lower_bound)

    def step(self, action):
        """
        执行一步动作。
        Action: (task_id, station_id, team_list)
        """
        task_id, station_id, team = action
        
        # 记录执行前的 makespan 与平衡差 (Telescoping Sum Calculation Base)
        # 变更为具备下界预测能力的 Cmax_est
        prev_makespan = self._get_estimated_cmax()
        prev_std = np.std(self.station_loads)
        
        # 1. 执行逻辑
        duration = self.calculate_duration(task_id, team)
        
        # ==========================================
        # [Forward Allocation Engine]
        # 计算该工序真正的起步时间：必须满足（现在，人齐，且工位有空）
        # ==========================================
        
        # 1. 团队集结完毕时间 (木桶原理)
        team_ready_time = self.current_time
        if team:
            team_ready_time = max([self.worker_free_time[w] for w in team])
            
        # [NEW] 2. 前置工序完成时间 (彻底解决拓扑时序错乱 Bug)
        pred_ready_time = self.current_time
        preds = self.predecessors.get(task_id, [])
        for p in preds:
            pred_ready_time = max(pred_ready_time, self.task_end_times[p])

        # 结合基本条件，计算进入工位的【最低期望起始点】
        min_start_bound = max(self.current_time, team_ready_time, pred_ready_time)

        # [NEW] 3. 站位槽位腾出时间 (解决超并发重叠，使用 Sweep Line)
        station_ready_time = min_start_bound
        if station_id >= 0:
            station_ready_time = self._get_station_earliest_available_time(station_id, min_start_bound, duration)
            
        # 4. 四者取大，得到实际安排进时间表里的开工时刻
        start_time = max(min_start_bound, station_ready_time)
        finish_time = start_time + duration
        
        # 更新工人状态与站位绑定
        for w in team:
            self.worker_free_time[w] = finish_time
            if self.worker_locks[w] == 0 and station_id != -1:
                self.worker_locks[w] = station_id + 1
        
        if station_id != -1:
            # [Slot Model] 将本工序完成时间塞入站位的可用时间池
            # 不再维护已废弃的 station_active_tasks
            heapq.heappush(self.station_task_finish_times[station_id], finish_time)
            
            # 更新站位工作量总和 (Workload - 人.小时)
            self.station_loads[station_id] += duration * len(team) 
            
            # 更新真实的站位物理下班时间 (Wall-Clock Makespan)
            self.station_wall_clock[station_id] = max(self.station_wall_clock[station_id], finish_time)
        
        # 更新任务状态
        self.task_status[task_id] = 2 # 2=已调度
        self.task_end_times[task_id] = finish_time
        self.task_station_map[task_id] = station_id
        
        self.assigned_tasks.append((task_id, station_id, team, start_time, finish_time))
        
        if station_id != -1: # exclude virtual zero-duration
            ts_ptr = self.edge_ts_cnt
            self.edge_ts_mem[0, ts_ptr] = task_id
            self.edge_ts_mem[1, ts_ptr] = station_id
            self.edge_ts_cnt += 1
            
            for w in team:
                tw_ptr = self.edge_tw_cnt
                self.edge_tw_mem[0, tw_ptr] = task_id
                self.edge_tw_mem[1, tw_ptr] = w
                self.edge_tw_cnt += 1
        
        # 2. 添加事件到队列
        heapq.heappush(self.event_queue, Event(finish_time, 'TASK_FINISH', 
                                               {'task_id': task_id, 'worker_ids': team, 'station_id': station_id}))
        
        # 3. 推进仿真时间 (离散事件引擎)
        self._advance_time()
        curr_makespan = self._get_estimated_cmax()
        curr_std = np.std(self.station_loads)
        
        delta_makespan = curr_makespan - prev_makespan
        delta_std = curr_std - prev_std
        
        coef_makespan = configs.r_coef_makespan
        coef_std = configs.r_coef_std
        
        # 将原有的 terminal 扣除分摊到每一步的改变中
        reward = -(coef_makespan * delta_makespan) - (coef_std * delta_std)
        
        # 单步奖励硬截断，防止梯度极值爆炸
        reward = np.clip(reward, -50.0, 50.0)
        
        # 全局奖励缩放乘数：把原始的巨大的 makespan 分差在底层压缩至 [-5, 5] 的健康小区间
        reward = reward * configs.reward_scale
        
        # F. 终局结算 (Final Cleansing)
        done = (len(self.assigned_tasks) == self.num_tasks)
        
        return self._get_observation(), reward, done, {}

    def _advance_time(self):
        """
        推进时间 current_time 到下一个事件点。
        处理逻辑:
        1. 处理所有 <= current_time 的事件 (Task Finish)，释放前驱。
        2. [Zero-Duration Logic]: 如果解锁了 0工时 任务，立即执行并完成，不推进时间。
        3. 检查是否有 Valid 任务可做。
           - 如果有 -> 返回控制权给 Agent。
           - 如果无 -> 跳跃到下一个事件发生的时间点。
        """
        while True:
            # 增加队列非空与异常容量断言防护
            if not self.event_queue:
                self.current_time = self.max_time
                # Queue empty means simulation ends
                return
            
            if len(self.event_queue) > 10000:
                print("WARNING: Event queue limit exceeded! Forcing episode end to prevent OOM/Infinite Loop.")
                self.current_time = self.max_time
                self.event_queue = []
                return
                
            # 1. 处理所有已到期的事件
            while self.event_queue and self.event_queue[0].time <= self.current_time + 1e-5:
                ev = heapq.heappop(self.event_queue)
                if ev.type == 'TASK_FINISH':
                    tid = ev.data['task_id']
                    sid = ev.data['station_id']
                    # [Slot Model] 释放工位的历史使用记录 (将其从堆中清理)
                    # 由于我们使用 finish_time 推入，这里理论上不需要严苛清理，
                    # 只要为了防止 heap 无限膨胀而在完成时 pop 一次堆顶即可 (或者让其在下一次被覆写)
                    if sid >= 0:
                        if self.station_task_finish_times[sid]:
                            heapq.heappop(self.station_task_finish_times[sid])
                    
                    # 解锁后继
                    for succ in self.successors[tid]:
                        self.completed_preds[succ] += 1
                        if self.completed_preds[succ] == self.num_preds[succ]:
                            if self.task_status[succ] == 0:
                                self.task_status[succ] = 1 # Ready
            
            # 2. 0工时任务穿透逻辑 (Zero-Duration Penetration)
            # 必须立即处理掉所有 Ready 的 0工时任务
            ready_indices = np.where(self.task_status == 1)[0]
            zero_run_count = 0
            for t in ready_indices:
                dur = self.task_static_feat[t, 0].item()
                if dur < 1e-5: # Zero duration
                    # 立即完成
                    self.task_status[t] = 2 # Scheduled/Done
                    finish_time = self.current_time
                    self.task_end_times[t] = finish_time
                    self.task_station_map[t] = -1 # Virtual task
                    self.assigned_tasks.append((t, -1, [], finish_time, finish_time))
                    
                    # 加入事件队列 (为了统一触发 unlock 逻辑)
                    heapq.heappush(self.event_queue, Event(finish_time, 'TASK_FINISH', 
                                                           {'task_id': t, 'worker_ids': [], 'station_id': -1}))
                    zero_run_count += 1
                    
            if zero_run_count > 0:
                # 如果处理了 0工时任务，可能解锁了新任务，需要重新进入循环检查
                continue
            
            # 3. 检查是否需要 Agent 介入
            # 只有当存在 "可行 (Valid)" 任务时，才暂停并在 State 中返回。
            task_mask, _, _ = self.get_masks()
            
            if not task_mask.all():
                 # 至少有一个任务是 False (即 Valid)
                 break
            
            # 4. 如果没有 Valid 任务，则必须跳跃时间 (交由 _advance_time 内部或外部决定)
            if not self.event_queue:
                # 真正的环境空转末端，退出循环，让外部拿到掩码后再决定是否调用 try_wait_for_resources
                break
            
            # Jump to next event
            next_ev = self.event_queue[0]
            self.current_time = next_ev.time
            # 循环会继续处理 next_ev

    def try_wait_for_resources(self):
        """
        [Deadlock Fix] 当外部(如 train.py)拿到全 False 的 mask 时调用此方法。
        主动将时间快进到下一个事件发生（释放工人或槽位），然后返回 True。
        如果连未来的事件也没有了，说明发生了真正的死锁，返回 False。
        """
        if not self.event_queue:
            return False  # 真正的死锁：无人可用，且也没有人正在干活
            
        next_ev = self.event_queue[0]
        self.current_time = next_ev.time
        self._advance_time()  # 触发内部事件释放并尝试解锁新任务
        return True

    def get_masks(self):
        """
        生成动作掩码 (Action Masking)。
        
        Returns:
            task_mask: [N], True=Invalid (Masked), False=Valid
            station_mask: [N, M], True=Invalid
            worker_mask: [W], True=Invalid
            
        逻辑:
        1. 任务必须 Ready。
        2. 必须有足够的工人 (具备相应技能 & 当前空闲)。
        3. 站位必须符合拓扑约束 (<= 前驱的最大站位) - 暂未严格强制，目前主要靠 Fixed Station 约束。
        """
        # 1. Worker Mask (Global)
        # [Forward Allocation Enable] 不再因为“现在正忙”而掩码。任何时间都可以接新单进入待办队伍。
        worker_mask_np = np.zeros(self.num_workers, dtype=bool)
        worker_mask = torch.tensor(worker_mask_np, dtype=torch.bool)
        
        # 2. Task Mask
        task_mask = torch.ones(self.num_tasks, dtype=torch.bool) # Default Invalid
        station_mask = torch.ones((self.num_tasks, self.num_stations), dtype=torch.bool)
        
        ready_indices = np.where(self.task_status == 1)[0]
        
        # 使用向量化计算获取空闲技能与锁定状态可用量
        # 因为所有工人都允许（worker_mask 全 False），所以 free_workers 就是全体工人
        free_workers_idx = np.arange(self.num_workers)
        free_skills = self.worker_skill_matrix.numpy() 
        free_locks = self.worker_locks
                 
        for t in ready_indices:
            # A. 站位约束
            min_station = 0
            for p in self.predecessors[t]:
                p_s = self.task_station_map.get(p, -1)
                if p_s != -1:
                    min_station = max(min_station, p_s)
            
            fixed = self.fixed_stations[t]
            
            # B. 资源约束与站位可行性检查
            req_skill = int(self.task_static_feat[t, 1].item())
            req_demand = int(self.task_static_feat[t, 2].item())
            
            valid_stations = False
            max_station = self.max_allowed_stations[t]
            
            # 构建该任务合法的备选站位域
            station_range = [fixed] if fixed != -1 else list(range(min_station, min(self.num_stations, max_station + 1)))
            
            # 动态容量硬限制防拥堵 (打破 Argmax 陷入单工位求生的黑洞)
            max_slots = getattr(configs, 'max_slots_per_station', 15)
            
            has_skill = free_skills[:, req_skill] > 0.5
            
            for s in station_range:
                if s < 0 or s >= self.num_stations: continue
                
                # 如果该工位正在施工的任务数量超标，则采取物理硬闭环屏蔽
                if len(self.station_task_finish_times[s]) >= max_slots:
                    continue 

                # 检查能够支持在这个站位s工作的空闲人员：即 未绑定(0) 或 已经绑定到(s+1) 的人，并且拥有 req_skill
                compatible_lock = (free_locks == 0) | (free_locks == s + 1)
                
                avail = np.sum(compatible_lock & has_skill)
                
                if avail >= req_demand:
                    # 只有当这个特定的站位能凑齐人数时，该站位对该任务才是合法的！
                    station_mask[t, s] = False
                    valid_stations = True
                    
            if valid_stations:
                task_mask[t] = False # 如果有至少一个合法的站位能开工，该任务才被视为 Valid
                    
        return task_mask, station_mask, worker_mask

    def _get_observation(self):
        """
        [Phase 3.1: O(1) In-place Observation]
        构建异构图观测状态 (Observation)。
        彻底放弃在仿真步内的张量创建和拼接，转为 O(1) 预建内存片段的原地刷新。
        """
        data = self.base_data.clone()
        
        # 1. Task Features (In-place refresh)
        task_x = self.base_task_x.clone()
        task_x[:, 1:5] = 0.0 # reset status
        task_x[torch.arange(self.num_tasks), self.task_status + 1] = 1.0 # set status (offset by 1 to skip duration)
        data['task'].x = task_x
        
        # 2. Worker Features (In-place refresh)
        worker_x = self.base_worker_x.clone()
        
        # [Feature Upgrade: 连续时间特征支撑排队决策]
        # 计算工人的预估等待时间: max(0, worker_free_time - current_time) / 100.0
        wait_times_w = np.maximum(0, self.worker_free_time - self.current_time)
        # Efficiency(0), Skills(1~10), ProjectedWait(11)
        worker_x[:, 11] = torch.tensor(wait_times_w, dtype=torch.float) / 100.0
        
        is_free_bool = (self.worker_free_time <= self.current_time)
        worker_x[:, 12] = torch.tensor(is_free_bool, dtype=torch.float)
        
        # [Feature Upgrade] One-Hot Encode Lock state 
        worker_x[:, 13:21] = 0.0 # Clear
        lock_indices = torch.tensor(self.worker_locks, dtype=torch.long)
        lock_indices = torch.clamp(lock_indices, max=7) 
        worker_x[torch.arange(self.num_workers), 13 + lock_indices] = 1.0
        
        data['worker'].x = worker_x
        
        # 3. Station Features (In-place refresh)
        station_x = self.base_station_x.clone()
        station_x[:, 0] = torch.tensor(self.station_loads, dtype=torch.float) / 1000.0
        
        # [Feature Upgrade: Relative Load Competition]
        sum_loads = np.sum(self.station_loads)
        max_load = np.max(self.station_loads)
        station_x[:, 5] = torch.tensor(self.station_loads / (sum_loads + 1e-6), dtype=torch.float)
        station_x[:, 6] = torch.tensor(self.station_loads / (max_load + 1e-6), dtype=torch.float)
        
        # [Feature Upgrade: 连续时间特征支撑排队决策]
        # 计算站位槽位释放时间
        max_slots = getattr(configs, 'max_slots_per_station', 15)
        for s in range(self.num_stations):
            heap = self.station_task_finish_times[s]
            if len(heap) >= max_slots:
                wait_time_s = max(0, heap[0] - self.current_time)
            else:
                wait_time_s = 0.0
            station_x[s, 4] = wait_time_s / 100.0
            
        # [Feature Upgrade] Macro Strategic Features for Path Planning
        global_mobile_count = np.sum(self.worker_locks == 0)
        station_x[:, 2] = float(global_mobile_count) / self.num_workers
        
        for s in range(self.num_stations):
            # bound workers ratio
            bound_count = np.sum(self.worker_locks == s + 1)
            station_x[s, 1] = float(bound_count) / self.num_workers
            
            # available stationed workers ratio
            free_and_bound = np.sum((self.worker_locks == s + 1) & is_free_bool)
            station_x[s, 3] = float(free_and_bound) / self.num_workers
            
        data['station'].x = station_x
        
        # 4. Dynamic Edges (Assigned To)
        # 极速视图切片: O(1) 获取所有当前边索引，彻底剥离 Python 列表转换与动态构建张量的 O(N) 原罪!
        if self.edge_ts_cnt > 0:
            t_s_edge = self.edge_ts_mem[:, :self.edge_ts_cnt].clone()
            s_t_edge = torch.stack([t_s_edge[1], t_s_edge[0]], dim=0)
        else:
            t_s_edge = torch.empty((2, 0), dtype=torch.long)
            s_t_edge = torch.empty((2, 0), dtype=torch.long)
            
        data['task', 'assigned_to', 'station'].edge_index = t_s_edge
        data['station', 'has_task', 'task'].edge_index = s_t_edge
        
        if self.edge_tw_cnt > 0:
             t_w_edge = self.edge_tw_mem[:, :self.edge_tw_cnt].clone()
        else:
             t_w_edge = torch.empty((2, 0), dtype=torch.long)
             
        data['task', 'done_by', 'worker'].edge_index = t_w_edge
        
        return data

    def get_state_snapshot(self):
        """生成状态轻量级切片以存入 Buffer。"""
        return {
            'task_status': self.task_status.copy(),
            'worker_free_time': self.worker_free_time.copy(),
            'worker_locks': self.worker_locks.copy(),
            'station_loads': self.station_loads.copy(),
            'station_wall_clock': self.station_wall_clock.copy(),
            'current_time': self.current_time,
            'edge_ts_cnt': self.edge_ts_cnt,
            'edge_tw_cnt': self.edge_tw_cnt,
            'edge_ts_mem': self.edge_ts_mem[:, :self.edge_ts_cnt].clone() if self.edge_ts_cnt > 0 else torch.empty((2,0), dtype=torch.long),
            'edge_tw_mem': self.edge_tw_mem[:, :self.edge_tw_cnt].clone() if self.edge_tw_cnt > 0 else torch.empty((2,0), dtype=torch.long),
            'base_worker_x': self.base_worker_x.clone(),
            'can_do_edge_index': self.obs_data['worker', 'can_do', 'task'].edge_index.clone()
        }
        
    def rebuild_state_from_snapshot(self, snapshot):
        """
        基于快照恢复成 PyG 图结构，避免完整异构图深拷贝带来的极高缓存占用。
        """
        data = self.base_data.clone()
        
        task_x = self.base_task_x.clone()
        task_x[:, 1:5] = 0.0
        task_x[torch.arange(self.num_tasks), snapshot['task_status'] + 1] = 1.0
        data['task'].x = task_x
        
        snap_num_workers = len(snapshot['worker_free_time'])
        worker_x = snapshot['base_worker_x'].clone()
        
        # [Feature Upgrade: Wait time rebuild]
        wait_times_w = np.maximum(0, snapshot['worker_free_time'] - snapshot['current_time'])
        worker_x[:, 11] = torch.tensor(wait_times_w, dtype=torch.float) / 100.0
        
        is_free_bool = (snapshot['worker_free_time'] <= snapshot['current_time'])
        worker_x[:, 12] = torch.tensor(is_free_bool, dtype=torch.float)
        
        worker_x[:, 13:21] = 0.0
        snap_locks = snapshot['worker_locks']
        lock_indices = torch.tensor(snap_locks, dtype=torch.long).clamp(max=7)
        worker_x[torch.arange(snap_num_workers), 13 + lock_indices] = 1.0
        
        data['worker'].x = worker_x
        data['worker', 'can_do', 'task'].edge_index = snapshot['can_do_edge_index'].clone()
        
        station_x = self.base_station_x.clone()
        station_x[:, 0] = torch.tensor(snapshot['station_loads'], dtype=torch.float) / 1000.0
        
        # [Feature Upgrade: Wait time rebuild]
        max_slots = getattr(configs, 'max_slots_per_station', 15)
        
        global_mobile_count = np.sum(snap_locks == 0)
        station_x[:, 2] = float(global_mobile_count) / snap_num_workers
        
        for s in range(self.num_stations):
            bound_count = np.sum(snap_locks == s + 1)
            station_x[s, 1] = float(bound_count) / snap_num_workers
            
            free_and_bound = np.sum((snap_locks == s + 1) & is_free_bool)
            station_x[s, 3] = float(free_and_bound) / snap_num_workers
            
        data['station'].x = station_x
        
        if snapshot['edge_ts_cnt'] > 0:
            t_s_edge = snapshot['edge_ts_mem'].clone()
            s_t_edge = torch.stack([t_s_edge[1], t_s_edge[0]], dim=0)
        else:
            t_s_edge = torch.empty((2, 0), dtype=torch.long)
            s_t_edge = torch.empty((2, 0), dtype=torch.long)
            
        data['task', 'assigned_to', 'station'].edge_index = t_s_edge
        data['station', 'has_task', 'task'].edge_index = s_t_edge
        
        if snapshot['edge_tw_cnt'] > 0:
             t_w_edge = snapshot['edge_tw_mem'].clone()
        else:
             t_w_edge = torch.empty((2, 0), dtype=torch.long)
             
        data['task', 'done_by', 'worker'].edge_index = t_w_edge
        
        return data
