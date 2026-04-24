import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any

class EventType(Enum):
    TASK_FINISH = "TASK_FINISH"
    WORKER_LEAVE = "WORKER_LEAVE"
    WORKER_RETURN = "WORKER_RETURN"

@dataclass(order=True)
class Event:
    """
    仿真事件类
    Attributes:
        time (float): 事件发生的时间
        type (EventType): 事件类型
        data (Dict): 事件携带的数据 (如 task_id, worker_ids 等)
    """
    time: float
    type: EventType = field(compare=False)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)

class EventQueue:
    """
    基于 Priority Queue (heapq) 的事件管理引擎
    """
    def __init__(self, max_size: int = 10000):
        self._queue: List[Event] = []
        self.max_size = max_size
        
    def push(self, event: Event):
        if len(self._queue) >= self.max_size:
            raise RuntimeError(f"EventQueue 超过最大容量限制 ({self.max_size})，可能存在死循环！")
        heapq.heappush(self._queue, event)
        
    def pop(self) -> Event:
        if not self._queue:
            raise IndexError("pop from empty EventQueue")
        return heapq.heappop(self._queue)
        
    def peek(self) -> Event:
        if not self._queue:
            raise IndexError("peek from empty EventQueue")
        return self._queue[0]
        
    def is_empty(self) -> bool:
        return len(self._queue) == 0
        
    def clear(self):
        self._queue.clear()
        
    def __len__(self) -> int:
        return len(self._queue)
