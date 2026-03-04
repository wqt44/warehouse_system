"""
任务定义：包括任务类型、位置、优先级等
"""
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum
import numpy as np


class TaskType(Enum):
    """任务类型"""
    PICKUP = "pickup"      # 取货任务
    DELIVERY = "delivery"  # 送货任务


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"        # 待分配
    ASSIGNED = "assigned"      # 已分配
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败


@dataclass
class Task:
    """任务类"""
    task_id: int
    task_type: TaskType
    pickup_location: Tuple[int, int]
    dropoff_location: Tuple[int, int]
    priority: float = 1.0  # 优先级，越高越紧急
    created_time: int = 0
    assigned_robot_id: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    deadline: Optional[int] = None  # 截止时间步
    
    def to_feature_vector(self) -> np.ndarray:
        """转换为特征向量用于神经网络输入"""
        return np.array([
            self.task_id,
            self.priority,
            self.pickup_location[0],
            self.pickup_location[1],
            # dropoff_location可以单独编码或合并
        ], dtype=np.float32)
    
    def get_distance(self) -> float:
        """计算取货点到送货点的距离（曼哈顿距离）"""
        return abs(self.pickup_location[0] - self.dropoff_location[0]) + \
               abs(self.pickup_location[1] - self.dropoff_location[1])
    
    def is_valid(self, current_time: int) -> bool:
        """检查任务是否仍然有效"""
        if self.status == TaskStatus.COMPLETED:
            return False
        if self.deadline is not None and current_time > self.deadline:
            return False
        return True
