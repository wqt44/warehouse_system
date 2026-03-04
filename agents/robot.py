"""
机器人智能体：定义机器人的状态、动作和行为
"""
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum
import numpy as np
from agents.task import Task, TaskStatus


class RobotState(Enum):
    """机器人状态"""
    IDLE = "idle"              # 空闲
    MOVING_TO_PICKUP = "moving_to_pickup"  # 前往取货点
    PICKING_UP = "picking_up"  # 取货中
    MOVING_TO_DROPOFF = "moving_to_dropoff"  # 前往送货点
    DROPPING_OFF = "dropping_off"  # 送货中
    CHARGING = "charging"      # 充电中
    RETURNING_TO_CHARGE = "returning_to_charge"  # 返回充电站


@dataclass
class Robot:
    """机器人类"""
    robot_id: int
    position: Tuple[int, int]
    battery: float
    max_battery: float
    state: RobotState = RobotState.IDLE
    current_task: Optional[Task] = None
    task_progress: float = 0.0  # 任务进度 0-1
    direction: int = 0  # 方向：0=上, 1=右, 2=下, 3=左
    velocity: Tuple[int, int] = (0, 0)  # 速度向量
    # 任务完成事件（用于奖励：在“完成那一步”给正奖励）
    just_completed_task: bool = False
    last_completed_task_id: Optional[int] = None
    
    def __post_init__(self):
        if self.battery is None:
            self.battery = self.max_battery
    
    def get_state_vector(self) -> np.ndarray:
        """获取机器人状态向量"""
        return np.array([
            self.position[0] / 50.0,  # 归一化位置
            self.position[1] / 50.0,
            self.battery / self.max_battery,  # 归一化电量
            self.state.value == RobotState.IDLE.value,  # 是否空闲
            self.state.value == RobotState.CHARGING.value,  # 是否充电
            self.task_progress,  # 任务进度
            self.direction / 4.0,  # 归一化方向
        ], dtype=np.float32)
    
    def consume_battery(self, consumption_rate: float) -> bool:
        """消耗电量，返回是否电量耗尽"""
        if self.state != RobotState.CHARGING:
            self.battery = max(0.0, self.battery - consumption_rate)
            return self.battery <= 0.0
        return False
    
    def charge(self, charging_rate: float):
        """充电"""
        if self.state == RobotState.CHARGING:
            self.battery = min(self.max_battery, self.battery + charging_rate)
    
    def needs_charging(self, low_battery_threshold: float = 0.2) -> bool:
        """检查是否需要充电"""
        return self.battery / self.max_battery < low_battery_threshold
    
    def update_position(self, new_position: Tuple[int, int]):
        """更新位置"""
        self.position = new_position
    
    def assign_task(self, task: Task):
        """分配任务"""
        self.current_task = task
        self.state = RobotState.MOVING_TO_PICKUP
        self.task_progress = 0.0
        if task:
            task.status = TaskStatus.ASSIGNED
            task.assigned_robot_id = self.robot_id
    
    def complete_task(self):
        """完成任务"""
        # 标记"本步刚完成任务"，用于奖励函数读取
        self.just_completed_task = False
        self.last_completed_task_id = None
        if self.current_task:
            self.current_task.status = TaskStatus.COMPLETED
            self.just_completed_task = True
            self.last_completed_task_id = self.current_task.task_id
        self.current_task = None
        self.state = RobotState.IDLE
        self.task_progress = 0.0
    
    def cancel_task(self):
        """取消/归还任务（例如电量不足时）"""
        if self.current_task:
            # 将任务状态重置为PENDING，使其可以被重新分配
            self.current_task.status = TaskStatus.PENDING
            self.current_task.assigned_robot_id = None
            self.current_task = None
        self.task_progress = 0.0
        # 注意：状态会在调用此方法后由环境设置为RETURNING_TO_CHARGE或CHARGING
    
    def get_distance_to(self, target: Tuple[int, int]) -> float:
        """计算到目标位置的曼哈顿距离"""
        return abs(self.position[0] - target[0]) + abs(self.position[1] - target[1])
    
    def is_at_position(self, pos: Tuple[int, int], tolerance: int = 0) -> bool:
        """检查是否在指定位置"""
        return abs(self.position[0] - pos[0]) <= tolerance and \
               abs(self.position[1] - pos[1]) <= tolerance
