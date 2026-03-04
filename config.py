"""
配置文件：定义仓库环境、智能体和训练的超参数
"""
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class WarehouseConfig:
    """仓库环境配置"""
    # 网格尺寸
    width: int = 50
    height: int = 50
    
    # 区域配置
    shelf_regions: List[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    workstation_positions: List[Tuple[int, int]] = None
    obstacle_positions: List[Tuple[int, int]] = None
    charging_stations: List[Tuple[int, int]] = None  # 充电站位置
    
    # 任务生成
    task_spawn_rate: float = 0.1  # 每个时间步生成任务的概率
    max_tasks: int = 20
    
    def __post_init__(self):
        if self.shelf_regions is None:
            # 默认货架区域：左侧和右侧
            self.shelf_regions = [
                (5, 5, 20, 45),   # 左侧货架区
                (30, 5, 45, 45)   # 右侧货架区
            ]
        if self.workstation_positions is None:
            # 默认工作站：底部中央
            self.workstation_positions = [(25, 48), (24, 48), (26, 48)]
        if self.obstacle_positions is None:
            self.obstacle_positions = []
        # 如果未指定充电站，将在工作站附近自动生成
        # charging_stations = None 表示自动生成


@dataclass
class RobotConfig:
    """机器人配置"""
    num_robots: int = 5
    max_battery: float = 100.0
    battery_consumption_rate: float = 0.1  # 每步消耗
    charging_rate: float = 2.0  # 充电速率
    max_speed: int = 1  # 每步最大移动距离
    observation_range: int = 5  # 观察范围（曼哈顿距离）
    max_steps_per_episode: int = 300  # Episode最大步数（与TrainingConfig保持一致）


@dataclass
class ObservationConfig:
    """观察空间配置"""
    # 自身状态维度
    self_state_dim: int = 7  # x, y, battery, is_idle, is_charging, task_progress, direction
    
    # 局部环境维度
    local_obs_size: int = 11  # 周围11x11网格的信息
    local_obs_channels: int = 4  # 障碍物、其他机器人、货架、工作站
    
    # 全局任务队列维度
    max_task_queue_size: int = 20
    task_feature_dim: int = 4  # task_id, priority, pickup_loc, dropoff_loc
    
    @property
    def total_dim(self) -> int:
        """总观察维度"""
        return (self.self_state_dim + 
                self.local_obs_size * self.local_obs_size * self.local_obs_channels +
                self.max_task_queue_size * self.task_feature_dim)


@dataclass
class RewardConfig:
    """奖励函数配置（训练版：平滑学习，避免奖励爆炸）"""
    # 任务完成正奖励（适中，鼓励完成任务但不过度）
    task_completion_reward: float = 50.0
    
    # 移动相关惩罚（很小，鼓励探索）
    distance_penalty: float = -0.0005  # 进一步减弱
    idle_penalty_coef: float = 0.2  # 原地等待惩罚很轻
    
    # 碰撞惩罚（保持适中）
    collision_penalty: float = -5.0
    
    # 等待时间惩罚（不过度，主要提示不要一直拖）
    waiting_penalty: float = -0.02  # 回到较温和的值
    waiting_threshold: int = 10  # 超过10步才开始罚
    
    # 低电量惩罚
    battery_low_penalty: float = -0.5
    
    # 内在奖励系数（探索奖励）
    intrinsic_reward_coef: float = 0.1
    
    # 效率奖励系数（适中，配合任务完成奖励）
    efficiency_reward_coef: float = 8.0
    
    # Episode未完成惩罚（训练阶段先调小，避免梯度爆炸）
    episode_incomplete_penalty: float = -100.0  # 从-500降低到-100，训练稳定后可逐步加大


@dataclass
class TrainingConfig:
    """训练配置"""
    # 算法参数
    algorithm: str = "MAPPO"  # MAPPO, PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # 训练参数（优化：缩短训练时间）
    num_episodes: int = 3000  # 从10000减少到3000
    max_steps_per_episode: int = 300  # 从500减少到300，加快训练
    batch_size: int = 64
    num_epochs: int = 10
    save_interval: int = 100  # 可以改为50以更频繁查看进度
    
    # 课程学习
    use_curriculum: bool = True
    curriculum_stages: List[dict] = None
    
    # 分布式训练
    num_parallel_envs: int = 4
    
    def __post_init__(self):
        if self.curriculum_stages is None:
            # 简化课程学习阶段，加快训练速度
            self.curriculum_stages = [
                {"num_robots": 2, "num_tasks": 5, "episodes": 400},   # 从1000减少到400
                {"num_robots": 5, "num_tasks": 10, "episodes": 800},  # 从2000减少到800
                {"num_robots": 10, "num_tasks": 20, "episodes": 1000}, # 从3000减少到1000
                {"num_robots": 20, "num_tasks": 30, "episodes": 800},  # 从4000减少到800
            ]


@dataclass
class TaskAllocatorConfig:
    """任务分配网络配置"""
    hidden_dim: int = 128
    num_layers: int = 3
    attention_heads: int = 4
    use_attention: bool = True
