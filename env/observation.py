"""
观察空间定义：融合机器人自身状态、局部环境信息和全局任务队列
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from config import ObservationConfig, WarehouseConfig
from agents.robot import Robot
from agents.task import Task


class ObservationSpace:
    """观察空间类"""
    
    def __init__(self, config: ObservationConfig, warehouse_config: WarehouseConfig):
        self.config = config
        self.warehouse_config = warehouse_config
        self.obs_size = config.local_obs_size
        self.half_obs = self.obs_size // 2
    
    def get_observation(self, robot: Robot, robots: List[Robot], 
                       tasks: List[Task], grid: np.ndarray) -> np.ndarray:
        """
        获取单个机器人的观察
        
        Args:
            robot: 当前机器人
            robots: 所有机器人列表
            tasks: 所有待分配任务列表
            grid: 仓库网格（0=空地, 1=障碍物, 2=货架, 3=工作站, 4=充电站）
        
        Returns:
            观察向量
        """
        # 1. 自身状态
        self_state = robot.get_state_vector()
        
        # 2. 局部环境观察（周围网格）
        local_obs = self._get_local_observation(robot, robots, grid)
        
        # 3. 全局任务队列信息
        task_queue_obs = self._get_task_queue_observation(tasks, robot)
        
        # 拼接所有观察
        observation = np.concatenate([
            self_state,
            local_obs.flatten(),
            task_queue_obs.flatten()
        ])
        
        return observation.astype(np.float32)
    
    def _get_local_observation(self, robot: Robot, robots: List[Robot], 
                               grid: np.ndarray) -> np.ndarray:
        """获取局部环境观察（11x11网格，4通道）"""
        x, y = robot.position
        obs = np.zeros((self.obs_size, self.obs_size, self.config.local_obs_channels), 
                       dtype=np.float32)
        
        # 遍历观察范围内的每个位置
        for i in range(self.obs_size):
            for j in range(self.obs_size):
                # 计算全局坐标
                global_x = x - self.half_obs + i
                global_y = y - self.half_obs + j
                
                # 检查是否在边界内
                if (0 <= global_x < self.warehouse_config.width and 
                    0 <= global_y < self.warehouse_config.height):
                    
                    # 通道0: 障碍物
                    if grid[global_y, global_x] == 1:
                        obs[i, j, 0] = 1.0
                    
                    # 通道1: 其他机器人
                    for other_robot in robots:
                        if (other_robot.robot_id != robot.robot_id and 
                            other_robot.position == (global_x, global_y)):
                            obs[i, j, 1] = 1.0
                            break
                    
                    # 通道2: 货架
                    if grid[global_y, global_x] == 2:
                        obs[i, j, 2] = 1.0
                    
                    # 通道3: 工作站/充电站
                    if grid[global_y, global_x] in [3, 4]:
                        obs[i, j, 3] = 1.0
                else:
                    # 边界外视为障碍物
                    obs[i, j, 0] = 1.0
        
        return obs
    
    def _get_task_queue_observation(self, tasks: List[Task], robot: Robot) -> np.ndarray:
        """获取任务队列观察"""
        # 只考虑待分配的任务
        pending_tasks = [t for t in tasks if t.status.value == "pending"]
        
        # 按优先级排序
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # 限制任务数量
        pending_tasks = pending_tasks[:self.config.max_task_queue_size]
        
        # 构建任务特征矩阵
        task_features = np.zeros((self.config.max_task_queue_size, 
                                  self.config.task_feature_dim), dtype=np.float32)
        
        for idx, task in enumerate(pending_tasks):
            # 计算到任务的距离（归一化）
            dist_to_pickup = robot.get_distance_to(task.pickup_location) / 100.0
            
            task_features[idx] = [
                task.task_id / 1000.0,  # 归一化任务ID
                task.priority,
                task.pickup_location[0] / self.warehouse_config.width,
                task.pickup_location[1] / self.warehouse_config.height,
            ]
        
        return task_features
    
    def get_global_state(self, robots: List[Robot], tasks: List[Task]) -> Dict[str, Any]:
        """
        获取全局状态（用于任务分配网络）
        
        Returns:
            全局状态字典
        """
        # 机器人状态
        robot_states = []
        for robot in robots:
            robot_states.append(robot.get_state_vector())
        
        # 填充到固定大小（用于批处理）
        max_robots = 50
        if len(robot_states) < max_robots:
            robot_state_dim = len(robot_states[0]) if robot_states else 7
            robot_states.extend([np.zeros(robot_state_dim, dtype=np.float32)] * 
                              (max_robots - len(robot_states)))
        robot_states = np.array(robot_states[:max_robots])
        
        # 任务状态
        task_states = []
        for task in tasks:
            if task.status.value == "pending":
                task_vec = np.array([
                    task.task_id / 1000.0,
                    task.priority,
                    task.pickup_location[0] / self.warehouse_config.width,
                    task.pickup_location[1] / self.warehouse_config.height,
                    task.dropoff_location[0] / self.warehouse_config.width,
                    task.dropoff_location[1] / self.warehouse_config.height,
                ], dtype=np.float32)
                task_states.append(task_vec)
        
        # 填充到固定大小
        max_tasks = self.config.max_task_queue_size
        if len(task_states) < max_tasks:
            task_states.extend([np.zeros(6, dtype=np.float32)] * 
                             (max_tasks - len(task_states)))
        task_states = np.array(task_states[:max_tasks])
        
        return {
            'robot_states': robot_states,
            'task_states': task_states,
            'num_robots': len(robots),
            'num_tasks': len([t for t in tasks if t.status.value == "pending"])
        }
