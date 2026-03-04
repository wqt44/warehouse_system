"""
奖励函数：任务完成奖励、距离惩罚、碰撞惩罚、等待时间惩罚等
"""
import numpy as np
from typing import List, Tuple, Dict
from agents.robot import Robot
from agents.task import Task
from config import RewardConfig


class RewardFunction:
    """奖励函数类"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.last_positions: Dict[int, Tuple[int, int]] = {}
        self.waiting_times: Dict[int, int] = {}  # 任务等待时间
        self.collision_history: Dict[int, int] = {}  # 碰撞历史
        # 效率相关统计
        self.last_completed_tasks: int = 0  # 上一次统计时已完成的任务数
        self.last_step: int = -1           # 上一次计算效率奖励的时间步
        self.last_efficiency_reward: float = 0.0  # 当前时间步的效率奖励（按机器人平均分摊）
    
    def compute_reward(self, robot: Robot, robots: List[Robot], 
                      tasks: List[Task], step: int) -> Tuple[float, Dict]:
        """
        计算单个机器人的奖励
        
        Returns:
            (奖励值, 奖励分解字典)
        """
        reward = 0.0
        reward_breakdown = {}
        
        # 0. 单步效率奖励（单位时间完成任务数）
        self._update_efficiency_reward(tasks, robots, step)
        reward += self.last_efficiency_reward
        reward_breakdown['efficiency'] = self.last_efficiency_reward
        
        # 1. 任务完成奖励（个体维度）
        task_reward = self._compute_task_reward(robot, tasks)
        reward += task_reward
        reward_breakdown['task_completion'] = task_reward
        
        # 2. 距离惩罚（鼓励高效移动）
        distance_penalty = self._compute_distance_penalty(robot)
        reward += distance_penalty
        reward_breakdown['distance'] = distance_penalty
        
        # 3. 碰撞惩罚
        collision_penalty = self._compute_collision_penalty(robot, robots)
        reward += collision_penalty
        reward_breakdown['collision'] = collision_penalty
        
        # 4. 等待时间惩罚（任务等待时间过长）——按机器人平均分摊，避免每步累计过大
        waiting_penalty = self._compute_waiting_penalty(tasks, step, num_robots=max(1, len(robots)))
        reward += waiting_penalty
        reward_breakdown['waiting'] = waiting_penalty
        
        # 5. 低电量惩罚
        battery_penalty = self._compute_battery_penalty(robot)
        reward += battery_penalty
        reward_breakdown['battery'] = battery_penalty
        
        # 6. 内在奖励（鼓励探索）
        intrinsic_reward = self._compute_intrinsic_reward(robot)
        reward += intrinsic_reward * self.config.intrinsic_reward_coef
        reward_breakdown['intrinsic'] = intrinsic_reward
        
        return reward, reward_breakdown
    
    def _update_efficiency_reward(self, tasks: List[Task], robots: List[Robot], step: int) -> None:
        """
        计算单位时间完成任务数对应的效率奖励（全局维度）
        在一个时间步内只计算一次，并按机器人平均分摊到每个智能体上。
        """
        if step == self.last_step:
            # 本时间步已经计算过
            return
        
        completed_now = sum(1 for t in tasks if t.status.value == "completed")
        delta_completed = max(0, completed_now - self.last_completed_tasks)
        num_robots = max(1, len(robots))
        
        # 每完成一个任务，给所有机器人共享一份正奖励
        self.last_efficiency_reward = (
            self.config.efficiency_reward_coef * delta_completed / num_robots
        )
        self.last_completed_tasks = completed_now
        self.last_step = step
    
    def _compute_task_reward(self, robot: Robot, tasks: List[Task]) -> float:
        """计算任务完成奖励"""
        # 检查是否“本步刚完成任务”
        if getattr(robot, "just_completed_task", False):
            return self.config.task_completion_reward
        return 0.0
    
    def _compute_distance_penalty(self, robot: Robot) -> float:
        """计算距离惩罚（鼓励移动）"""
        if robot.robot_id in self.last_positions:
            last_pos = self.last_positions[robot.robot_id]
            # 如果位置没变，给予小惩罚
            if robot.position == last_pos:
                # 使用单独的无移动惩罚缩放系数，减弱对原地等待的惩罚
                return self.config.distance_penalty * self.config.idle_penalty_coef
            # 移动距离奖励（鼓励高效移动）
            dist = abs(robot.position[0] - last_pos[0]) + \
                   abs(robot.position[1] - last_pos[1])
            return self.config.distance_penalty * dist
        
        self.last_positions[robot.robot_id] = robot.position
        return 0.0
    
    def _compute_collision_penalty(self, robot: Robot, robots: List[Robot]) -> float:
        """计算碰撞惩罚"""
        penalty = 0.0
        collision_count = 0
        
        for other_robot in robots:
            if (other_robot.robot_id != robot.robot_id and 
                other_robot.position == robot.position):
                collision_count += 1
        
        if collision_count > 0:
            penalty = self.config.collision_penalty * collision_count
            self.collision_history[robot.robot_id] = \
                self.collision_history.get(robot.robot_id, 0) + 1
        
        return penalty
    
    def _compute_waiting_penalty(self, tasks: List[Task], step: int, num_robots: int) -> float:
        """计算等待时间惩罚"""
        penalty = 0.0
        # 使用配置中的等待阈值（默认5步，更早触发惩罚）
        threshold = getattr(self.config, 'waiting_threshold', 5)
        
        for task in tasks:
            if task.status.value == "pending":
                waiting_time = step - task.created_time
                if waiting_time > threshold:  # 等待超过阈值
                    penalty += self.config.waiting_penalty * (waiting_time - threshold)
        
        # 关键：每步等待惩罚不要"按任务数堆叠"后再给每个机器人完整承受
        # 这里改为：按机器人平均分摊，并按任务数做一次归一化，防止 500 步时爆炸式累计
        pending_cnt = max(1, len([t for t in tasks if t.status.value == "pending"]))
        per_robot = penalty / pending_cnt / max(1, num_robots)
        # 额外保护：单步等待惩罚下限（避免极端长等待把梯度打爆）
        return max(per_robot, -3.0)  # 从-2.0调整到-3.0，允许稍大的等待惩罚
    
    def _compute_battery_penalty(self, robot: Robot) -> float:
        """计算低电量惩罚"""
        battery_ratio = robot.battery / robot.max_battery
        if battery_ratio < 0.2:
            return self.config.battery_low_penalty * (0.2 - battery_ratio)
        return 0.0
    
    def _compute_intrinsic_reward(self, robot: Robot) -> float:
        """计算内在奖励（鼓励探索新区域）"""
        # 简单的基于位置访问频率的内在奖励
        # 这里可以扩展为更复杂的探索奖励机制
        return 0.0  # 暂时返回0，可以在后续扩展
    
    def reset(self):
        """重置奖励函数状态"""
        self.last_positions.clear()
        self.waiting_times.clear()
        self.collision_history.clear()
    
    def update_positions(self, robots: List[Robot]):
        """更新位置历史"""
        for robot in robots:
            self.last_positions[robot.robot_id] = robot.position
