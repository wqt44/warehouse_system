"""
性能评估指标：任务完成率、平均等待时间、系统效率等
"""
import numpy as np
from typing import List, Dict
from agents.task import Task, TaskStatus
from agents.robot import Robot


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.task_completion_times = []
        self.task_waiting_times = []
        self.robot_utilization = []  # 存储每一步的利用率
        self.collision_count = 0
        self.total_rewards = []
        self.episode_lengths = []
    
    def record_task_completion(self, task: Task, completion_time: int):
        """记录任务完成"""
        waiting_time = task.created_time - task.created_time  # 等待时间
        completion_time_total = completion_time - task.created_time
        
        self.task_completion_times.append(completion_time_total)
        self.task_waiting_times.append(waiting_time)
    
    def record_collision(self):
        """记录碰撞"""
        self.collision_count += 1
    
    def record_episode(self, episode_reward: float, episode_length: int):
        """记录episode统计"""
        self.total_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
    
    def compute_metrics(self) -> Dict[str, float]:
        """计算所有指标"""
        metrics = {}
        
        # 任务完成率
        if self.task_completion_times:
            metrics['avg_completion_time'] = np.mean(self.task_completion_times)
            metrics['std_completion_time'] = np.std(self.task_completion_times)
        else:
            metrics['avg_completion_time'] = 0.0
            metrics['std_completion_time'] = 0.0
        
        # 平均等待时间
        if self.task_waiting_times:
            metrics['avg_waiting_time'] = np.mean(self.task_waiting_times)
        else:
            metrics['avg_waiting_time'] = 0.0
        
        # 碰撞次数
        metrics['total_collisions'] = self.collision_count
        
        # 平均奖励
        if self.total_rewards:
            metrics['avg_episode_reward'] = np.mean(self.total_rewards)
            metrics['std_episode_reward'] = np.std(self.total_rewards)
        else:
            metrics['avg_episode_reward'] = 0.0
            metrics['std_episode_reward'] = 0.0
        
        # 平均episode长度
        if self.episode_lengths:
            metrics['avg_episode_length'] = np.mean(self.episode_lengths)
        else:
            metrics['avg_episode_length'] = 0.0
        
        return metrics
    
    def compute_task_metrics(self, tasks: List[Task], current_step: int) -> Dict[str, float]:
        """计算任务相关指标"""
        # 使用.value进行字符串比较，保持与其他代码一致
        pending_tasks = [t for t in tasks if t.status.value == "pending"]
        completed_tasks = [t for t in tasks if t.status.value == "completed"]
        in_progress_tasks = [t for t in tasks if t.status.value == "in_progress"]
        
        total_tasks = len(tasks)
        completion_rate = len(completed_tasks) / total_tasks if total_tasks > 0 else 0.0
        
        # 计算平均等待时间
        waiting_times = []
        for task in pending_tasks + in_progress_tasks:
            waiting_time = current_step - task.created_time
            waiting_times.append(waiting_time)
        
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0.0
        
        return {
            'completion_rate': completion_rate,
            'pending_tasks': len(pending_tasks),
            'completed_tasks': len(completed_tasks),
            'in_progress_tasks': len(in_progress_tasks),
            'avg_waiting_time': avg_waiting_time
        }
    
    def compute_robot_metrics(self, robots: List[Robot], record: bool = False) -> Dict[str, float]:
        """
        计算机人相关指标
        
        Args:
            robots: 机器人列表
            record: 是否记录当前利用率到历史记录中（用于计算平均利用率）
        """
        idle_count = sum(1 for r in robots if r.state.value == "idle")
        busy_count = sum(1 for r in robots if r.state.value != "idle" and 
                        r.state.value != "charging")
        charging_count = sum(1 for r in robots if r.state.value == "charging")
        
        total_robots = len(robots)
        utilization = (busy_count + charging_count) / total_robots if total_robots > 0 else 0.0
        
        # 如果要求记录，将当前利用率添加到历史记录中
        if record:
            self.robot_utilization.append(utilization)
        
        # 平均电量
        avg_battery = np.mean([r.battery / r.max_battery for r in robots])
        
        return {
            'idle_robots': idle_count,
            'busy_robots': busy_count,
            'charging_robots': charging_count,
            'utilization': utilization,  # 当前时刻的利用率
            'avg_utilization': np.mean(self.robot_utilization) if self.robot_utilization else 0.0,  # 平均利用率
            'avg_battery': avg_battery
        }
