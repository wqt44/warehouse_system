"""
课程学习：从简单场景逐步过渡到复杂场景
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from config import WarehouseConfig, RobotConfig, TrainingConfig


@dataclass
class CurriculumStage:
    """课程学习阶段"""
    num_robots: int
    num_tasks: int
    episodes: int
    task_spawn_rate: float = 0.1
    warehouse_size: Optional[tuple] = None  # (width, height)
    description: str = ""


class CurriculumLearning:
    """课程学习管理器"""
    
    def __init__(self, stages: List[Dict]):
        """
        初始化课程学习
        
        Args:
            stages: 阶段配置列表，每个阶段包含：
                - num_robots: 机器人数量
                - num_tasks: 最大任务数
                - episodes: 该阶段的训练轮数
                - task_spawn_rate: 任务生成率（可选）
                - warehouse_size: 仓库大小（可选）
        """
        self.stages = []
        for stage_dict in stages:
            stage = CurriculumStage(
                num_robots=stage_dict['num_robots'],
                num_tasks=stage_dict['num_tasks'],
                episodes=stage_dict['episodes'],
                task_spawn_rate=stage_dict.get('task_spawn_rate', 0.1),
                warehouse_size=stage_dict.get('warehouse_size'),
                description=stage_dict.get('description', '')
            )
            self.stages.append(stage)
        
        self.current_stage_idx = 0
        self.episode_count = 0
        self.stage_episode_count = 0
    
    def get_current_config(self) -> Dict[str, Any]:
        """获取当前阶段的配置"""
        if self.current_stage_idx >= len(self.stages):
            # 使用最后一个阶段
            stage = self.stages[-1]
        else:
            stage = self.stages[self.current_stage_idx]
        
        # 构建配置
        config = {
            'num_robots': stage.num_robots,
            'max_tasks': stage.num_tasks,
            'task_spawn_rate': stage.task_spawn_rate,
            'warehouse_width': 50,  # 默认值
            'warehouse_height': 50,  # 默认值
            'max_steps': 300  # 默认值，与TrainingConfig保持一致
        }
        
        if stage.warehouse_size:
            config['warehouse_size'] = stage.warehouse_size
            config['warehouse_width'] = stage.warehouse_size[0]
            config['warehouse_height'] = stage.warehouse_size[1]
        
        return config
    
    def update(self, episodes: int = 1) -> bool:
        """
        更新课程学习状态
        
        Args:
            episodes: 完成的episode数量
        
        Returns:
            是否进入新阶段
        """
        self.episode_count += episodes
        self.stage_episode_count += episodes
        
        if self.current_stage_idx >= len(self.stages):
            return False
        
        current_stage = self.stages[self.current_stage_idx]
        
        # 检查是否应该进入下一阶段
        if self.stage_episode_count >= current_stage.episodes:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.stage_episode_count = 0
                return True
        
        return False
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """获取当前阶段信息"""
        if self.current_stage_idx >= len(self.stages):
            stage = self.stages[-1]
        else:
            stage = self.stages[self.current_stage_idx]
        
        return {
            'stage_idx': self.current_stage_idx,
            'total_stages': len(self.stages),
            'stage_episodes': self.stage_episode_count,
            'stage_total_episodes': stage.episodes,
            'num_robots': stage.num_robots,
            'num_tasks': stage.num_tasks,
            'description': stage.description
        }
    
    def is_complete(self) -> bool:
        """检查课程学习是否完成"""
        return self.current_stage_idx >= len(self.stages) - 1 and \
               self.stage_episode_count >= self.stages[-1].episodes
    
    def reset(self):
        """重置课程学习"""
        self.current_stage_idx = 0
        self.episode_count = 0
        self.stage_episode_count = 0


def create_default_curriculum() -> List[Dict]:
    """创建默认课程学习配置"""
    return [
        {
            'num_robots': 2,
            'num_tasks': 5,
            'episodes': 1000,
            'task_spawn_rate': 0.05,
            'description': '阶段1: 简单场景（2个机器人，5个任务）'
        },
        {
            'num_robots': 5,
            'num_tasks': 10,
            'episodes': 2000,
            'task_spawn_rate': 0.08,
            'description': '阶段2: 中等场景（5个机器人，10个任务）'
        },
        {
            'num_robots': 10,
            'num_tasks': 20,
            'episodes': 3000,
            'task_spawn_rate': 0.1,
            'description': '阶段3: 复杂场景（10个机器人，20个任务）'
        },
        {
            'num_robots': 20,
            'num_tasks': 30,
            'episodes': 4000,
            'task_spawn_rate': 0.12,
            'description': '阶段4: 大规模场景（20个机器人，30个任务）'
        },
        {
            'num_robots': 50,
            'num_tasks': 50,
            'episodes': 5000,
            'task_spawn_rate': 0.15,
            'description': '阶段5: 超大规模场景（50个机器人，50个任务）'
        }
    ]
