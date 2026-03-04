"""
算法模块
"""
from algorithms.mappo import MAPPO
from algorithms.curriculum import CurriculumLearning, CurriculumStage, create_default_curriculum

__all__ = ['MAPPO', 'CurriculumLearning', 'CurriculumStage', 'create_default_curriculum']
