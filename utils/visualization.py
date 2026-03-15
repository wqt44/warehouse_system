"""
Pygame可视化：实时渲染仓库状态
"""
import pygame
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from agents.robot import Robot, RobotState
from agents.task import Task, TaskStatus
from config import WarehouseConfig


class WarehouseVisualizer:
    """仓库可视化类"""
    
    def __init__(self, 
                 warehouse_config: WarehouseConfig,
                 width: int = 1000,
                 height: int = 1000,
                 cell_size: int = 20):
        self.warehouse_config = warehouse_config
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("仓储系统多智能体强化学习仿真")
        self.clock = pygame.time.Clock()
        
        # 颜色定义
        self.colors = {
            'empty': (255, 255, 255),      # 白色 - 空地
            'obstacle': (100, 100, 100),   # 灰色 - 障碍物
            'shelf': (139, 69, 19),        # 棕色 - 货架
            'workstation': (0, 100, 200),  # 蓝色 - 工作站
            'charging': (0, 200, 100),     # 绿色 - 充电站
            'robot_idle': (255, 0, 0),    # 红色 - 空闲机器人
            'robot_busy': (255, 165, 0),  # 橙色 - 忙碌机器人
            'robot_charging': (0, 255, 0), # 绿色 - 充电机器人
            'task_pending': (255, 255, 0), # 黄色 - 待分配任务
            'task_assigned': (255, 192, 203), # 粉色 - 已分配任务
            'text': (0, 0, 0),             # 黑色 - 文本
            'background': (240, 240, 240)  # 浅灰色 - 背景
        }
        
        # 字体
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 14)
    
    def draw_grid(self, grid: np.ndarray):
        """绘制网格（工作站、充电站按配置顺序编号）"""
        grid_height, grid_width = grid.shape
        ws_positions = getattr(self.warehouse_config, 'workstation_positions', None) or []
        charge_positions = getattr(self.warehouse_config, 'charging_stations', None) or []
        ws_pos_to_id = {tuple(p): i for i, p in enumerate(ws_positions)}
        charge_pos_to_id = {tuple(p): i for i, p in enumerate(charge_positions)}
        
        for y in range(grid_height):
            for x in range(grid_width):
                cell_value = grid[y, x]
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                if cell_value == 0:
                    color = self.colors['empty']
                elif cell_value == 1:
                    color = self.colors['obstacle']
                elif cell_value == 2:
                    color = self.colors['shelf']
                elif cell_value == 3:
                    color = self.colors['workstation']
                elif cell_value == 4:
                    color = self.colors['charging']
                else:
                    color = self.colors['empty']
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
                if cell_value == 3:
                    wid = ws_pos_to_id.get((x, y))
                    if wid is not None:
                        num_surf = self.small_font.render(str(wid), True, (255, 255, 255))
                        num_surf.set_alpha(220)
                        tr = num_surf.get_rect(center=rect.center)
                        self.screen.blit(num_surf, tr)
                if cell_value == 4:
                    cid = charge_pos_to_id.get((x, y))
                    if cid is not None:
                        num_surf = self.small_font.render(str(cid), True, (255, 255, 255))
                        num_surf.set_alpha(220)
                        tr = num_surf.get_rect(center=rect.center)
                        self.screen.blit(num_surf, tr)
    
    def draw_robots(self, robots: List[Robot]):
        """绘制机器人"""
        for robot in robots:
            x, y = robot.position
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            
            # 根据状态选择颜色
            if robot.state == RobotState.CHARGING:
                color = self.colors['robot_charging']
            elif robot.state == RobotState.IDLE:
                color = self.colors['robot_idle']
            else:
                color = self.colors['robot_busy']
            
            # 绘制机器人（圆形）
            radius = self.cell_size // 2 - 2
            pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
            
            # 绘制机器人ID
            text = self.small_font.render(str(robot.robot_id), True, self.colors['text'])
            text_rect = text.get_rect(center=(center_x, center_y))
            self.screen.blit(text, text_rect)
            
            # 绘制电量条
            battery_ratio = robot.battery / robot.max_battery
            bar_width = self.cell_size - 4
            bar_height = 3
            bar_x = x * self.cell_size + 2
            bar_y = y * self.cell_size - 5
            
            # 背景
            pygame.draw.rect(self.screen, (100, 100, 100), 
                           (bar_x, bar_y, bar_width, bar_height))
            # 电量
            battery_color = (0, 255, 0) if battery_ratio > 0.5 else \
                           (255, 255, 0) if battery_ratio > 0.2 else (255, 0, 0)
            pygame.draw.rect(self.screen, battery_color,
                           (bar_x, bar_y, int(bar_width * battery_ratio), bar_height))
    
    def draw_tasks(self, tasks: List[Task]):
        """绘制任务"""
        for task in tasks:
            if task.status == TaskStatus.COMPLETED:
                continue
            
            # 绘制取货点
            pickup_x, pickup_y = task.pickup_location
            center_x = pickup_x * self.cell_size + self.cell_size // 2
            center_y = pickup_y * self.cell_size + self.cell_size // 2
            
            if task.status == TaskStatus.PENDING:
                color = self.colors['task_pending']
            else:
                color = self.colors['task_assigned']
            
            # 绘制任务标记（三角形）
            points = [
                (center_x, center_y - self.cell_size // 3),
                (center_x - self.cell_size // 3, center_y + self.cell_size // 3),
                (center_x + self.cell_size // 3, center_y + self.cell_size // 3)
            ]
            pygame.draw.polygon(self.screen, color, points)
            
            # 绘制到送货点的连线
            dropoff_x, dropoff_y = task.dropoff_location
            dropoff_center_x = dropoff_x * self.cell_size + self.cell_size // 2
            dropoff_center_y = dropoff_y * self.cell_size + self.cell_size // 2
            
            pygame.draw.line(self.screen, color, 
                           (center_x, center_y),
                           (dropoff_center_x, dropoff_center_y), 2)
    
    def draw_info(self, step: int, episode: int, rewards: Optional[Dict[int, float]] = None,
                 stage_info: Optional[Dict[str, Any]] = None):
        """绘制信息面板"""
        y_offset = 10
        x_offset = self.warehouse_config.width * self.cell_size + 20
        
        # Episode信息
        text = self.font.render(f"Episode: {episode}", True, self.colors['text'])
        self.screen.blit(text, (x_offset, y_offset))
        y_offset += 30
        
        # Step信息
        text = self.font.render(f"Step: {step}", True, self.colors['text'])
        self.screen.blit(text, (x_offset, y_offset))
        y_offset += 30
        
        # 课程学习阶段信息
        if stage_info:
            text = self.font.render(f"Stage: {stage_info['stage_idx']+1}/{stage_info['total_stages']}", 
                                  True, self.colors['text'])
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += 30
            
            text = self.font.render(f"Robots: {stage_info['num_robots']}", 
                                  True, self.colors['text'])
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += 30
            
            text = self.font.render(f"Tasks: {stage_info['num_tasks']}", 
                                  True, self.colors['text'])
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += 30
        
        # 奖励信息
        if rewards:
            avg_reward = sum(rewards.values()) / len(rewards) if rewards else 0
            text = self.font.render(f"Avg Reward: {avg_reward:.2f}", 
                                  True, self.colors['text'])
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += 30
    
    def render(self, grid: np.ndarray, robots: List[Robot], tasks: List[Task],
               step: int, episode: int, rewards: Optional[Dict[int, float]] = None,
               stage_info: Optional[Dict[str, Any]] = None, fps: int = 10):
        """
        渲染一帧
        
        Args:
            grid: 仓库网格
            robots: 机器人列表
            tasks: 任务列表
            step: 当前步数
            episode: 当前episode
            rewards: 奖励字典
            stage_info: 课程学习阶段信息
            fps: 帧率
        """
        # 清空屏幕
        self.screen.fill(self.colors['background'])
        
        # 绘制网格
        self.draw_grid(grid)
        
        # 绘制任务
        self.draw_tasks(tasks)
        
        # 绘制机器人
        self.draw_robots(robots)
        
        # 绘制信息
        self.draw_info(step, episode, rewards, stage_info)
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(fps)
    
    def handle_events(self) -> bool:
        """
        处理事件
        
        Returns:
            是否继续运行
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def close(self):
        """关闭可视化"""
        pygame.quit()
