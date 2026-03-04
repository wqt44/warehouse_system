"""
深度任务分配网络：以全局状态为输入，输出任务-机器人分配概率矩阵
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from config import TaskAllocatorConfig


class TaskAllocatorNetwork(nn.Module):
    """任务分配网络"""
    
    def __init__(self, 
                 robot_state_dim: int = 7,
                 task_state_dim: int = 6,
                 max_robots: int = 50,
                 max_tasks: int = 20,
                 config: TaskAllocatorConfig = None):
        super().__init__()
        
        self.config = config or TaskAllocatorConfig()
        self.max_robots = max_robots
        self.max_tasks = max_tasks
        self.robot_state_dim = robot_state_dim
        self.task_state_dim = task_state_dim
        
        # 机器人编码器
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU()
        )
        
        # 任务编码器
        self.task_encoder = nn.Sequential(
            nn.Linear(task_state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU()
        )
        
        # 注意力机制（可选）
        if self.config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.config.hidden_dim,
                num_heads=self.config.attention_heads,
                batch_first=True
            )
        
        # 分配网络
        self.allocation_net = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 1)  # 输出分配分数
        )
    
    def forward(self, robot_states: torch.Tensor, task_states: torch.Tensor,
                num_robots: int, num_tasks: int) -> torch.Tensor:
        """
        前向传播
        
        Args:
            robot_states: [batch_size, max_robots, robot_state_dim]
            task_states: [batch_size, max_tasks, task_state_dim]
            num_robots: 实际机器人数量
            num_tasks: 实际任务数量
        
        Returns:
            allocation_matrix: [batch_size, num_tasks, num_robots] 分配概率矩阵
        """
        batch_size = robot_states.size(0)
        
        # 编码机器人状态
        robot_features = self.robot_encoder(robot_states)  # [B, max_robots, hidden_dim]
        robot_features = robot_features[:, :num_robots, :]  # [B, num_robots, hidden_dim]
        
        # 编码任务状态
        task_features = self.task_encoder(task_states)  # [B, max_tasks, hidden_dim]
        task_features = task_features[:, :num_tasks, :]  # [B, num_tasks, hidden_dim]
        
        # 注意力机制（任务关注机器人）
        if self.config.use_attention:
            task_features, _ = self.attention(
                task_features, robot_features, robot_features
            )
        
        # 构建分配矩阵
        # 对每个任务-机器人对计算分配分数
        allocation_scores = []
        for t_idx in range(num_tasks):
            task_feat = task_features[:, t_idx:t_idx+1, :]  # [B, 1, hidden_dim]
            scores = []
            for r_idx in range(num_robots):
                robot_feat = robot_features[:, r_idx:r_idx+1, :]  # [B, 1, hidden_dim]
                # 拼接任务和机器人特征
                combined = torch.cat([task_feat, robot_feat], dim=-1)  # [B, 1, 2*hidden_dim]
                score = self.allocation_net(combined)  # [B, 1, 1]
                scores.append(score.squeeze(-1))  # [B, 1]
            allocation_scores.append(torch.cat(scores, dim=-1))  # [B, num_robots]
        
        allocation_matrix = torch.stack(allocation_scores, dim=1)  # [B, num_tasks, num_robots]
        
        # 应用softmax得到概率分布（每个任务分配给哪个机器人）
        allocation_probs = F.softmax(allocation_matrix, dim=-1)  # [B, num_tasks, num_robots]
        
        return allocation_probs
    
    def sample_allocation(self, robot_states: torch.Tensor, task_states: torch.Tensor,
                         num_robots: int, num_tasks: int, temperature: float = 1.0) -> Dict[int, int]:
        """
        采样分配结果
        
        Returns:
            {task_id: robot_id} 分配映射
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(robot_states, task_states, num_robots, num_tasks)
            probs = probs / temperature
            
            # 采样分配
            allocation = {}
            for t_idx in range(num_tasks):
                task_probs = probs[0, t_idx, :num_robots]  # [num_robots]
                robot_idx = torch.multinomial(task_probs, 1).item()
                allocation[t_idx] = robot_idx
        
        return allocation


class ImprovedTaskAllocator(nn.Module):
    """改进的任务分配网络（使用图神经网络）"""
    
    def __init__(self, 
                 robot_state_dim: int = 7,
                 task_state_dim: int = 6,
                 max_robots: int = 50,
                 max_tasks: int = 20,
                 config: TaskAllocatorConfig = None):
        super().__init__()
        
        self.config = config or TaskAllocatorConfig()
        self.max_robots = max_robots
        self.max_tasks = max_tasks
        
        # 使用Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.hidden_dim * 2,
            batch_first=True
        )
        self.robot_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.task_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 输入投影
        self.robot_proj = nn.Linear(robot_state_dim, self.config.hidden_dim)
        self.task_proj = nn.Linear(task_state_dim, self.config.hidden_dim)
        
        # 分配网络
        self.allocation_net = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )
    
    def forward(self, robot_states: torch.Tensor, task_states: torch.Tensor,
                num_robots: int, num_tasks: int) -> torch.Tensor:
        """前向传播"""
        batch_size = robot_states.size(0)
        
        # 投影到隐藏维度
        robot_emb = self.robot_proj(robot_states[:, :num_robots, :])
        task_emb = self.task_proj(task_states[:, :num_tasks, :])
        
        # Transformer编码
        robot_features = self.robot_encoder(robot_emb)
        task_features = self.task_encoder(task_emb)
        
        # 计算分配分数
        allocation_scores = []
        for t_idx in range(num_tasks):
            task_feat = task_features[:, t_idx:t_idx+1, :]
            scores = []
            for r_idx in range(num_robots):
                robot_feat = robot_features[:, r_idx:r_idx+1, :]
                combined = torch.cat([task_feat, robot_feat], dim=-1)
                score = self.allocation_net(combined)
                scores.append(score.squeeze(-1))
            allocation_scores.append(torch.cat(scores, dim=-1))
        
        allocation_matrix = torch.stack(allocation_scores, dim=1)
        allocation_probs = F.softmax(allocation_matrix, dim=-1)
        
        return allocation_probs
