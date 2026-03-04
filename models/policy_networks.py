"""
策略网络：多智能体策略网络（参数共享）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class PolicyNetwork(nn.Module):
    """策略网络（Actor）"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 特征提取（处理高维观察）
        # 如果观察包含图像特征，使用CNN
        if obs_dim > 1000:  # 假设包含局部观察图像
            # 局部观察部分（假设是11x11x4的图像）
            self.local_conv = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten()
            )
            local_feat_dim = 64 * 4 * 4
            # 其他特征维度
            other_feat_dim = obs_dim - (11 * 11 * 4)
            self.other_linear = nn.Linear(other_feat_dim, hidden_dim // 2)
            self.fc = nn.Linear(local_feat_dim + hidden_dim // 2, hidden_dim)
        else:
            # 纯全连接网络
            self.fc = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """
        前向传播
        
        Returns:
            Categorical分布
        """
        if hasattr(self, 'local_conv'):
            # 分离局部观察和其他特征
            batch_size = obs.size(0)
            local_obs = obs[:, :11*11*4].view(batch_size, 4, 11, 11)
            other_obs = obs[:, 11*11*4:]
            
            local_feat = self.local_conv(local_obs)
            other_feat = self.other_linear(other_obs)
            feat = torch.cat([local_feat, other_feat], dim=-1)
            x = self.fc(feat)
        else:
            x = self.fc(obs)
        
        logits = self.policy_head(x)
        dist = torch.distributions.Categorical(logits=logits)
        
        return dist
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        采样动作
        
        Returns:
            (action, log_prob)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        dist = self.forward(obs_tensor)
        
        if deterministic:
            action = dist.probs.argmax(dim=-1).item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        else:
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """价值网络（Critic）"""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.obs_dim = obs_dim
        
        # 特征提取（与策略网络类似）
        if obs_dim > 1000:
            self.local_conv = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten()
            )
            local_feat_dim = 64 * 4 * 4
            other_feat_dim = obs_dim - (11 * 11 * 4)
            self.other_linear = nn.Linear(other_feat_dim, hidden_dim // 2)
            self.fc = nn.Linear(local_feat_dim + hidden_dim // 2, hidden_dim)
        else:
            self.fc = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if hasattr(self, 'local_conv'):
            batch_size = obs.size(0)
            local_obs = obs[:, :11*11*4].view(batch_size, 4, 11, 11)
            other_obs = obs[:, 11*11*4:]
            
            local_feat = self.local_conv(local_obs)
            other_feat = self.other_linear(other_obs)
            feat = torch.cat([local_feat, other_feat], dim=-1)
            x = self.fc(feat)
        else:
            x = self.fc(obs)
        
        value = self.value_head(x)
        return value


class ActorCritic(nn.Module):
    """Actor-Critic网络（共享特征提取）"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 共享特征提取
        if obs_dim > 1000:
            self.local_conv = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten()
            )
            local_feat_dim = 64 * 4 * 4
            other_feat_dim = obs_dim - (11 * 11 * 4)
            self.other_linear = nn.Linear(other_feat_dim, hidden_dim // 2)
            self.shared_fc = nn.Linear(local_feat_dim + hidden_dim // 2, hidden_dim)
        else:
            self.shared_fc = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        前向传播
        
        Returns:
            (policy_dist, value)
        """
        if hasattr(self, 'local_conv'):
            batch_size = obs.size(0)
            local_obs = obs[:, :11*11*4].view(batch_size, 4, 11, 11)
            other_obs = obs[:, 11*11*4:]
            
            local_feat = self.local_conv(local_obs)
            other_feat = self.other_linear(other_obs)
            feat = torch.cat([local_feat, other_feat], dim=-1)
            x = self.shared_fc(feat)
        else:
            x = self.shared_fc(obs)
        
        logits = self.policy_head(x)
        value = self.value_head(x)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        return dist, value
