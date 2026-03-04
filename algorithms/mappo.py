"""
MAPPO (Multi-Agent Proximal Policy Optimization) 算法实现
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import copy

from models.policy_networks import ActorCritic
from config import TrainingConfig


class MAPPO:
    """MAPPO算法类"""
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 num_agents: int,
                 config: TrainingConfig,
                 device: str = 'cpu'):
        self.config = config
        self.num_agents = num_agents
        self.device = device
        
        # 共享策略网络（参数共享）
        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # 经验缓冲区
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # 训练统计
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
    
    def select_actions(self, observations: Dict[int, np.ndarray], 
                      deterministic: bool = False) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
        """
        选择动作
        
        Returns:
            (actions, log_probs, values)
        """
        actions = {}
        log_probs = {}
        values = {}
        
        self.policy.eval()
        with torch.no_grad():
            for agent_id, obs in observations.items():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                dist, value = self.policy(obs_tensor)
                
                if deterministic:
                    action = dist.probs.argmax(dim=-1).item()
                    log_prob = dist.log_prob(torch.tensor(action)).item()
                else:
                    action = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor(action)).item()
                
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
                values[agent_id] = value.item()
        
        return actions, log_probs, values
    
    def store_transition(self, obs: Dict[int, np.ndarray],
                        actions: Dict[int, int],
                        rewards: Dict[int, float],
                        values: Dict[int, float],
                        log_probs: Dict[int, float],
                        dones: Dict[int, bool]):
        """存储经验"""
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(actions)
        self.buffer['rewards'].append(rewards)
        self.buffer['values'].append(values)
        self.buffer['log_probs'].append(log_probs)
        self.buffer['dones'].append(dones)
    
    def compute_gae(self, rewards: List[Dict[int, float]],
                   values: List[Dict[int, float]],
                   dones: List[Dict[int, bool]],
                   next_value: Optional[Dict[int, float]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        计算GAE (Generalized Advantage Estimation)
        
        Returns:
            (advantages, returns)
        """
        advantages = []
        returns = []
        
        num_steps = len(rewards)
        num_agents = len(rewards[0]) if rewards else 0
        
        # 初始化
        if next_value is None:
            next_value = {i: 0.0 for i in range(num_agents)}
        
        gae = {i: 0.0 for i in range(num_agents)}
        
        # 从后往前计算
        for step in reversed(range(num_steps)):
            step_advantages = {}
            step_returns = {}
            
            for agent_id in range(num_agents):
                if agent_id not in rewards[step]:
                    continue
                
                reward = rewards[step][agent_id]
                value = values[step][agent_id]
                done = dones[step].get(agent_id, False)
                next_val = next_value.get(agent_id, 0.0) if step == num_steps - 1 else values[step + 1].get(agent_id, 0.0)
                
                # TD误差
                delta = reward + self.config.gamma * next_val * (1 - done) - value
                
                # GAE
                gae[agent_id] = delta + self.config.gamma * self.config.gae_lambda * (1 - done) * gae[agent_id]
                
                step_advantages[agent_id] = gae[agent_id]
                step_returns[agent_id] = gae[agent_id] + value
            
            advantages.insert(0, step_advantages)
            returns.insert(0, step_returns)
        
        return advantages, returns
    
    def update(self):
        """更新策略"""
        if len(self.buffer['obs']) < self.config.batch_size:
            return
        
        # 准备数据
        num_steps = len(self.buffer['obs'])
        num_agents = len(self.buffer['obs'][0])
        
        # 计算GAE
        advantages, returns = self.compute_gae(
            self.buffer['rewards'],
            self.buffer['values'],
            self.buffer['dones']
        )
        
        # 展平数据
        obs_batch = []
        action_batch = []
        old_log_prob_batch = []
        advantage_batch = []
        return_batch = []
        
        for step in range(num_steps):
            for agent_id in range(num_agents):
                if agent_id not in self.buffer['obs'][step]:
                    continue
                obs_batch.append(self.buffer['obs'][step][agent_id])
                action_batch.append(self.buffer['actions'][step][agent_id])
                old_log_prob_batch.append(self.buffer['log_probs'][step][agent_id])
                advantage_batch.append(advantages[step].get(agent_id, 0.0))
                return_batch.append(returns[step].get(agent_id, 0.0))
        
        # 转换为tensor
        obs_tensor = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        old_log_prob_tensor = torch.FloatTensor(old_log_prob_batch).to(self.device)
        advantage_tensor = torch.FloatTensor(advantage_batch).to(self.device)
        return_tensor = torch.FloatTensor(return_batch).to(self.device)
        
        # 归一化优势
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)
        
        # 训练多个epoch
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.config.num_epochs):
            # 随机打乱
            indices = torch.randperm(len(obs_batch)).to(self.device)
            
            for i in range(0, len(obs_batch), self.config.batch_size):
                batch_indices = indices[i:i+self.config.batch_size]
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                batch_old_log_probs = old_log_prob_tensor[batch_indices]
                batch_advantages = advantage_tensor[batch_indices]
                batch_returns = return_tensor[batch_indices]
                
                # 前向传播
                dist, values = self.policy(batch_obs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算损失
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                  1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # 总损失
                loss = policy_loss + self.config.value_loss_coef * value_loss - \
                      self.config.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # 记录统计信息
        num_updates = self.config.num_epochs * (len(obs_batch) // self.config.batch_size + 1)
        self.training_stats['policy_loss'].append(total_policy_loss / num_updates)
        self.training_stats['value_loss'].append(total_value_loss / num_updates)
        self.training_stats['entropy'].append(total_entropy / num_updates)
        self.training_stats['total_loss'].append(
            (total_policy_loss + total_value_loss) / num_updates
        )
        
        # 清空缓冲区
        self.clear_buffer()
    
    def clear_buffer(self):
        """清空经验缓冲区"""
        for key in self.buffer:
            self.buffer[key].clear()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
