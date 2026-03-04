"""
训练脚本：主训练流程
"""
import argparse
import sys
import numpy as np
import torch
from tqdm import tqdm
import os

from config import ObservationConfig, TrainingConfig
from env.warehouse_env import WarehouseEnv
from algorithms.mappo import MAPPO
from algorithms.curriculum import CurriculumLearning, create_default_curriculum
from models.task_allocator import TaskAllocatorNetwork
from utils.visualization import WarehouseVisualizer
from utils.metrics import MetricsCollector
from baselines.optimization import BaselineComparison
from utils.env_utils import create_env
from agents.robot import RobotState
import random


def post_process_actions(actions: dict, env: WarehouseEnv) -> dict:
    """
    后处理动作：应用充完电离开充电站的逻辑
    
    如果机器人在充电站且已充满电（>=90%），修改其动作为尝试离开充电站
    礼让逻辑已经在环境层的 _execute_actions 中实现，这里不需要处理
    """
    processed_actions = actions.copy()
    
    for robot_id, action in actions.items():
        if robot_id >= len(env.robots):
            continue
        
        robot = env.robots[robot_id]
        current_pos = robot.position
        is_at_charging_station = env.grid[current_pos[1], current_pos[0]] == 4
        
        # 如果机器人在充电站且已充满电（>=90%），尝试离开充电站
        if is_at_charging_station and robot.battery >= robot.max_battery * 0.9:
            # 尝试离开充电站：先尝试相邻位置
            adjacent_positions = [
                (current_pos[0] + dx, current_pos[1] + dy)
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
            ]
            random.shuffle(adjacent_positions)
            
            moved = False
            for adj_pos in adjacent_positions:
                adj_x, adj_y = adj_pos
                if (0 <= adj_x < env.width and 0 <= adj_y < env.height and
                    env.grid[adj_y, adj_x] not in (1, 2)):
                    occupied = any(
                        r.position == adj_pos and r.robot_id != robot_id
                        for r in env.robots
                    )
                    if not occupied:
                        dx = adj_pos[0] - current_pos[0]
                        dy = adj_pos[1] - current_pos[1]
                        if dx > 0:
                            processed_actions[robot_id] = 1  # 右
                        elif dx < 0:
                            processed_actions[robot_id] = 3  # 左
                        elif dy > 0:
                            processed_actions[robot_id] = 2  # 下
                        elif dy < 0:
                            processed_actions[robot_id] = 0  # 上
                        moved = True
                        break
            
            # 如果相邻位置都被占用，尝试更远的位置
            if not moved:
                for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0), 
                               (-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    target_x = current_pos[0] + dx
                    target_y = current_pos[1] + dy
                    if (0 <= target_x < env.width and 0 <= target_y < env.height and
                        env.grid[target_y, target_x] not in (1, 2)):
                        occupied = any(
                            r.position == (target_x, target_y) and r.robot_id != robot_id
                            for r in env.robots
                        )
                        if not occupied:
                            # 先移动到中间位置
                            mid_x = current_pos[0] + (dx // 2 if dx != 0 else 0)
                            mid_y = current_pos[1] + (dy // 2 if dy != 0 else 0)
                            if (0 <= mid_x < env.width and 0 <= mid_y < env.height and
                                env.grid[mid_y, mid_x] not in (1, 2)):
                                occupied_mid = any(
                                    r.position == (mid_x, mid_y) and r.robot_id != robot_id
                                    for r in env.robots
                                )
                                if not occupied_mid:
                                    # 移动到中间位置
                                    if mid_x > current_pos[0]:
                                        processed_actions[robot_id] = 1  # 右
                                    elif mid_x < current_pos[0]:
                                        processed_actions[robot_id] = 3  # 左
                                    elif mid_y > current_pos[1]:
                                        processed_actions[robot_id] = 2  # 下
                                    elif mid_y < current_pos[1]:
                                        processed_actions[robot_id] = 0  # 上
                                    moved = True
                                    break
                if not moved:
                    # 如果无法离开，保持等待动作（让环境处理）
                    processed_actions[robot_id] = 4
    
    return processed_actions


def train_episode(env: WarehouseEnv, agent: MAPPO, task_allocator: TaskAllocatorNetwork,
                 baseline_comparison: BaselineComparison, use_learning_allocator: bool = True,
                 max_steps: int = 300):
    """训练一个episode"""
    obs, info = env.reset()
    done = False
    episode_reward = {robot_id: 0.0 for robot_id in obs.keys()}
    step_count = 0
    
    while not done:
        # 任务分配
        pending_tasks = [t for t in env.tasks if t.status.value == "pending"]
        # 获取空闲机器人：包括IDLE状态和CHARGING状态但电量已满的机器人
        idle_robots = [
            r for r in env.robots 
            if r.state.value == "idle" or 
               (r.state.value == "charging" and r.battery >= r.max_battery * 0.9)
        ]
        
        if pending_tasks and idle_robots:
            if use_learning_allocator and task_allocator is not None:
                # 使用学习到的任务分配网络
                global_state = env.get_global_state()
                robot_states = torch.FloatTensor(
                    global_state['robot_states']
                ).unsqueeze(0).to(agent.device)
                task_states = torch.FloatTensor(
                    global_state['task_states']
                ).unsqueeze(0).to(agent.device)
                
                # 采样分配
                allocation = task_allocator.sample_allocation(
                    robot_states, task_states,
                    global_state['num_robots'],
                    global_state['num_tasks']
                )
            else:
                # 使用基线算法
                baseline_results = baseline_comparison.compare(env.robots, env.tasks)
                allocation = baseline_results['greedy']  # 使用贪心算法作为默认
            
            # 执行分配
            for task_id, robot_id in allocation.items():
                if robot_id < len(env.robots):
                    task = next((t for t in env.tasks if t.task_id == task_id), None)
                    robot = env.robots[robot_id]
                    # 确保任务存在、状态为pending、机器人没有任务且状态允许
                    if (task and task.status.value == "pending" and 
                        robot.current_task is None and
                        (robot.state.value == "idle" or 
                         (robot.state.value == "charging" and robot.battery >= robot.max_battery * 0.9))):
                        robot.assign_task(task)
        
        # 选择动作
        actions, log_probs, values = agent.select_actions(obs)
        
        # 后处理动作：应用充完电离开充电站的逻辑
        # 礼让逻辑已经在环境层的 _execute_actions 中实现，会自动应用
        actions = post_process_actions(actions, env)
        
        # 执行动作
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # 存储经验
        dones = {k: terminated[k] or truncated[k] for k in terminated.keys()}
        agent.store_transition(obs, actions, rewards, values, log_probs, dones)
        
        # 更新
        for robot_id in rewards.keys():
            episode_reward[robot_id] += rewards[robot_id]
        
        obs = next_obs
        step_count += 1
        
        # 检查是否结束
        # 允许在达到最大步数时结束，即使有未完成任务（避免训练初期卡死）
        # 未完成任务的惩罚已经在环境step中通过episode_incomplete_penalty处理
        done = all(dones.values()) or step_count >= max_steps
    
    return episode_reward, step_count


def main():
    parser = argparse.ArgumentParser(description='训练多智能体强化学习系统')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径（默认: config.json）')
    parser.add_argument('--use_curriculum', action='store_true', help='使用课程学习')
    parser.add_argument('--use_learning_allocator', action='store_true', 
                       help='使用学习到的任务分配网络')
    parser.add_argument('--render', action='store_true', help='渲染环境')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                       help='模型保存目录')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 检测 GPU，若支持且在交互模式下询问是否使用
    if torch.cuda.is_available() and args.device == 'cpu' and sys.stdin.isatty():
        try:
            choice = input('检测到可用 GPU，是否使用 GPU 加速训练? (y/n，默认 y): ').strip().lower()
            if choice != 'n' and choice != 'no':
                args.device = 'cuda'
                print(f'将使用 GPU: {torch.cuda.get_device_name(0)}')
        except (EOFError, KeyboardInterrupt):
            pass
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化配置
    training_config = TrainingConfig()
    obs_config = ObservationConfig()
    
    # 确保RobotConfig和TrainingConfig的max_steps一致
    max_steps = training_config.max_steps_per_episode
    
    # 加载配置文件（默认 config.json）
    config_dict = {}
    if args.config and os.path.exists(args.config):
        from utils.config_loader import load_config_from_json
        try:
            config_dict = load_config_from_json(args.config)
            print(f"已加载配置文件: {args.config}")
        except Exception as e:
            print(f"警告：无法加载配置文件 {args.config}: {e}")
            print("使用默认配置")
    elif args.config and not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}，使用默认配置")
    
    # 课程学习
    curriculum = None
    if args.use_curriculum:
        curriculum = CurriculumLearning(create_default_curriculum())
        current_config = curriculum.get_current_config()
        current_config['max_steps'] = max_steps  # 添加max_steps
        # 合并配置文件中的仓库配置
        if 'warehouse' in config_dict:
            current_config['warehouse'] = config_dict['warehouse']
        if 'robot' in config_dict:
            current_config['robot'] = config_dict['robot']
    else:
        current_config = {
            'num_robots': 5,
            'max_tasks': 20,
            'task_spawn_rate': 0.1,
            'warehouse_width': 50,
            'warehouse_height': 50,
            'max_steps': max_steps  # 添加max_steps
        }
        # 合并配置文件
        if 'warehouse' in config_dict:
            current_config['warehouse'] = config_dict['warehouse']
        if 'robot' in config_dict:
            current_config['robot'] = config_dict['robot']
    
    # 创建环境
    env = create_env(current_config)
    
    # 创建智能体
    agent = MAPPO(
        obs_dim=obs_config.total_dim,
        action_dim=5,  # 5个动作
        num_agents=current_config['num_robots'],
        config=training_config,
        device=args.device
    )
    
    # 创建任务分配网络
    task_allocator = None
    if args.use_learning_allocator:
        task_allocator = TaskAllocatorNetwork(
            robot_state_dim=7,
            task_state_dim=6,
            max_robots=50,
            max_tasks=20
        ).to(args.device)
    
    # 基线算法对比
    baseline_comparison = BaselineComparison()
    
    # 可视化
    visualizer = None
    if args.render:
        visualizer = WarehouseVisualizer(
            env.warehouse_config,
            width=1200,
            height=1000
        )
    
    # 指标收集
    metrics_collector = MetricsCollector()
    
    # 训练循环
    num_episodes = training_config.num_episodes
    print(f"开始训练，共 {num_episodes} 个episodes")
    print(f"设备: {args.device}, 课程学习: {args.use_curriculum}, 学习分配器: {args.use_learning_allocator}")
    
    pbar = tqdm(range(num_episodes), desc="训练进度")
    for episode in pbar:
        # 课程学习更新
        if curriculum:
            stage_changed = curriculum.update(1)
            if stage_changed:
                print(f"\n进入新阶段: {curriculum.get_current_stage_info()}")
                current_config = curriculum.get_current_config()
                current_config['max_steps'] = max_steps  # 确保max_steps一致
                env = create_env(current_config)
                agent = MAPPO(
                    obs_dim=obs_config.total_dim,
                    action_dim=5,
                    num_agents=current_config['num_robots'],
                    config=training_config,
                    device=args.device
                )
        
        # 训练一个episode
        try:
            episode_reward, episode_length = train_episode(
                env, agent, task_allocator, baseline_comparison,
                args.use_learning_allocator, max_steps
            )
        except Exception as e:
            print(f"\nEpisode {episode} 出错: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 记录指标
        total_reward = sum(episode_reward.values())
        metrics_collector.record_episode(total_reward, episode_length)
        
        # 更新进度条描述
        if episode % 10 == 0:
            metrics = metrics_collector.compute_metrics()
            pbar.set_description(
                f"Episode {episode} | 奖励: {metrics['avg_episode_reward']:.1f} | "
                f"长度: {metrics['avg_episode_length']:.1f}"
            )
        
        # 更新策略
        if episode % training_config.batch_size == 0:
            agent.update()
        
        # 渲染
        if args.render and visualizer and episode % 10 == 0:
            stage_info = curriculum.get_current_stage_info() if curriculum else None
            visualizer.render(
                env.grid, env.robots, env.tasks,
                env.step_count, episode,
                episode_reward, stage_info
            )
            if not visualizer.handle_events():
                break
        
        # 保存模型
        if episode % training_config.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(
                args.save_dir,
                f'checkpoint_episode_{episode}.pth'
            )
            agent.save(checkpoint_path)
            
            # 打印指标
            metrics = metrics_collector.compute_metrics()
            print(f"\nEpisode {episode}:")
            print(f"  平均奖励: {metrics['avg_episode_reward']:.2f}")
            print(f"  平均episode长度: {metrics['avg_episode_length']:.2f}")
            if curriculum:
                print(f"  当前阶段: {curriculum.get_current_stage_info()}")
    
    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    agent.save(final_path)
    print(f"\n训练完成！模型已保存到: {final_path}")
    
    # 关闭可视化
    if visualizer:
        visualizer.close()


if __name__ == '__main__':
    main()
