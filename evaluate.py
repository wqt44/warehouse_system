"""
评估脚本：性能评估和对比
"""
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 配置中文字体支持
from utils.plot_utils import setup_chinese_font
setup_chinese_font()

from config import ObservationConfig, TrainingConfig
from env.warehouse_env import WarehouseEnv, LEAVE_WORKSTATION_MIN_DISTANCE
from algorithms.mappo import MAPPO
from models.task_allocator import TaskAllocatorNetwork
from utils.metrics import MetricsCollector
from baselines.optimization import BaselineComparison
from agents.robot import RobotState
from utils.env_utils import create_env
from utils.pathfinding import astar_path, path_to_action


def get_heuristic_action(robot, env: WarehouseEnv) -> int:
    """获取启发式动作（基于规则导航）"""
    current_pos = robot.position
    is_at_charging_station = env.grid[current_pos[1], current_pos[0]] == 4
    
    # 如果机器人在充电站且已充满电（>=90%），优先离开充电站
    if is_at_charging_station and robot.battery >= robot.max_battery * 0.9:
        # 尝试离开充电站
        for action in [0, 1, 2, 3]:  # 上、右、下、左
            action_deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
            dx_move, dy_move = action_deltas[action]
            new_x = current_pos[0] + dx_move
            new_y = current_pos[1] + dy_move
            if (0 <= new_x < env.width and 0 <= new_y < env.height and
                env.grid[new_y, new_x] not in (1, 2)):
                occupied = any(
                    r.position == (new_x, new_y) and r.robot_id != robot.robot_id
                    for r in env.robots
                )
                if not occupied:
                    return action
    
    # 如果机器人正在充电且电量未满，等待充电
    if robot.state == RobotState.CHARGING and robot.battery < robot.max_battery * 0.9:
        return 4  # 等待充电
    
    # 如果机器人正在返回充电站：用 A* 寻路到己编号对应的充电站（与任务寻路一致，避免穿站、绕障）
    elif robot.state == RobotState.RETURNING_TO_CHARGE:
        assigned = env.get_charging_station_for_robot(robot.robot_id)
        if assigned:
            if robot.is_at_position(assigned, tolerance=0):
                return 4
            occupied = {r.position for r in env.robots if r.robot_id != robot.robot_id}
            path = astar_path(
                env.grid, env.width, env.height,
                robot.position, assigned,
                occupied=occupied,
                allow_goal_on_shelf=False,
            )
            if path:
                return path_to_action(path)
        return 4
    
    # 如果有任务，用 A* 寻路到目标（能准确进入目标货物所在通道）
    if robot.current_task:
        task = robot.current_task
        if robot.state == RobotState.MOVING_TO_PICKUP:
            target = task.pickup_location
            allow_goal_on_shelf = True
        elif robot.state == RobotState.MOVING_TO_DROPOFF:
            target = task.dropoff_location
            allow_goal_on_shelf = False
        else:
            return 4
        occupied = {r.position for r in env.robots if r.robot_id != robot.robot_id}
        path = astar_path(
            env.grid, env.width, env.height,
            robot.position, target,
            occupied=occupied,
            allow_goal_on_shelf=allow_goal_on_shelf,
        )
        return path_to_action(path)
    
    # 没有任务时，检查是否需要充电
    else:
        current_pos = robot.position
        low_battery_threshold = 0.2
        needs_charge = robot.battery / robot.max_battery < low_battery_threshold
        
        if needs_charge or robot.state == RobotState.RETURNING_TO_CHARGE:
            assigned = env.get_charging_station_for_robot(robot.robot_id)
            if assigned:
                if robot.is_at_position(assigned, tolerance=0):
                    return 4  # 已在充电站，等待充电
                occupied = {r.position for r in env.robots if r.robot_id != robot.robot_id}
                path = astar_path(
                    env.grid, env.width, env.height,
                    current_pos, assigned,
                    occupied=occupied,
                    allow_goal_on_shelf=False,
                )
                if path:
                    return path_to_action(path)
            return 4
        
        # 交付后须离开到“所有工作站整体”至少 3 格以外；若 3 格有机器人则试 4 格，以此类推
        ws_positions = getattr(env.warehouse_config, 'workstation_positions', None) or []
        if getattr(robot, 'last_dropoff_workstation', None) and ws_positions:
            min_dist = min(robot.get_distance_to((wx, wy)) for (wx, wy) in ws_positions)
            if min_dist < LEAVE_WORKSTATION_MIN_DISTANCE:
                occupied = {r.position for r in env.robots if r.robot_id != robot.robot_id}
                max_d = max(env.width, env.height)
                for d in range(LEAVE_WORKSTATION_MIN_DISTANCE, max_d + 1):
                    # 候选格：从当前交付站沿四向取距离 d 的格
                    wx, wy = robot.last_dropoff_workstation[0], robot.last_dropoff_workstation[1]
                    candidates = [
                        (min(env.width - 1, wx + d), wy),
                        (max(0, wx - d), wy),
                        (wx, min(env.height - 1, wy + d)),
                        (wx, max(0, wy - d)),
                    ]
                    for target in candidates:
                        # 目标须与所有工作站都至少 3 格
                        if any(abs(target[0] - twx) + abs(target[1] - twy) < LEAVE_WORKSTATION_MIN_DISTANCE for (twx, twy) in ws_positions):
                            continue
                        if target in occupied:
                            continue
                        path = astar_path(
                            env.grid, env.width, env.height,
                            current_pos, target,
                            occupied=occupied,
                            allow_goal_on_shelf=False,
                        )
                        if path:
                            return path_to_action(path)
        
        # 不需要充电，如果机器人在工作站或充电站，尝试离开
        is_at_workstation = env.grid[current_pos[1], current_pos[0]] == 3
        is_at_charging_station = env.grid[current_pos[1], current_pos[0]] == 4
        if is_at_workstation or is_at_charging_station:
            # 尝试移动到相邻空位
            for action in [0, 1, 2, 3]:  # 上、右、下、左
                action_deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
                dx_move, dy_move = action_deltas[action]
                new_x = current_pos[0] + dx_move
                new_y = current_pos[1] + dy_move
                if (0 <= new_x < env.width and 0 <= new_y < env.height and
                    env.grid[new_y, new_x] not in (1, 2)):
                    return action
            return 4
    
    return 4  # 默认等待


def evaluate_episode(env: WarehouseEnv, agent: MAPPO, 
                    task_allocator: TaskAllocatorNetwork,
                    baseline_comparison: BaselineComparison,
                    use_learning: bool = True,
                    allocator_type: str = 'learning',
                    use_heuristic: bool = True,
                    max_steps: int = 1000):
    """评估一个episode"""
    obs, info = env.reset()
    done = False
    episode_reward = {robot_id: 0.0 for robot_id in obs.keys()}
    step_count = 0
    metrics_collector = MetricsCollector()
    
    while not done:
        # 任务分配
        pending_tasks = [t for t in env.tasks if t.status.value == "pending"]
        # 获取空闲机器人：包括IDLE状态和CHARGING状态但电量已满的机器人
        # 充完电的机器人（电量>=90%）即使还在充电站，也应该被视为空闲，可以分配任务
        idle_robots = [
            r for r in env.robots 
            if r.current_task is None and (
                r.state.value == "idle" or 
                (r.state.value == "charging" and r.battery >= r.max_battery * 0.9)
            )
        ]
        
        if pending_tasks and idle_robots:
            if use_learning and task_allocator is not None and allocator_type == 'learning':
                # 使用学习到的任务分配网络
                global_state = env.get_global_state()
                robot_states = torch.FloatTensor(
                    global_state['robot_states']
                ).unsqueeze(0)
                task_states = torch.FloatTensor(
                    global_state['task_states']
                ).unsqueeze(0)
                
                allocation = task_allocator.sample_allocation(
                    robot_states, task_states,
                    global_state['num_robots'],
                    global_state['num_tasks']
                )
            else:
                # 使用基线算法
                baseline_results = baseline_comparison.compare(env.robots, env.tasks)
                if allocator_type in baseline_results:
                    allocation = baseline_results[allocator_type]
                else:
                    allocation = baseline_results['greedy']
            
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
        if use_heuristic:
            # 使用启发式导航
            actions = {}
            for robot in env.robots:
                actions[robot.robot_id] = get_heuristic_action(robot, env)
        else:
            # 使用策略网络（确定性）
            actions, _, _ = agent.select_actions(obs, deterministic=True)
        
        # 执行动作
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # 更新
        for robot_id in rewards.keys():
            episode_reward[robot_id] += rewards[robot_id]
        
        # 记录指标（每一步都记录利用率，用于计算平均值）
        task_metrics = metrics_collector.compute_task_metrics(env.tasks, env.step_count)
        robot_metrics = metrics_collector.compute_robot_metrics(env.robots, record=True)
        
        obs = next_obs
        step_count += 1
        
        # 检查是否结束
        dones = {k: terminated[k] or truncated[k] for k in terminated.keys()}
        done = all(dones.values()) or step_count >= max_steps
    
    # 最终指标
    final_task_metrics = metrics_collector.compute_task_metrics(env.tasks, env.step_count)
    final_robot_metrics = metrics_collector.compute_robot_metrics(env.robots, record=False)
    
    # 使用平均利用率（整个episode期间的平均值）
    avg_utilization = final_robot_metrics.get('avg_utilization', final_robot_metrics['utilization'])
    
    # 调试信息：检查任务状态分布
    task_status_dist = {}
    for task in env.tasks:
        status = task.status.value
        task_status_dist[status] = task_status_dist.get(status, 0) + 1
    
    return {
        'total_reward': sum(episode_reward.values()),
        'episode_length': step_count,
        'task_completion_rate': final_task_metrics['completion_rate'],
        'avg_waiting_time': final_task_metrics['avg_waiting_time'],
        'robot_utilization': avg_utilization,  # 使用平均利用率
        'completed_tasks': final_task_metrics['completed_tasks'],
        'total_tasks': len(env.tasks),
        'task_status_dist': task_status_dist
    }


def compare_algorithms(env_config: dict, model_path: str = None, 
                      num_episodes: int = 10, device: str = 'cpu',
                      use_heuristic: bool = True, max_steps: int = 1000):
    """对比不同算法"""
    results = {
        'learning': [],
        'greedy': [],
        'hungarian': [],
        'genetic': []
    }
    
    # 创建环境
    env = create_env(env_config)
    
    # 加载模型（如果提供）
    agent = None
    task_allocator = None
    if model_path and os.path.exists(model_path):
        obs_config = ObservationConfig()
        training_config = TrainingConfig()
        
        agent = MAPPO(
            obs_dim=obs_config.total_dim,
            action_dim=5,
            num_agents=env_config.get('num_robots', 5),
            config=training_config,
            device=device
        )
        agent.load(model_path)
        
        task_allocator = TaskAllocatorNetwork(
            robot_state_dim=7,
            task_state_dim=6,
            max_robots=50,
            max_tasks=20
        ).to(device)
    
    baseline_comparison = BaselineComparison()
    
    # 评估每个算法
    for allocator_type in results.keys():
        print(f"\n评估算法: {allocator_type}")
        for episode in tqdm(range(num_episodes)):
            metrics = evaluate_episode(
                env, agent, task_allocator, baseline_comparison,
                use_learning=(allocator_type == 'learning'),
                allocator_type=allocator_type,
                use_heuristic=use_heuristic,
                max_steps=max_steps
            )
            results[allocator_type].append(metrics)
    
    return results


def plot_comparison(results: dict, save_path: str = None):
    """绘制对比结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_to_plot = [
        ('total_reward', '总奖励'),
        ('task_completion_rate', '任务完成率'),
        ('avg_waiting_time', '平均等待时间'),
        ('robot_utilization', '机器人利用率')
    ]
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        for algo_name, algo_results in results.items():
            values = [r[metric_key] for r in algo_results]
            ax.plot(values, label=algo_name, alpha=0.7)
            ax.axhline(y=np.mean(values), linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"对比图已保存到: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='评估多智能体强化学习系统')
    parser.add_argument('--model_path', type=str, default=None, 
                      help='模型路径')
    parser.add_argument('--num_episodes', type=int, default=10, 
                      help='评估episode数量')
    parser.add_argument('--num_robots', type=int, default=5, 
                      help='机器人数量')
    parser.add_argument('--max_tasks', type=int, default=20, 
                      help='最大任务数')
    parser.add_argument('--device', type=str, default='cpu', 
                      help='设备 (cpu/cuda)')
    parser.add_argument('--save_plot', type=str, default=None, 
                      help='保存对比图路径')
    parser.add_argument('--use_heuristic', action='store_true', default=True,
                      help='使用启发式导航（默认启用）')
    parser.add_argument('--no_heuristic', dest='use_heuristic', action='store_false',
                      help='不使用启发式导航，使用策略网络')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='每个episode的最大步数（默认1000）')
    
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径（默认: config.json）')
    
    args = parser.parse_args()
    
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
    
    # 环境配置
    env_config = {
        'num_robots': args.num_robots,
        'max_tasks': args.max_tasks,
        'task_spawn_rate': 0.1,
        'warehouse_width': 50,
        'warehouse_height': 50
    }
    
    # 如果提供了配置字典，合并仓库配置
    if config_dict:
        if 'warehouse' in config_dict:
            env_config['warehouse'] = config_dict['warehouse']
        if 'robot' in config_dict:
            env_config['robot'] = config_dict['robot']
            if 'num_robots' in config_dict['robot']:
                env_config['num_robots'] = config_dict['robot']['num_robots']
    
    # 对比算法
    print("开始算法对比评估...")
    print(f"使用启发式导航: {args.use_heuristic}")
    print(f"最大步数: {args.max_steps}")
    results = compare_algorithms(
        env_config, args.model_path, args.num_episodes, args.device,
        use_heuristic=args.use_heuristic, max_steps=args.max_steps
    )
    
    # 打印结果
    print("\n=== 评估结果 ===")
    for algo_name, algo_results in results.items():
        avg_reward = np.mean([r['total_reward'] for r in algo_results])
        avg_completion = np.mean([r['task_completion_rate'] for r in algo_results])
        avg_waiting = np.mean([r['avg_waiting_time'] for r in algo_results])
        avg_utilization = np.mean([r['robot_utilization'] for r in algo_results])
        avg_total_tasks = np.mean([r.get('total_tasks', 0) for r in algo_results])
        avg_completed = np.mean([r.get('completed_tasks', 0) for r in algo_results])
        
        print(f"\n{algo_name}:")
        print(f"  平均总奖励: {avg_reward:.2f}")
        print(f"  平均任务完成率: {avg_completion:.2%}")
        print(f"  平均完成任务数: {avg_completed:.1f} / {avg_total_tasks:.1f}")
        print(f"  平均等待时间: {avg_waiting:.2f}")
        print(f"  平均机器人利用率: {avg_utilization:.2%}")
        
        # 打印第一个episode的任务状态分布（用于调试）
        if algo_results and 'task_status_dist' in algo_results[0]:
            print(f"  任务状态分布（示例）: {algo_results[0]['task_status_dist']}")
    
    # 绘制对比图
    if args.save_plot:
        plot_comparison(results, args.save_plot)
    else:
        plot_comparison(results)


if __name__ == '__main__':
    main()
