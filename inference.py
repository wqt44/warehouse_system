"""
推理脚本：使用训练好的模型进行推理和可视化
"""
import argparse
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 配置中文字体支持
from utils.plot_utils import setup_chinese_font
setup_chinese_font()

from config import ObservationConfig, TrainingConfig
from algorithms.mappo import MAPPO
from models.task_allocator import TaskAllocatorNetwork
from utils.visualization import WarehouseVisualizer
from utils.metrics import MetricsCollector
from baselines.optimization import BaselineComparison
from agents.robot import RobotState
from utils.env_utils import create_env
from utils.pathfinding import astar_path, path_to_action


def _plot_and_save_paths(paths: dict, grid: np.ndarray, width: int, height: int, save_path: str):
    """
    绘制并保存机器人路径图片
    
    Args:
        paths: 路径字典 {robot_id: [(x, y), ...]}
        grid: 网格数组 (height, width)
        width: 仓库宽度
        height: 仓库高度
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 绘制网格背景
    if grid is not None:
        cmap = ListedColormap(['white', 'gray', 'brown', 'blue', 'green'])
        im = ax.imshow(grid, cmap=cmap, origin='upper', 
                      extent=[-0.5, width-0.5, height-0.5, -0.5],
                      alpha=0.3, interpolation='nearest')
        
        # 添加网格图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', label='空地'),
            Patch(facecolor='gray', label='障碍物'),
            Patch(facecolor='brown', label='货架'),
            Patch(facecolor='blue', label='工作站'),
            Patch(facecolor='green', label='充电站')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    else:
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.set_title('机器人移动轨迹', fontsize=14, fontweight='bold')
    
    # 为每个机器人分配不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
    
    # 绘制每个机器人的路径
    for robot_id, traj in paths.items():
        if len(traj) < 2:
            continue
        
        color = colors[robot_id % len(colors)]
        
        # 提取x和y坐标
        x_coords = [pos[0] for pos in traj]
        y_coords = [pos[1] for pos in traj]
        
        # 绘制路径线
        ax.plot(x_coords, y_coords, color=color, linewidth=2.5, 
               alpha=0.7, label=f'机器人 {robot_id}', zorder=5)
        
        # 标记起点（绿色圆圈）
        ax.plot(x_coords[0], y_coords[0], 'o', color='green', 
               markersize=12, markeredgecolor='black', markeredgewidth=2,
               zorder=6, label='起点' if robot_id == 0 else '')
        
        # 标记终点（红色方块）
        ax.plot(x_coords[-1], y_coords[-1], 's', color='red', 
               markersize=12, markeredgecolor='black', markeredgewidth=2,
               zorder=6, label='终点' if robot_id == 0 else '')
    
    # 添加路径图例
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    
    # 添加统计信息
    stats_text = f"总机器人数: {len(paths)}\n"
    total_steps = sum(len(traj) for traj in paths.values())
    stats_text += f"总步数: {total_steps}\n"
    avg_steps = total_steps / len(paths) if paths else 0
    stats_text += f"平均步数: {avg_steps:.1f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_inference(model_path: str, num_episodes: int = 1, 
                 num_robots: int = 5, max_tasks: int = 20,
                 render: bool = True, device: str = 'cpu',
                 config_dict: dict = None):
    """
    使用训练好的模型进行推理
    
    Args:
        model_path: 模型文件路径
        num_episodes: 运行的episode数量
        num_robots: 机器人数量
        max_tasks: 最大任务数
        render: 是否可视化
        device: 计算设备
        config_dict: 配置字典（可选，用于自定义仓库布局）
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        print("请先训练模型或检查模型路径是否正确")
        return
    
    print(f"加载模型: {model_path}")
    
    # 环境配置（提高任务生成率以确保有任务）
    env_config = {
        'num_robots': num_robots,
        'max_tasks': max_tasks,
        'task_spawn_rate': 0.3,  # 提高任务生成率
    }
    
    # 如果提供了配置字典，合并仓库配置
    if config_dict is not None:
        if 'warehouse' in config_dict:
            env_config['warehouse'] = config_dict['warehouse']
        if 'robot' in config_dict:
            env_config['robot'] = config_dict['robot']
    
    # 创建环境
    env = create_env(env_config)
    
    # 创建智能体
    obs_config = ObservationConfig()
    training_config = TrainingConfig()
    
    agent = MAPPO(
        obs_dim=obs_config.total_dim,
        action_dim=5,
        num_agents=num_robots,
        config=training_config,
        device=device
    )
    
    # 加载模型
    print("正在加载模型权重...")
    agent.load(model_path)
    print("模型加载完成！")
    
    # 创建任务分配网络（如果使用学习到的分配器）
    task_allocator = None
    use_learning_allocator = False  # 可以根据需要启用
    use_rule_navigation = True      # 使用启发式导航以提高完成率
    
    # 创建可视化
    visualizer = None
    if render:
        visualizer = WarehouseVisualizer(
            env.warehouse_config,
            width=1200,
            height=1000,
            cell_size=20
        )
        print("可视化已启用")
    
    # 指标收集
    metrics_collector = MetricsCollector()
    baseline_comparison = BaselineComparison()
    
    # 运行推理
    print(f"\n开始运行 {num_episodes} 个episode...")
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        # 重置环境
        obs, info = env.reset()
        done = False
        episode_reward = {robot_id: 0.0 for robot_id in obs.keys()}
        step_count = 0
        
        # 记录每个机器人的路径：robot_id -> [(x, y), ...]
        paths = {robot.robot_id: [] for robot in env.robots}
        # 记录初始位置
        for robot in env.robots:
            paths[robot.robot_id].append(robot.position)
        
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
                if use_learning_allocator and task_allocator is not None:
                    # 使用学习到的任务分配网络
                    global_state = env.get_global_state()
                    robot_states = torch.FloatTensor(
                        global_state['robot_states']
                    ).unsqueeze(0).to(device)
                    task_states = torch.FloatTensor(
                        global_state['task_states']
                    ).unsqueeze(0).to(device)
                    
                    allocation = task_allocator.sample_allocation(
                        robot_states, task_states,
                        global_state['num_robots'],
                        global_state['num_tasks']
                    )
                else:
                    # 使用基线算法（贪心）
                    baseline_results = baseline_comparison.compare(env.robots, env.tasks)
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
            
            # 选择动作：优先启发式导航以提高任务完成率
            if use_rule_navigation:
                actions = {}
                for robot in env.robots:
                    current_pos = robot.position
                    is_at_charging_station = env.grid[current_pos[1], current_pos[0]] == 4
                    
                    # 如果机器人在充电站且已充满电（>=90%），优先离开充电站
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
                                    r.position == adj_pos and r.robot_id != robot.robot_id
                                    for r in env.robots
                                )
                                if not occupied:
                                    dx = adj_pos[0] - current_pos[0]
                                    dy = adj_pos[1] - current_pos[1]
                                    if dx > 0:
                                        actions[robot.robot_id] = 1  # 右
                                    elif dx < 0:
                                        actions[robot.robot_id] = 3  # 左
                                    elif dy > 0:
                                        actions[robot.robot_id] = 2  # 下
                                    elif dy < 0:
                                        actions[robot.robot_id] = 0  # 上
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
                                        r.position == (target_x, target_y) and r.robot_id != robot.robot_id
                                        for r in env.robots
                                    )
                                    if not occupied:
                                        # 先移动到中间位置
                                        mid_x = current_pos[0] + (dx // 2 if dx != 0 else 0)
                                        mid_y = current_pos[1] + (dy // 2 if dy != 0 else 0)
                                        if (0 <= mid_x < env.width and 0 <= mid_y < env.height and
                                            env.grid[mid_y, mid_x] not in (1, 2)):
                                            occupied_mid = any(
                                                r.position == (mid_x, mid_y) and r.robot_id != robot.robot_id
                                                for r in env.robots
                                            )
                                            if not occupied_mid:
                                                # 移动到中间位置
                                                if mid_x > current_pos[0]:
                                                    actions[robot.robot_id] = 1  # 右
                                                elif mid_x < current_pos[0]:
                                                    actions[robot.robot_id] = 3  # 左
                                                elif mid_y > current_pos[1]:
                                                    actions[robot.robot_id] = 2  # 下
                                                elif mid_y < current_pos[1]:
                                                    actions[robot.robot_id] = 0  # 上
                                                moved = True
                                                break
                        
                        if moved:
                            continue
                        # 如果无法离开，继续后续逻辑（可能有任务需要执行）
                    
                    # 如果机器人正在充电（且电量未满），等待充电
                    if robot.state == RobotState.CHARGING and robot.battery < robot.max_battery * 0.9:
                        actions[robot.robot_id] = 4  # 等待充电
                        continue
                    elif robot.state == RobotState.RETURNING_TO_CHARGE:
                        # 前往最近的充电站
                        if env.charging_stations:
                            nearest_charging = min(env.charging_stations, 
                                                 key=lambda p: robot.get_distance_to(p))
                            dx = nearest_charging[0] - robot.position[0]
                            dy = nearest_charging[1] - robot.position[1]
                            if abs(dx) > abs(dy):
                                actions[robot.robot_id] = 1 if dx > 0 else 3
                            elif dy != 0:
                                actions[robot.robot_id] = 2 if dy > 0 else 0
                            else:
                                actions[robot.robot_id] = 4
                        else:
                            actions[robot.robot_id] = 4
                        continue
                    
                    # 没有任务时，检查是否需要充电
                    if not robot.current_task:
                        current_pos = robot.position
                        
                        # 检查是否需要充电（电量低于阈值）
                        low_battery_threshold = 0.2
                        needs_charge = robot.battery / robot.max_battery < low_battery_threshold
                        
                        # 如果正在充电且电量未满，继续等待
                        if robot.state == RobotState.CHARGING and robot.battery < robot.max_battery * 0.9:
                            actions[robot.robot_id] = 4  # 等待充电
                            continue
                        
                        # 如果需要充电，前往充电站
                        if needs_charge or robot.state == RobotState.RETURNING_TO_CHARGE:
                            if env.charging_stations:
                                # 找到最近的充电站
                                nearest_charging = min(env.charging_stations, 
                                                     key=lambda p: robot.get_distance_to(p))
                                
                                # 检查是否已经在充电站
                                if robot.is_at_position(nearest_charging, tolerance=0):
                                    actions[robot.robot_id] = 4  # 已在充电站，等待充电
                                else:
                                    # 前往充电站
                                    dx = nearest_charging[0] - current_pos[0]
                                    dy = nearest_charging[1] - current_pos[1]
                                    
                                    # 优先选择主要方向
                                    if abs(dx) > abs(dy):
                                        preferred_action = 1 if dx > 0 else 3  # 右或左
                                    elif dy != 0:
                                        preferred_action = 2 if dy > 0 else 0  # 下或上
                                    else:
                                        preferred_action = 4
                                    
                                    # 检查首选动作是否可通行
                                    if preferred_action != 4:
                                        action_deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, 0)}
                                        dx_move, dy_move = action_deltas[preferred_action]
                                        new_x = current_pos[0] + dx_move
                                        new_y = current_pos[1] + dy_move
                                        
                                        if (0 <= new_x < env.width and 0 <= new_y < env.height and
                                            env.grid[new_y, new_x] not in (1, 2)):
                                            actions[robot.robot_id] = preferred_action
                                        else:
                                            # 尝试其他方向
                                            if abs(dx) > abs(dy):
                                                alt_action = 2 if dy > 0 else 0 if dy < 0 else 4
                                            else:
                                                alt_action = 1 if dx > 0 else 3 if dx < 0 else 4
                                            
                                            if alt_action != 4:
                                                dx_alt, dy_alt = action_deltas[alt_action]
                                                new_x_alt = current_pos[0] + dx_alt
                                                new_y_alt = current_pos[1] + dy_alt
                                                if (0 <= new_x_alt < env.width and 0 <= new_y_alt < env.height and
                                                    env.grid[new_y_alt, new_x_alt] not in (1, 2)):
                                                    actions[robot.robot_id] = alt_action
                                                else:
                                                    actions[robot.robot_id] = 4
                                            else:
                                                actions[robot.robot_id] = 4
                                    else:
                                        actions[robot.robot_id] = 4
                            else:
                                actions[robot.robot_id] = 4  # 没有充电站，等待
                            continue
                        
                        # 不需要充电，如果机器人在工作站或充电站，尝试离开
                        is_at_workstation = env.grid[current_pos[1], current_pos[0]] == 3
                        is_at_charging_station = env.grid[current_pos[1], current_pos[0]] == 4
                        
                        if is_at_workstation or is_at_charging_station:
                            # 在工作站或充电站且无任务，尝试移动到相邻空位
                            # 优先尝试所有相邻位置（包括空地、工作站、充电站，但不能是障碍物或货架）
                            adjacent_positions = [
                                (current_pos[0] + dx, current_pos[1] + dy)
                                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
                            ]
                            
                            # 随机打乱顺序，避免所有机器人选择同一方向
                            random.shuffle(adjacent_positions)
                            
                            moved = False
                            for adj_pos in adjacent_positions:
                                adj_x, adj_y = adj_pos
                                # 检查是否在边界内且不是障碍物
                                if (0 <= adj_x < env.width and 0 <= adj_y < env.height and
                                    env.grid[adj_y, adj_x] not in (1, 2)):  # 可通行
                                    # 检查是否有其他机器人在该位置
                                    occupied = any(
                                        r.position == adj_pos and r.robot_id != robot.robot_id 
                                        for r in env.robots
                                    )
                                    if not occupied:
                                        # 找到空位，移动到该位置
                                        dx = adj_pos[0] - current_pos[0]
                                        dy = adj_pos[1] - current_pos[1]
                                        if dx > 0:
                                            actions[robot.robot_id] = 1  # 右
                                        elif dx < 0:
                                            actions[robot.robot_id] = 3  # 左
                                        elif dy > 0:
                                            actions[robot.robot_id] = 2  # 下
                                        elif dy < 0:
                                            actions[robot.robot_id] = 0  # 上
                                        moved = True
                                        break
                            
                            if not moved:
                                # 如果无法移动到相邻位置，尝试移动到更远的位置（2步距离）
                                # 这样可以避免多个机器人在充电站互相阻挡
                                for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0), 
                                               (-1, -1), (1, -1), (1, 1), (-1, 1)]:
                                    target_x = current_pos[0] + dx
                                    target_y = current_pos[1] + dy
                                    if (0 <= target_x < env.width and 0 <= target_y < env.height and
                                        env.grid[target_y, target_x] not in (1, 2)):
                                        occupied = any(
                                            r.position == (target_x, target_y) and r.robot_id != robot.robot_id
                                            for r in env.robots
                                        )
                                        if not occupied:
                                            # 先移动到中间位置
                                            mid_x = current_pos[0] + (dx // 2 if dx != 0 else 0)
                                            mid_y = current_pos[1] + (dy // 2 if dy != 0 else 0)
                                            if (0 <= mid_x < env.width and 0 <= mid_y < env.height and
                                                env.grid[mid_y, mid_x] not in (1, 2)):
                                                occupied_mid = any(
                                                    r.position == (mid_x, mid_y) and r.robot_id != robot.robot_id
                                                    for r in env.robots
                                                )
                                                if not occupied_mid:
                                                    # 移动到中间位置
                                                    if mid_x > current_pos[0]:
                                                        actions[robot.robot_id] = 1  # 右
                                                    elif mid_x < current_pos[0]:
                                                        actions[robot.robot_id] = 3  # 左
                                                    elif mid_y > current_pos[1]:
                                                        actions[robot.robot_id] = 2  # 下
                                                    elif mid_y < current_pos[1]:
                                                        actions[robot.robot_id] = 0  # 上
                                                    moved = True
                                                    break
                                if not moved:
                                    actions[robot.robot_id] = 4  # 无法移动，等待
                        else:
                            actions[robot.robot_id] = 4  # 不在工作站或充电站，等待
                        continue
                    
                    # 有任务时，用 A* 寻路到目标（能准确进入目标货物所在通道）
                    task = robot.current_task
                    if robot.state.value in ["moving_to_pickup", "picking_up"]:
                        target = task.pickup_location
                        allow_goal_on_shelf = True
                    elif robot.state.value in ["moving_to_dropoff", "dropping_off"]:
                        target = task.dropoff_location
                        allow_goal_on_shelf = False
                    else:
                        actions[robot.robot_id] = 4
                        continue
                    occupied = {r.position for r in env.robots if r.robot_id != robot.robot_id}
                    path = astar_path(
                        env.grid, env.width, env.height,
                        robot.position, target,
                        occupied=occupied,
                        allow_goal_on_shelf=allow_goal_on_shelf,
                    )
                    actions[robot.robot_id] = path_to_action(path)
            else:
                # 使用训练策略（确定性）
                actions, _, _ = agent.select_actions(obs, deterministic=True)
            
            # 执行动作
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # 累计奖励
            for robot_id in rewards.keys():
                episode_reward[robot_id] += rewards[robot_id]
            
            # 记录当前步各机器人位置
            for robot in env.robots:
                paths[robot.robot_id].append(robot.position)
            
            # 记录每一步的利用率（用于计算平均值）
            metrics_collector.compute_robot_metrics(env.robots, record=True)
            
            # 检查是否完成：只有当所有任务完成时才结束
            # 检查是否有未完成的任务（pending, assigned, in_progress）
            unfinished_tasks = [
                t for t in env.tasks 
                if t.status.value in ["pending", "assigned", "in_progress"]
            ]
            
            # 更新done状态：所有任务完成或所有机器人终止/截断
            dones = {k: terminated[k] or truncated[k] for k in terminated.keys()}
            # 如果有未完成任务，强制继续运行
            if unfinished_tasks:
                done = False
            else:
                done = all(dones.values())
            
            # 渲染
            if visualizer:
                visualizer.render(
                    env.grid, env.robots, env.tasks,
                    step_count, episode,
                    rewards
                )
                if not visualizer.handle_events():
                    print("用户中断")
                    done = True
                    break
            
            obs = next_obs
            step_count += 1
            
            # 调试信息（可选）
            if step_count % 50 == 0:
                print(f"  步数: {step_count}, 任务数: {len(env.tasks)}, "
                      f"待分配: {len([t for t in env.tasks if t.status.value == 'pending'])}")
            
            # 安全限制：如果步数过多（例如10000步），强制结束以避免无限循环
            # 但优先检查任务完成情况
            if step_count >= 10000:
                print(f"  警告：达到最大安全步数限制（10000步），强制结束")
                done = True
                break
        
        # 计算episode指标（使用平均利用率）
        task_metrics = metrics_collector.compute_task_metrics(env.tasks, env.step_count)
        robot_metrics = metrics_collector.compute_robot_metrics(env.robots, record=False)
        # 如果有历史记录，使用平均利用率；否则使用当前利用率
        if metrics_collector.robot_utilization:
            robot_metrics['utilization'] = np.mean(metrics_collector.robot_utilization)
        total_reward = sum(episode_reward.values())
        
        # 打印结果
        print(f"Episode {episode + 1} 完成:")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  步数: {step_count}")
        print(f"  任务完成率: {task_metrics['completion_rate']:.2%}")
        print(f"  完成任务数: {task_metrics['completed_tasks']}/{len(env.tasks)}")
        print(f"  机器人利用率: {robot_metrics['utilization']:.2%}")
        print(f"  平均等待时间: {task_metrics['avg_waiting_time']:.2f}")
        
        # 打印路径信息
        print(f"\n  轨迹信息：")
        for rid, traj in paths.items():
            print(f"    机器人 {rid}: 路径长度 {len(traj)}, 起点 {traj[0]}, 终点 {traj[-1]}")
            if len(traj) > 1:
                # 计算移动距离
                total_distance = sum(
                    abs(traj[i][0] - traj[i-1][0]) + abs(traj[i][1] - traj[i-1][1])
                    for i in range(1, len(traj))
                )
                print(f"      总移动距离: {total_distance} 步")
        
        # 绘制并保存路径图片
        save_path = "robot_paths.png"
        _plot_and_save_paths(paths, env.grid, env.warehouse_config.width, 
                            env.warehouse_config.height, save_path)
        print(f"  路径已保存到: {save_path}")
        
        # 记录指标
        metrics_collector.record_episode(total_reward, step_count)
    
    # 打印总体统计
    print("\n=== 总体统计 ===")
    final_metrics = metrics_collector.compute_metrics()
    print(f"平均episode奖励: {final_metrics['avg_episode_reward']:.2f}")
    print(f"平均episode长度: {final_metrics['avg_episode_length']:.2f}")
    
    # 关闭可视化
    if visualizer:
        visualizer.close()
    
    print("\n推理完成！")


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行推理')
    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/final_model.pth',
                       help='模型文件路径（默认: ./checkpoints/final_model.pth）')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径（默认: config.json）')
    parser.add_argument('--num_episodes', type=int, default=1,
                       help='运行的episode数量（默认: 1）')
    parser.add_argument('--num_robots', type=int, default=5,
                       help='机器人数量（默认: 5）')
    parser.add_argument('--max_tasks', type=int, default=20,
                       help='最大任务数（默认: 20）')
    parser.add_argument('--no_render', action='store_true',
                       help='禁用可视化')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备（cpu/cuda，默认: cpu）')
    
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
    
    # 如果提供了配置文件，使用配置文件中的值，否则使用命令行参数
    num_robots = args.num_robots
    max_tasks = args.max_tasks
    if 'robot' in config_dict and 'num_robots' in config_dict['robot']:
        num_robots = config_dict['robot']['num_robots']
    if 'warehouse' in config_dict and 'max_tasks' in config_dict['warehouse']:
        max_tasks = config_dict['warehouse']['max_tasks']
    
    run_inference(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        num_robots=num_robots,
        max_tasks=max_tasks,
        render=not args.no_render,
        device=args.device,
        config_dict=config_dict
    )


if __name__ == '__main__':
    main()
