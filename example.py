"""
示例脚本：演示如何使用仓储系统
"""
import os
import numpy as np
from utils.env_utils import create_env
from utils.config_loader import load_config_from_json
from utils.visualization import WarehouseVisualizer
from baselines.optimization import GreedyAllocator


def main():
    """简单示例：使用贪心算法进行任务分配"""
    # 从 config.json 加载配置并创建环境
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        config_dict = load_config_from_json(config_path)
    else:
        config_dict = {
            'warehouse': {'width': 50, 'height': 50, 'shelf_regions': [], 'workstation_positions': [],
                          'obstacle_positions': [], 'charging_stations': None, 'task_spawn_rate': 0.1, 'max_tasks': 20},
            'robot': {'num_robots': 5},
        }
    env = create_env(config_dict)
    
    # 创建可视化
    visualizer = WarehouseVisualizer(
        env.warehouse_config,
        width=800,
        height=800,
        cell_size=25
    )
    
    # 创建任务分配器
    allocator = GreedyAllocator()
    
    # 重置环境
    obs, info = env.reset()
    
    # 运行几个步骤
    for step in range(100):
        # 任务分配
        pending_tasks = [t for t in env.tasks if t.status.value == "pending"]
        idle_robots = [r for r in env.robots if r.state.value == "idle"]
        
        if pending_tasks and idle_robots:
            allocation = allocator.allocate(env.robots, env.tasks)
            for task_id, robot_id in allocation.items():
                if robot_id < len(env.robots):
                    task = next((t for t in env.tasks if t.task_id == task_id), None)
                    if task and task.status.value == "pending":
                        env.robots[robot_id].assign_task(task)
        
        # 随机动作（演示用）
        actions = {}
        for robot_id in obs.keys():
            actions[robot_id] = np.random.randint(0, 5)
        
        # 执行动作
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # 渲染
        visualizer.render(
            env.grid, env.robots, env.tasks,
            step, 0, rewards
        )
        
        # 检查事件
        if not visualizer.handle_events():
            break
        
        # 检查是否结束
        if all(terminated.values()) or all(truncated.values()):
            break
    
    # 关闭可视化
    visualizer.close()
    print("示例运行完成！")


if __name__ == '__main__':
    main()
