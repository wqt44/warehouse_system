"""
环境工具函数：提供创建环境的公共函数
"""
from env.warehouse_env import WarehouseEnv
from config import RobotConfig, ObservationConfig, RewardConfig
from utils.config_loader import load_warehouse_config_from_dict


def create_env(config_dict: dict) -> WarehouseEnv:
    """
    创建仓库环境
    
    Args:
        config_dict: 配置字典，支持以下格式：
            - 新格式: {'warehouse': {...}, 'robot': {...}}
            - 旧格式: {'warehouse_width': ..., 'num_robots': ..., ...}
    
    Returns:
        WarehouseEnv对象
    """
    # 支持从warehouse子字典或直接键加载配置
    if 'warehouse' in config_dict:
        warehouse_dict = config_dict['warehouse']
    else:
        # 兼容旧格式
        warehouse_dict = {
            'width': config_dict.get('warehouse_width', 50),
            'height': config_dict.get('warehouse_height', 50),
            'task_spawn_rate': config_dict.get('task_spawn_rate', 0.1),
            'max_tasks': config_dict.get('max_tasks', 20),
            'shelf_regions': config_dict.get('shelf_regions'),
            'workstation_positions': config_dict.get('workstation_positions'),
            'obstacle_positions': config_dict.get('obstacle_positions'),
            'charging_stations': config_dict.get('charging_stations')
        }
    
    warehouse_config = load_warehouse_config_from_dict(warehouse_dict)
    
    # 机器人配置
    if 'robot' in config_dict:
        robot_dict = config_dict['robot']
        robot_config = RobotConfig(
            num_robots=robot_dict.get('num_robots', config_dict.get('num_robots', 5)),
            max_battery=robot_dict.get('max_battery', 100.0),
            battery_consumption_rate=robot_dict.get('battery_consumption_rate', 0.1),
            charging_rate=robot_dict.get('charging_rate', 2.0),
            max_steps_per_episode=config_dict.get('max_steps', 300)
        )
    else:
        robot_config = RobotConfig(
            num_robots=config_dict.get('num_robots', 5),
            max_battery=100.0,
            battery_consumption_rate=0.1,
            charging_rate=2.0,
            max_steps_per_episode=config_dict.get('max_steps', 300)
        )
    
    obs_config = ObservationConfig()
    reward_config = RewardConfig()
    
    env = WarehouseEnv(
        warehouse_config=warehouse_config,
        robot_config=robot_config,
        obs_config=obs_config,
        reward_config=reward_config
    )
    
    return env
