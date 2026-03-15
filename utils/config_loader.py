"""
配置加载工具：支持从JSON文件或字典加载仓库配置
"""
import json
from typing import Dict, Any
from config import WarehouseConfig


def load_warehouse_config_from_dict(config_dict: Dict[str, Any]) -> WarehouseConfig:
    """
    从字典创建WarehouseConfig
    
    Args:
        config_dict: 包含配置的字典，支持以下键：
            - width: 网格宽度
            - height: 网格高度
            - shelf_regions: 货架区域列表，格式 [[x1, y1, x2, y2], ...]
            - workstation_positions: 工作站位置列表，格式 [[x, y], ...]
            - obstacle_positions: 障碍物位置列表，格式 [[x, y], ...]
            - charging_stations: 充电站位置列表，格式 [[x, y], ...]（可选）
            - task_spawn_rate: 任务生成率
            - max_tasks: 最大任务数
    
    Returns:
        WarehouseConfig对象
    """
    # 转换列表为元组；键存在但为空列表时保留 []，键不存在时为 None 由 WarehouseConfig.__post_init__ 填默认
    shelf_regions = None
    if 'shelf_regions' in config_dict:
        shelf_regions = [tuple(region) for region in config_dict['shelf_regions']] if config_dict['shelf_regions'] else []
    workstation_positions = None
    if 'workstation_positions' in config_dict:
        workstation_positions = [tuple(pos) for pos in config_dict['workstation_positions']] if config_dict['workstation_positions'] else []
    obstacle_positions = None
    if 'obstacle_positions' in config_dict:
        obstacle_positions = [tuple(pos) for pos in config_dict['obstacle_positions']] if config_dict['obstacle_positions'] else []
    charging_stations = None
    if 'charging_stations' in config_dict and config_dict['charging_stations']:
        charging_stations = [tuple(pos) for pos in config_dict['charging_stations']]
    
    return WarehouseConfig(
        width=config_dict.get('width', 50),
        height=config_dict.get('height', 50),
        shelf_regions=shelf_regions,
        workstation_positions=workstation_positions,
        obstacle_positions=obstacle_positions,
        charging_stations=charging_stations,
        task_spawn_rate=config_dict.get('task_spawn_rate', 0.1),
        max_tasks=config_dict.get('max_tasks', 20)
    )


def load_config_from_json(json_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载配置
    
    Args:
        json_path: JSON配置文件路径
    
    Returns:
        配置字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def save_config_to_json(config_dict: Dict[str, Any], json_path: str):
    """
    保存配置到JSON文件
    
    Args:
        config_dict: 配置字典
        json_path: 保存路径
    """
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


def get_default_warehouse_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    获取默认仓库配置字典
    优先从 config.json 加载；若不存在或加载失败则返回最小默认配置
    
    Args:
        config_path: 配置文件路径（默认 config.json）
    
    Returns:
        默认配置字典
    """
    import os
    if os.path.exists(config_path):
        try:
            config = load_config_from_json(config_path)
            if config and "warehouse" in config:
                return config["warehouse"].copy()
        except Exception:
            pass
    return get_empty_warehouse_config()


def get_empty_warehouse_config() -> Dict[str, Any]:
    """获取空白仓库配置（用于清空等场景）"""
    return {
        "width": 50,
        "height": 50,
        "shelf_regions": [],
        "workstation_positions": [],
        "obstacle_positions": [],
        "charging_stations": None,
        "task_spawn_rate": 0.1,
        "max_tasks": 20
    }
