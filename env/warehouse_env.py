"""
仓库环境：可配置的网格化仓库世界，支持多智能体强化学习
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Optional, Any
import random

from agents.robot import Robot, RobotState
from agents.task import Task, TaskType, TaskStatus
from env.observation import ObservationSpace
from env.reward import RewardFunction
from config import WarehouseConfig, RobotConfig, ObservationConfig, RewardConfig


class WarehouseEnv(gym.Env):
    """仓库多智能体环境"""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 warehouse_config: WarehouseConfig,
                 robot_config: RobotConfig,
                 obs_config: ObservationConfig,
                 reward_config: RewardConfig,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.warehouse_config = warehouse_config
        self.robot_config = robot_config
        self.obs_config = obs_config
        # 保存奖励配置，方便在episode结束时使用硬约束奖励
        self.reward_config = reward_config
        self.render_mode = render_mode
        
        # 初始化组件
        self.observation_space_obj = ObservationSpace(obs_config, warehouse_config)
        self.reward_fn = RewardFunction(reward_config)
        
        # 环境状态
        self.width = warehouse_config.width
        self.height = warehouse_config.height
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.robots: List[Robot] = []
        self.tasks: List[Task] = []
        self.task_counter = 0
        self.step_count = 0
        
        # 动作空间：5个动作（上、右、下、左、等待）
        self.action_space = spaces.Discrete(5)
        
        # 观察空间维度
        obs_dim = obs_config.total_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 初始化网格
        self._initialize_grid()
        
        # 初始化机器人
        self._initialize_robots()
    
    def _initialize_grid(self):
        """初始化仓库网格"""
        # 0=空地, 1=障碍物, 2=货架, 3=工作站, 4=充电站
        
        # 设置货架区域
        for region in self.warehouse_config.shelf_regions:
            x1, y1, x2, y2 = region
            # 确保不超出网格边界
            y1 = max(0, min(y1, self.height - 1))
            y2 = max(0, min(y2, self.height - 1))
            x1 = max(0, min(x1, self.width - 1))
            x2 = max(0, min(x2, self.width - 1))
            # 确保 y1 <= y2 和 x1 <= x2
            if y1 <= y2 and x1 <= x2:
                self.grid[y1:y2+1, x1:x2+1] = 2
        
        # 设置工作站
        for pos in self.warehouse_config.workstation_positions:
            x, y = pos
            # 确保不超出边界
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            self.grid[y, x] = 3
        
        # 设置障碍物
        for pos in self.warehouse_config.obstacle_positions:
            x, y = pos
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1
        
        # 设置充电站
        if self.warehouse_config.charging_stations is not None:
            # 使用自定义充电站位置
            charging_stations = []
            for pos in self.warehouse_config.charging_stations:
                x, y = pos
                # 确保不超出边界
                x = max(0, min(x, self.width - 1))
                y = max(0, min(y, self.height - 1))
                if self.grid[y, x] not in (1, 2):  # 可通行（非障碍物、非货架）
                    self.grid[y, x] = 4
                    charging_stations.append((x, y))
            self.charging_stations = charging_stations
        else:
            # 自动在工作站附近生成充电站
            charging_stations = []
            for pos in self.warehouse_config.workstation_positions:
                x, y = pos
                # 在工作站周围添加充电站
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    cx, cy = x + dx, y + dy
                    if (0 <= cx < self.width and 0 <= cy < self.height and 
                        self.grid[cy, cx] == 0):
                        self.grid[cy, cx] = 4
                        charging_stations.append((cx, cy))
                        break
            self.charging_stations = charging_stations
    
    def _initialize_robots(self):
        """初始化机器人"""
        self.robots = []
        valid_positions = []
        
        # 找到所有可通行位置（非障碍物、非货架）
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:  # 空地
                    valid_positions.append((x, y))
        
        # 随机选择机器人初始位置
        start_positions = random.sample(valid_positions, 
                                       min(self.robot_config.num_robots, 
                                           len(valid_positions)))
        
        for i in range(self.robot_config.num_robots):
            pos = start_positions[i] if i < len(start_positions) else valid_positions[0]
            robot = Robot(
                robot_id=i,
                position=pos,
                battery=self.robot_config.max_battery,
                max_battery=self.robot_config.max_battery
            )
            self.robots.append(robot)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 重置状态
        self.step_count = 0
        self.task_counter = 0
        self.tasks.clear()
        self.reward_fn.reset()
        
        # 重新初始化网格和机器人
        self._initialize_grid()
        self._initialize_robots()
        
        # 获取初始观察
        observations = self._get_observations()
        infos = self._get_infos()
        
        return observations, infos
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        执行一步
        
        Args:
            actions: 机器人ID到动作的映射 {robot_id: action}
        
        Returns:
            observations, rewards, terminated, truncated, infos
        """
        self.step_count += 1
        
        # 清理“本步完成任务”标记（由 Robot.complete_task 设置）
        for r in self.robots:
            r.just_completed_task = False
            r.last_completed_task_id = None
        
        # 1. 执行动作
        self._execute_actions(actions)
        
        # 2. 更新机器人状态（电量、任务进度等）
        self._update_robots()
        
        # 3. 生成新任务
        self._spawn_tasks()
        
        # 4. 计算奖励
        rewards = self._compute_rewards()
        
        # 5. 检查终止条件
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        # 6. 硬约束：episode 结束时仍有未完成任务，给所有机器人一个大负奖励
        is_final_step = all(terminated.values()) or any(truncated.values())
        if is_final_step:
            # 如果存在未完成任务（pending 或 in_progress），施加episode级惩罚
            has_unfinished = any(
                t.status.value != "completed" for t in self.tasks
            )
            if has_unfinished:
                penalty = self.reward_config.episode_incomplete_penalty
                num_robots = max(1, len(self.robots))
                shared_penalty = penalty / num_robots
                for rid in rewards.keys():
                    rewards[rid] += shared_penalty
        
        # 7. 获取观察和信息
        observations = self._get_observations()
        infos = self._get_infos()
        
        # 更新位置历史
        self.reward_fn.update_positions(self.robots)
        
        return observations, rewards, terminated, truncated, infos
    
    def _choose_yielder(self, robot_id_a: int, robot_id_b: int) -> int:
        """在冲突的两人中选谁礼让：有货不动，没货让开；否则 ID 大的让"""
        ra, rb = self.robots[robot_id_a], self.robots[robot_id_b]
        has_a, has_b = ra.current_task is not None, rb.current_task is not None
        if has_a and not has_b:
            return robot_id_b
        if has_b and not has_a:
            return robot_id_a
        return max(robot_id_a, robot_id_b)
    
    def _resolve_predictive_collisions(self, new_positions: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """预测碰撞并提前礼让：同目标、互换位置等冲突，让应礼让者本步等待"""
        current_positions = {r.robot_id: r.position for r in self.robots}
        new_positions = dict(new_positions)
        while True:
            conflict_found = False
            for i in list(new_positions.keys()):
                if conflict_found:
                    break
                for j in list(new_positions.keys()):
                    if i >= j:
                        continue
                    pos_i, pos_j = current_positions[i], current_positions[j]
                    new_i, new_j = new_positions[i], new_positions[j]
                    # 同目标
                    if new_i == new_j:
                        yielder = self._choose_yielder(i, j)
                        new_positions[yielder] = current_positions[yielder]
                        conflict_found = True
                        break
                    # 互换位置（对穿）
                    if new_i == pos_j and new_j == pos_i:
                        yielder = self._choose_yielder(i, j)
                        new_positions[yielder] = current_positions[yielder]
                        conflict_found = True
                        break
            if not conflict_found:
                break
        return new_positions
    
    def _execute_actions(self, actions: Dict[int, int]):
        """执行动作"""
        # 动作：0=上, 1=右, 2=下, 3=左, 4=等待
        action_deltas = {
            0: (0, -1),   # 上
            1: (1, 0),    # 右
            2: (0, 1),    # 下
            3: (-1, 0),   # 左
            4: (0, 0)     # 等待
        }
        
        # 先计算所有新位置
        new_positions = {}
        for robot_id, action in actions.items():
            if robot_id >= len(self.robots):
                continue
            
            robot = self.robots[robot_id]
            current_pos = robot.position
            is_at_charging_station = self.grid[current_pos[1], current_pos[0]] == 4
            
            # 如果机器人在充电站且正在充电，强制保持原位置（不移动）
            if is_at_charging_station and robot.state == RobotState.CHARGING:
                new_positions[robot_id] = current_pos  # 保持原位置，不移动
                continue
            
            dx, dy = action_deltas[action]
            new_x = robot.position[0] + dx
            new_y = robot.position[1] + dy
            
            # 检查边界和障碍物
            # 机器人可进入：空地(0)、工作站(3)、充电站(4)；货架(2)仅当前往取货点时允许进入
            if (0 <= new_x < self.width and 0 <= new_y < self.height):
                grid_value = self.grid[new_y, new_x]
                if grid_value == 1:
                    new_positions[robot_id] = robot.position  # 障碍物不可进入
                elif grid_value == 2:
                    # 货架：仅当目标格为当前任务的取货点且处于“前往取货”状态时可进入
                    if (robot.current_task is not None and robot.state == RobotState.MOVING_TO_PICKUP and
                            robot.current_task.pickup_location == (new_x, new_y)):
                        new_positions[robot_id] = (new_x, new_y)
                    else:
                        new_positions[robot_id] = robot.position
                else:
                    new_positions[robot_id] = (new_x, new_y)
            else:
                new_positions[robot_id] = robot.position  # 保持原位置
        
        # 预测碰撞并提前礼让：有货的机器人不动，没货的机器人提前改等待
        new_positions = self._resolve_predictive_collisions(new_positions)
        
        # 检查碰撞并实现互相礼让机制（处理“目标被占”时的现场让路）
        # 机器人有物理体积，不能直接交换位置
        # 1. 检测挡路情况（有其他机器人想通过当前位置）
        # 2. 实现主动避让：让路的机器人移动到相邻空位
        
        # 检测挡路情况：哪些机器人挡了其他机器人的路
        blocking_info = {}  # robot_id -> [想通过当前位置的其他机器人ID列表]
        for robot_id, new_pos in new_positions.items():
            robot = self.robots[robot_id]
            current_pos = robot.position
            
            # 检查是否有其他机器人想移动到当前位置（说明当前机器人可能挡路了）
            blocking_list = []
            for other_id, other_new_pos in new_positions.items():
                if other_id != robot_id and other_new_pos == current_pos:
                    # 其他机器人想移动到当前位置，说明当前机器人可能挡路了
                    blocking_list.append(other_id)
            
            if blocking_list:
                blocking_info[robot_id] = blocking_list
        
        # 记录已占用的位置（实时更新）
        occupied_positions = {}
        # 记录在充电站充电的机器人位置（这些位置不能被其他机器人占用）
        charging_station_positions = {}
        for robot in self.robots:
            occupied_positions[robot.position] = robot.robot_id
            # 如果机器人在充电站且正在充电，标记该位置为充电站位置
            if (self.grid[robot.position[1], robot.position[0]] == 4 and 
                robot.state == RobotState.CHARGING):
                charging_station_positions[robot.position] = robot.robot_id
        
        # 按ID顺序处理机器人移动（优先级机制）
        for robot_id in sorted(new_positions.keys()):
            robot = self.robots[robot_id]
            new_pos = new_positions[robot_id]
            current_pos = robot.position
            
            # 如果目标位置被在充电站充电的机器人占用，当前机器人不能移动（让路给充电的机器人）
            if new_pos in charging_station_positions:
                # 保持原位置，不移动
                continue
            
            # 如果目标位置未被占用，允许移动
            if new_pos not in occupied_positions:
                old_pos = robot.position
                robot.update_position(new_pos)
                # 更新方向
                if new_pos != old_pos:
                    dx = new_pos[0] - old_pos[0]
                    dy = new_pos[1] - old_pos[1]
                    if dx > 0:
                        robot.direction = 1
                    elif dx < 0:
                        robot.direction = 3
                    elif dy > 0:
                        robot.direction = 2
                    elif dy < 0:
                        robot.direction = 0
                # 更新占用位置
                occupied_positions.pop(old_pos, None)
                occupied_positions[new_pos] = robot_id
            else:
                # 目标位置被占用，尝试主动避让
                # 检查是否挡了其他机器人的路
                is_blocking = robot_id in blocking_info
                blocked_robots = blocking_info.get(robot_id, [])
                
                # 礼让策略：有货（有任务）的机器人不动，没货的机器人让开道路
                is_at_charging_station = self.grid[current_pos[1], current_pos[0]] == 4
                if is_at_charging_station and robot.state == RobotState.CHARGING:
                    continue  # 充电中不让路
                if robot.current_task is not None:
                    continue  # 有货的机器人不让路，保持不动
                
                if is_blocking and blocked_robots:
                    # 仅当本机无任务时才让路（没货的机器人让开）
                    adjacent_positions = [
                        (current_pos[0] + dx, current_pos[1] + dy)
                        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
                    ]
                    random.shuffle(adjacent_positions)
                    moved = False
                    for adj_pos in adjacent_positions:
                        adj_x, adj_y = adj_pos
                        if (0 <= adj_x < self.width and 0 <= adj_y < self.height and
                            self.grid[adj_y, adj_x] not in (1, 2) and
                            adj_pos not in occupied_positions):
                            old_pos = robot.position
                            robot.update_position(adj_pos)
                            dx = adj_pos[0] - old_pos[0]
                            dy = adj_pos[1] - old_pos[1]
                            if dx > 0:
                                robot.direction = 1
                            elif dx < 0:
                                robot.direction = 3
                            elif dy > 0:
                                robot.direction = 2
                            elif dy < 0:
                                robot.direction = 0
                            occupied_positions.pop(old_pos, None)
                            occupied_positions[adj_pos] = robot_id
                            moved = True
                            break
                    if moved:
                        continue  # 已让路，跳过后续处理
                
                # 如果目标位置被占用且没有让路成功
                # 检查是否机器人在工作站且已完成任务，尝试移动到相邻空位
                is_at_workstation = self.grid[current_pos[1], current_pos[0]] == 3
                is_idle_or_completed = (robot.state == RobotState.IDLE or 
                                       (robot.current_task is None))
                
                if is_at_workstation and is_idle_or_completed:
                    # 尝试找到相邻的空位置
                    adjacent_positions = [
                        (current_pos[0] + dx, current_pos[1] + dy)
                        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
                    ]
                    
                    # 随机打乱顺序，避免所有机器人选择同一方向
                    random.shuffle(adjacent_positions)
                    
                    moved = False
                    for adj_pos in adjacent_positions:
                        adj_x, adj_y = adj_pos
                        # 检查是否在边界内且可通行（非障碍物、非货架）
                        if (0 <= adj_x < self.width and 0 <= adj_y < self.height and
                            self.grid[adj_y, adj_x] not in (1, 2) and
                            adj_pos not in occupied_positions):
                            # 找到空位，移动到该位置
                            old_pos = robot.position
                            robot.update_position(adj_pos)
                            # 更新方向
                            dx = adj_pos[0] - old_pos[0]
                            dy = adj_pos[1] - old_pos[1]
                            if dx > 0:
                                robot.direction = 1
                            elif dx < 0:
                                robot.direction = 3
                            elif dy > 0:
                                robot.direction = 2
                            elif dy < 0:
                                robot.direction = 0
                            # 更新占用位置
                            occupied_positions.pop(old_pos, None)
                            occupied_positions[adj_pos] = robot_id
                            moved = True
                            break
                
                # 如果都无法移动，保持原位置
    
    def _update_robots(self):
        """更新机器人状态"""
        for robot in self.robots:
            # 先检查是否在充电站，如果在充电站且状态不是CHARGING，自动切换到CHARGING
            if self.charging_stations:
                is_at_charging_station = any(
                    robot.is_at_position(cs, tolerance=0) for cs in self.charging_stations
                )
                if is_at_charging_station and robot.state != RobotState.CHARGING:
                    # 如果机器人在充电站但状态不是CHARGING，且没有任务或电量不足，切换到CHARGING
                    # 但是，如果机器人已经充完电（电量 >= 90%），不应该强制设置为CHARGING
                    if robot.battery / robot.max_battery < 0.9:
                        if robot.current_task is None or robot.battery / robot.max_battery < 0.1:
                            robot.state = RobotState.CHARGING
            
            # 消耗电量（充电时不消耗）
            robot.consume_battery(self.robot_config.battery_consumption_rate)
            
            # 检查是否需要充电
            # 如果机器人没有任务且电量不足，自动去充电
            if robot.current_task is None and robot.needs_charging() and robot.state != RobotState.CHARGING:
                # 寻找最近的充电站
                if self.charging_stations:
                    nearest_charging = min(self.charging_stations, 
                                         key=lambda p: robot.get_distance_to(p))
                    if robot.is_at_position(nearest_charging, tolerance=0):
                        robot.state = RobotState.CHARGING
                    else:
                        robot.state = RobotState.RETURNING_TO_CHARGE
            # 如果有任务但电量严重不足（低于20%），归还任务并去充电
            elif robot.current_task is not None and robot.battery / robot.max_battery < 0.2:
                # 归还任务
                robot.cancel_task()
                # 去充电
                if self.charging_stations:
                    nearest_charging = min(self.charging_stations, 
                                         key=lambda p: robot.get_distance_to(p))
                    if robot.is_at_position(nearest_charging, tolerance=0):
                        robot.state = RobotState.CHARGING
                    else:
                        robot.state = RobotState.RETURNING_TO_CHARGE
            
            # 充电
            if robot.state == RobotState.CHARGING:
                robot.charge(self.robot_config.charging_rate)
                if robot.battery >= robot.max_battery * 0.9:
                    # 充满电后，初始化机器人状态，清除所有任务相关状态
                    robot.state = RobotState.IDLE
                    robot.task_progress = 0.0
                    # 确保任务已被清除（双重检查）
                    if robot.current_task is not None:
                        robot.cancel_task()
            
            # 更新任务状态（只有在非充电相关状态时才更新）
            if robot.current_task and robot.state not in [RobotState.CHARGING, RobotState.RETURNING_TO_CHARGE]:
                task = robot.current_task
                
                if robot.state == RobotState.MOVING_TO_PICKUP:
                    if robot.is_at_position(task.pickup_location, tolerance=0):
                        robot.state = RobotState.PICKING_UP
                        task.status = TaskStatus.IN_PROGRESS
                
                elif robot.state == RobotState.PICKING_UP:
                    robot.task_progress = 0.5
                    robot.state = RobotState.MOVING_TO_DROPOFF
                    # 取完货后退回通道：若当前在货架上，自动移到相邻可通行格
                    px, py = robot.position
                    if self.grid[py, px] == 2:
                        occupied = {r.position for r in self.robots}
                        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                            ax, ay = px + dx, py + dy
                            if (0 <= ax < self.width and 0 <= ay < self.height and
                                    self.grid[ay, ax] not in (1, 2) and (ax, ay) not in occupied):
                                robot.update_position((ax, ay))
                                break
                
                elif robot.state == RobotState.MOVING_TO_DROPOFF:
                    if robot.is_at_position(task.dropoff_location, tolerance=0):
                        robot.state = RobotState.DROPPING_OFF
                
                elif robot.state == RobotState.DROPPING_OFF:
                    robot.task_progress = 1.0
                    robot.complete_task()
    
    def _spawn_tasks(self):
        """生成新任务：取货点在货架上，机器人需进入货架取货后退回通道"""
        if random.random() < self.warehouse_config.task_spawn_rate:
            if len(self.tasks) < self.warehouse_config.max_tasks:
                shelf_positions = []
                for region in self.warehouse_config.shelf_regions:
                    x1, y1, x2, y2 = region
                    y1 = max(0, min(y1, self.height - 1))
                    y2 = max(0, min(y2, self.height - 1))
                    x1 = max(0, min(x1, self.width - 1))
                    x2 = max(0, min(x2, self.width - 1))
                    for y in range(y1, y2 + 1):
                        for x in range(x1, x2 + 1):
                            if 0 <= y < self.height and 0 <= x < self.width and self.grid[y, x] == 2:
                                shelf_positions.append((x, y))
                
                workstation_positions = self.warehouse_config.workstation_positions
                
                if shelf_positions and workstation_positions:
                    pickup = random.choice(shelf_positions)
                    dropoff = random.choice(workstation_positions)
                    
                    task = Task(
                        task_id=self.task_counter,
                        task_type=TaskType.PICKUP,
                        pickup_location=pickup,
                        dropoff_location=dropoff,
                        priority=random.uniform(0.5, 2.0),
                        created_time=self.step_count
                    )
                    self.tasks.append(task)
                    self.task_counter += 1
    
    def _compute_rewards(self) -> Dict[int, float]:
        """计算所有机器人的奖励"""
        rewards = {}
        for robot in self.robots:
            reward, _ = self.reward_fn.compute_reward(
                robot, self.robots, self.tasks, self.step_count
            )
            rewards[robot.robot_id] = reward
        return rewards
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """获取所有机器人的观察"""
        observations = {}
        for robot in self.robots:
            obs = self.observation_space_obj.get_observation(
                robot, self.robots, self.tasks, self.grid
            )
            observations[robot.robot_id] = obs
        return observations
    
    def _get_infos(self) -> Dict[int, Dict]:
        """获取信息"""
        infos = {}
        for robot in self.robots:
            info = {
                'battery': robot.battery,
                'state': robot.state.value,
                'has_task': robot.current_task is not None,
                'position': robot.position
            }
            infos[robot.robot_id] = info
        return infos
    
    def _check_terminated(self) -> Dict[int, bool]:
        """检查终止条件"""
        # 只有当有任务且所有任务都完成时才终止
        if len(self.tasks) == 0:
            # 没有任务时，不终止（等待任务生成）
            all_tasks_completed = False
        else:
            # 所有任务完成
            all_tasks_completed = all(t.status == TaskStatus.COMPLETED 
                                     for t in self.tasks)
        terminated = {robot.robot_id: all_tasks_completed for robot in self.robots}
        return terminated
    
    def _check_truncated(self) -> Dict[int, bool]:
        """检查截断条件"""
        # 如果有未完成任务（pending 或 in_progress），不允许截断
        has_unfinished_tasks = any(
            t.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]
            for t in self.tasks
        )
        
        # 只有在没有未完成任务时才允许因步数限制而截断
        if has_unfinished_tasks:
            # 有未完成任务，不允许截断（必须完成所有任务）
            truncated = {robot.robot_id: False for robot in self.robots}
        else:
            # 所有任务都已完成或失败，可以使用步数限制
            max_steps = getattr(self.robot_config, 'max_steps_per_episode', 300)
            truncated = {robot.robot_id: self.step_count >= max_steps 
                        for robot in self.robots}
        return truncated
    
    def get_global_state(self) -> Dict[str, Any]:
        """获取全局状态（用于任务分配）"""
        return self.observation_space_obj.get_global_state(self.robots, self.tasks)
    
    def render(self):
        """渲染环境（由可视化模块处理）"""
        pass
