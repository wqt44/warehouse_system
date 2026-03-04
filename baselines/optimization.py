"""
经典优化算法：用于对比研究
包括贪心算法、匈牙利算法、遗传算法等
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from agents.robot import Robot
from agents.task import Task, TaskStatus


class GreedyAllocator:
    """贪心任务分配算法"""
    
    def allocate(self, robots: List[Robot], tasks: List[Task]) -> Dict[int, int]:
        """
        贪心分配：每次将任务分配给距离最近的空闲机器人
        
        Returns:
            {task_id: robot_id}
        """
        allocation = {}
        pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
        
        # 按优先级排序
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # 获取空闲机器人
        idle_robots = [r for r in robots if r.state.value == "idle"]
        
        for task in pending_tasks:
            if not idle_robots:
                break
            
            # 找到距离最近的机器人
            best_robot = None
            min_distance = float('inf')
            
            for robot in idle_robots:
                distance = robot.get_distance_to(task.pickup_location)
                if distance < min_distance:
                    min_distance = distance
                    best_robot = robot
            
            if best_robot:
                allocation[task.task_id] = best_robot.robot_id
                idle_robots.remove(best_robot)
        
        return allocation


class HungarianAllocator:
    """匈牙利算法任务分配（最小化总距离）"""
    
    def allocate(self, robots: List[Robot], tasks: List[Task]) -> Dict[int, int]:
        """
        使用匈牙利算法进行最优分配
        
        Returns:
            {task_id: robot_id}
        """
        allocation = {}
        pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
        idle_robots = [r for r in robots if r.state.value == "idle"]
        
        if not pending_tasks or not idle_robots:
            return allocation
        
        # 构建成本矩阵
        num_tasks = len(pending_tasks)
        num_robots = len(idle_robots)
        
        # 如果任务数多于机器人，只考虑前num_robots个任务
        if num_tasks > num_robots:
            pending_tasks = pending_tasks[:num_robots]
            num_tasks = num_robots
        
        cost_matrix = np.zeros((num_tasks, num_robots))
        
        for i, task in enumerate(pending_tasks):
            for j, robot in enumerate(idle_robots):
                # 成本 = 距离 - 优先级奖励
                distance = robot.get_distance_to(task.pickup_location)
                cost = distance - task.priority * 10  # 优先级越高，成本越低
                cost_matrix[i, j] = cost
        
        # 使用匈牙利算法
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 构建分配结果
        for i, j in zip(row_indices, col_indices):
            task = pending_tasks[i]
            robot = idle_robots[j]
            allocation[task.task_id] = robot.robot_id
        
        return allocation


class GeneticAllocator:
    """遗传算法任务分配"""
    
    def __init__(self, population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def allocate(self, robots: List[Robot], tasks: List[Task]) -> Dict[int, int]:
        """
        使用遗传算法进行任务分配
        
        Returns:
            {task_id: robot_id}
        """
        allocation = {}
        pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
        idle_robots = [r for r in robots if r.state.value == "idle"]
        
        if not pending_tasks or not idle_robots:
            return allocation
        
        num_tasks = len(pending_tasks)
        num_robots = len(idle_robots)
        
        # 如果任务数多于机器人，只考虑前num_robots个任务
        if num_tasks > num_robots:
            pending_tasks = pending_tasks[:num_robots]
            num_tasks = num_robots
        
        # 初始化种群
        population = self._initialize_population(num_tasks, num_robots)
        
        # 进化
        for generation in range(self.generations):
            # 评估适应度
            fitness_scores = self._evaluate_population(
                population, pending_tasks, idle_robots
            )
            
            # 选择
            population = self._select(population, fitness_scores)
            
            # 交叉
            population = self._crossover(population)
            
            # 变异
            population = self._mutate(population, num_robots)
        
        # 选择最优解
        final_fitness = self._evaluate_population(
            population, pending_tasks, idle_robots
        )
        best_idx = np.argmax(final_fitness)
        best_solution = population[best_idx]
        
        # 构建分配结果
        for i, robot_idx in enumerate(best_solution):
            if i < len(pending_tasks):
                task = pending_tasks[i]
                robot = idle_robots[robot_idx]
                allocation[task.task_id] = robot.robot_id
        
        return allocation
    
    def _initialize_population(self, num_tasks: int, num_robots: int) -> np.ndarray:
        """初始化种群"""
        population = np.random.randint(0, num_robots, 
                                     size=(self.population_size, num_tasks))
        return population
    
    def _evaluate_population(self, population: np.ndarray,
                            tasks: List[Task], robots: List[Robot]) -> np.ndarray:
        """评估种群适应度"""
        fitness_scores = np.zeros(len(population))
        
        for i, solution in enumerate(population):
            total_cost = 0.0
            for j, robot_idx in enumerate(solution):
                if j < len(tasks):
                    task = tasks[j]
                    robot = robots[robot_idx]
                    distance = robot.get_distance_to(task.pickup_location)
                    # 适应度 = 负成本（距离越小，优先级越高，适应度越高）
                    cost = distance - task.priority * 10
                    total_cost += cost
            
            fitness_scores[i] = -total_cost  # 负成本作为适应度
        
        return fitness_scores
    
    def _select(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """选择（轮盘赌）"""
        # 归一化适应度
        fitness_scores = fitness_scores - fitness_scores.min() + 1e-8
        probs = fitness_scores / fitness_scores.sum()
        
        # 选择
        selected_indices = np.random.choice(
            len(population), size=self.population_size, p=probs
        )
        
        return population[selected_indices]
    
    def _crossover(self, population: np.ndarray) -> np.ndarray:
        """交叉"""
        new_population = []
        
        for i in range(0, len(population), 2):
            if i + 1 >= len(population):
                new_population.append(population[i])
                break
            
            parent1 = population[i]
            parent2 = population[i + 1]
            
            if np.random.random() < self.crossover_rate and len(parent1) > 1:
                # 单点交叉（只有当长度大于1时才进行交叉）
                crossover_point = np.random.randint(1, len(parent1))
                child1 = np.concatenate([parent1[:crossover_point], 
                                       parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], 
                                       parent1[crossover_point:]])
                new_population.extend([child1, child2])
            else:
                # 不进行交叉，直接复制父代
                new_population.extend([parent1, parent2])
        
        return np.array(new_population)
    
    def _mutate(self, population: np.ndarray, num_robots: int) -> np.ndarray:
        """变异"""
        for i in range(len(population)):
            if np.random.random() < self.mutation_rate:
                # 随机改变一个基因
                mutation_point = np.random.randint(len(population[i]))
                population[i][mutation_point] = np.random.randint(0, num_robots)
        
        return population


class BaselineComparison:
    """基线算法对比类"""
    
    def __init__(self):
        self.allocators = {
            'greedy': GreedyAllocator(),
            'hungarian': HungarianAllocator(),
            'genetic': GeneticAllocator()
        }
    
    def compare(self, robots: List[Robot], tasks: List[Task]) -> Dict[str, Dict[int, int]]:
        """
        对比所有基线算法
        
        Returns:
            {algorithm_name: allocation_dict}
        """
        results = {}
        for name, allocator in self.allocators.items():
            results[name] = allocator.allocate(robots, tasks)
        
        return results
