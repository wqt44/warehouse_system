"""
网格 A* 寻路：供启发式导航使用，使机器人能准确进入目标通道
"""
import heapq
from typing import List, Tuple, Set, Optional


def astar_path(
    grid,
    width: int,
    height: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    occupied: Optional[Set[Tuple[int, int]]] = None,
    allow_goal_on_shelf: bool = False,
) -> List[Tuple[int, int]]:
    """
    A* 寻路，返回从 start 到 goal 的路径（含起点和终点）。
    若无法到达则返回空列表。

    Args:
        grid: 二维数组，0=空地 1=障碍 2=货架 3=工作站 4=充电站
        width, height: 网格尺寸
        start: 起点 (x, y)
        goal: 终点 (x, y)
        occupied: 被占用的格子（如其他机器人位置），不进入
        allow_goal_on_shelf: 若 True，则 goal 所在格视为可通行（用于取货点在货架上）
    """
    occupied = occupied or set()
    if start == goal:
        return [start]

    def passable(x: int, y: int) -> bool:
        if not (0 <= x < width and 0 <= y < height):
            return False
        if (x, y) in occupied:
            return False
        if (x, y) == goal and allow_goal_on_shelf:
            return True
        v = grid[y, x]
        return v not in (1, 2)  # 可通行：非障碍、非货架（或上面已处理 goal）

    # (f, counter, (x,y), path) 用 counter 避免比较 tuple
    counter = 0
    open_heap = [(0, counter, start, [start])]
    closed: Set[Tuple[int, int]] = set()
    while open_heap:
        f, _, (x, y), path = heapq.heappop(open_heap)
        if (x, y) in closed:
            continue
        closed.add((x, y))
        if (x, y) == goal:
            return path
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in closed:
                continue
            if not passable(nx, ny):
                continue
            new_path = path + [(nx, ny)]
            g = len(new_path) - 1
            h = abs(nx - goal[0]) + abs(ny - goal[1])
            counter += 1
            heapq.heappush(open_heap, (g + h, counter, (nx, ny), new_path))
    return []


def path_to_action(path: List[Tuple[int, int]]) -> int:
    """
    根据路径的前两步得到动作：0=上 1=右 2=下 3=左 4=等待
    若路径长度不足 2，返回 4（等待）。
    """
    if len(path) < 2:
        return 4
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    if dx == 1:
        return 1
    if dx == -1:
        return 3
    if dy == 1:
        return 2
    if dy == -1:
        return 0
    return 4
