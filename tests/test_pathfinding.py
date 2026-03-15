"""
pathfinding 模块测试：astar_path、path_to_action
"""
import numpy as np
import pytest

from utils.pathfinding import astar_path, path_to_action


class TestAstarPath:
    """astar_path 基本与边界行为"""

    def test_start_equals_goal_returns_single_cell(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        path = astar_path(grid, 5, 5, (2, 2), (2, 2))
        assert path == [(2, 2)]

    def test_goal_start_as_list_normalized_to_tuple(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        path = astar_path(grid, 5, 5, [1, 1], [3, 3])
        assert path == [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)] or len(path) >= 2
        assert path[0] == (1, 1) and path[-1] == (3, 3)

    def test_straight_path_empty_grid(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        path = astar_path(grid, 5, 5, (0, 0), (4, 4))
        assert path[0] == (0, 0) and path[-1] == (4, 4)
        assert len(path) == 9  # 曼哈顿 8+1

    def test_obstacle_blocked_returns_empty(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[2, :] = 1  # 横墙
        path = astar_path(grid, 5, 5, (0, 0), (4, 4))
        assert path == []

    def test_goal_on_shelf_without_allow_goal_on_shelf_unchanged_behavior(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[2, 2] = 2  # 货架
        path = astar_path(grid, 5, 5, (0, 0), (2, 2), allow_goal_on_shelf=False)
        assert path == []

    def test_goal_on_shelf_with_allow_goal_on_shelf_reachable(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[2, 2] = 2
        path = astar_path(grid, 5, 5, (0, 0), (2, 2), allow_goal_on_shelf=True)
        assert path[-1] == (2, 2) and len(path) >= 2

    def test_workstation_only_passable_as_goal(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[2, 2] = 3  # 工作站
        path = astar_path(grid, 5, 5, (0, 0), (2, 2))
        assert path[-1] == (2, 2)

    def test_charging_station_only_passable_as_goal(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[2, 2] = 4
        path = astar_path(grid, 5, 5, (0, 0), (2, 2))
        assert path[-1] == (2, 2)

    def test_occupied_cells_blocked(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        occupied = {(1, 1)}  # 占用 (1,1)，路径可绕行
        path = astar_path(grid, 5, 5, (0, 0), (2, 2), occupied=occupied)
        assert (1, 1) not in path
        assert path[0] == (0, 0) and path[-1] == (2, 2)

    def test_out_of_bounds_returns_empty(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        path = astar_path(grid, 5, 5, (0, 0), (10, 10))
        assert path == []


class TestPathToAction:
    """path_to_action：路径转动作 0=上 1=右 2=下 3=左 4=等待"""

    def test_empty_path_returns_wait(self):
        assert path_to_action([]) == 4

    def test_single_cell_returns_wait(self):
        assert path_to_action([(1, 1)]) == 4

    def test_up(self):
        assert path_to_action([(1, 1), (1, 0)]) == 0

    def test_right(self):
        assert path_to_action([(1, 1), (2, 1)]) == 1

    def test_down(self):
        assert path_to_action([(1, 1), (1, 2)]) == 2

    def test_left(self):
        assert path_to_action([(1, 1), (0, 1)]) == 3

    def test_no_move_returns_wait(self):
        assert path_to_action([(1, 1), (1, 1)]) == 4
