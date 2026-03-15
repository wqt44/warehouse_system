"""
interactive_config 模块测试：不依赖 Pygame 显示的逻辑（config 与 grid 同步、货架合并）
需 mock pygame 以便无头环境运行。
"""
import pytest
from unittest.mock import MagicMock, patch

# 无显示时跳过：若未安装 pygame 或无法 mock 则 skip
pytest.importorskip("pygame")

from interactive_config import (
    InteractiveConfigEditor,
    CELL_EMPTY,
    CELL_SHELF,
    CELL_WORKSTATION,
    CELL_OBSTACLE,
    CELL_CHARGING,
)
from utils.config_loader import get_empty_warehouse_config


def _mock_pygame_for_editor():
    """Mock pygame 以便在无显示环境下实例化 InteractiveConfigEditor"""
    with patch("pygame.init"):
        with patch("pygame.display.set_mode", return_value=MagicMock()):
            with patch("pygame.display.set_caption"):
                with patch("interactive_config.get_chinese_font", return_value=MagicMock()):
                    with patch("pygame.time.Clock", return_value=MagicMock()):
                        yield


class TestMergeShelfCellsToRegions:
    """_merge_shelf_cells_to_regions：货架单元格合并为矩形区域"""

    def test_empty_cells_returns_empty_list(self):
        with patch("pygame.init"), patch("pygame.display.set_mode", return_value=MagicMock()), \
             patch("pygame.display.set_caption"), patch("interactive_config.get_chinese_font", return_value=MagicMock()), \
             patch("pygame.time.Clock", return_value=MagicMock()):
            editor = InteractiveConfigEditor.__new__(InteractiveConfigEditor)
            editor.config_dict = {"warehouse": get_empty_warehouse_config()}
            editor.config_dict["warehouse"]["width"] = 10
            editor.config_dict["warehouse"]["height"] = 10
            editor.grid_w = 10
            editor.grid_h = 10
            editor.edit_grid = [[CELL_EMPTY] * 10 for _ in range(10)]
            regions = InteractiveConfigEditor._merge_shelf_cells_to_regions(editor, set())
            assert regions == []

    def test_single_cell_one_region(self):
        with patch("pygame.init"), patch("pygame.display.set_mode", return_value=MagicMock()), \
             patch("pygame.display.set_caption"), patch("interactive_config.get_chinese_font", return_value=MagicMock()), \
             patch("pygame.time.Clock", return_value=MagicMock()):
            editor = InteractiveConfigEditor.__new__(InteractiveConfigEditor)
            regions = InteractiveConfigEditor._merge_shelf_cells_to_regions(editor, {(3, 3)})
            assert regions == [[3, 3, 3, 3]]

    def test_connected_cells_one_region(self):
        with patch("pygame.init"), patch("pygame.display.set_mode", return_value=MagicMock()), \
             patch("pygame.display.set_caption"), patch("interactive_config.get_chinese_font", return_value=MagicMock()), \
             patch("pygame.time.Clock", return_value=MagicMock()):
            editor = InteractiveConfigEditor.__new__(InteractiveConfigEditor)
            cells = {(1, 1), (2, 1), (1, 2), (2, 2)}
            regions = InteractiveConfigEditor._merge_shelf_cells_to_regions(editor, cells)
            assert len(regions) == 1
            assert regions[0] == [1, 1, 2, 2]


class TestSyncConfigFromGrid:
    """_sync_config_from_grid：从 edit_grid 同步回 config_dict"""

    def test_sync_config_from_grid_updates_warehouse_keys(self):
        with patch("pygame.init"), patch("pygame.display.set_mode", return_value=MagicMock()), \
             patch("pygame.display.set_caption"), patch("interactive_config.get_chinese_font", return_value=MagicMock()), \
             patch("pygame.time.Clock", return_value=MagicMock()):
            editor = InteractiveConfigEditor.__new__(InteractiveConfigEditor)
            base = get_empty_warehouse_config()
            base["width"] = 5
            base["height"] = 5
            editor.config_dict = {"warehouse": base}
            editor.grid_w = 5
            editor.grid_h = 5
            editor.edit_grid = [
                [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
                [CELL_EMPTY, CELL_SHELF, CELL_SHELF, CELL_EMPTY, CELL_EMPTY],
                [CELL_EMPTY, CELL_SHELF, CELL_SHELF, CELL_EMPTY, CELL_EMPTY],
                [CELL_EMPTY, CELL_EMPTY, CELL_WORKSTATION, CELL_EMPTY, CELL_EMPTY],
                [CELL_OBSTACLE, CELL_EMPTY, CELL_EMPTY, CELL_CHARGING, CELL_EMPTY],
            ]
            InteractiveConfigEditor._sync_config_from_grid(editor)
            wh = editor.config_dict["warehouse"]
            assert "shelf_regions" in wh
            assert wh["shelf_regions"] == [[1, 1, 2, 2]]
            assert wh["workstation_positions"] == [[2, 3]]
            assert wh["obstacle_positions"] == [[0, 4]]
            assert wh["charging_stations"] == [[3, 4]]


class TestSyncGridFromConfig:
    """_sync_grid_from_config：从 config_dict 同步到 edit_grid"""

    def test_sync_grid_from_config_fills_edit_grid(self):
        with patch("pygame.init"), patch("pygame.display.set_mode", return_value=MagicMock()), \
             patch("pygame.display.set_caption"), patch("interactive_config.get_chinese_font", return_value=MagicMock()), \
             patch("pygame.time.Clock", return_value=MagicMock()):
            editor = InteractiveConfigEditor.__new__(InteractiveConfigEditor)
            base = get_empty_warehouse_config()
            base["width"] = 4
            base["height"] = 4
            base["shelf_regions"] = [[1, 1, 2, 2]]
            base["workstation_positions"] = [[0, 0]]
            base["obstacle_positions"] = [[3, 3]]
            base["charging_stations"] = [[2, 2]]
            editor.config_dict = {"warehouse": base}
            InteractiveConfigEditor._sync_grid_from_config(editor)
            assert editor.grid_w == 4 and editor.grid_h == 4
            assert editor.edit_grid[0][0] == CELL_WORKSTATION
            assert editor.edit_grid[1][1] == editor.edit_grid[1][2] == CELL_SHELF
            assert editor.edit_grid[2][2] == CELL_CHARGING
            assert editor.edit_grid[3][3] == CELL_OBSTACLE
