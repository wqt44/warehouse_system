# 仓库系统测试说明

## 安装

在项目根目录或 `warehouse_system` 目录下安装依赖后，再安装测试依赖：

```bash
# 若在 PyCharmMiscProject 根目录
pip install -r warehouse_system/requirements.txt
pip install -r warehouse_system/requirements-dev.txt

# 若已在 warehouse_system 目录
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

或仅安装 pytest：

```bash
pip install pytest
```

## 运行测试

**推荐在 `warehouse_system` 目录下运行**，以便正确解析包导入（`env`、`utils`、`config` 等）：

```bash
cd warehouse_system
pytest tests/ -v
```

在项目根目录运行时，需保证能导入 `warehouse_system` 内模块（本仓库通过 `tests/conftest.py` 将 `warehouse_system` 加入 `sys.path`，因此从 `warehouse_system` 下执行 `pytest tests/` 即可）：

```bash
cd warehouse_system
pytest tests/ -v
```

### 按模块运行

- 只跑 **pathfinding** 测试：
  ```bash
  cd warehouse_system
  pytest tests/test_pathfinding.py -v
  ```

- 只跑 **warehouse_env** 测试：
  ```bash
  cd warehouse_system
  pytest tests/test_warehouse_env.py -v
  ```

- 只跑 **interactive_config** 测试：
  ```bash
  cd warehouse_system
  pytest tests/test_interactive_config.py -v
  ```

### 其他常用选项

- 显示打印输出：`pytest tests/ -v -s`
- 只跑匹配名称的用例：`pytest tests/ -v -k "astar_path"`
- 失败时进入调试：`pytest tests/ -v --pdb`

## 测试模块说明

| 模块 | 文件 | 覆盖内容 |
|------|------|----------|
| pathfinding | `test_pathfinding.py` | `astar_path`（起点=终点、list 规范化、障碍/货架/工作站/充电站/占用）、`path_to_action`（上下左右与等待） |
| warehouse_env | `test_warehouse_env.py` | `create_env`、`reset`、`step`、充电站分配 `get_charging_station_for_robot`、移动规则（等待、撞障碍不动） |
| interactive_config | `test_interactive_config.py` | `_merge_shelf_cells_to_regions`、`_sync_config_from_grid`、`_sync_grid_from_config`（不启动 Pygame 窗口，仅测逻辑） |

## 注意事项

- **interactive_config** 测试通过 mock `pygame.display.set_mode` 等避免依赖显示器；若未安装 `pygame` 会跳过该模块（`pytest.importorskip("pygame")`）。
- 运行环境需已安装 `numpy`、`gymnasium` 等（见 `requirements.txt`），否则 `warehouse_env` 测试会失败。
