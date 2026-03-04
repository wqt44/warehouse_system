"""
可交互仓库配置编辑器
支持通过图形界面配置货架区域、工作站、障碍物、充电站等，并保存/加载配置
"""
import pygame
import json
import sys
import os
import platform
from typing import List, Tuple, Optional, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import (
    save_config_to_json, load_config_from_json,
    get_default_warehouse_config, get_empty_warehouse_config,
)


# 单元格类型常量（与 warehouse_env 一致）
CELL_EMPTY = 0
CELL_OBSTACLE = 1
CELL_SHELF = 2
CELL_WORKSTATION = 3
CELL_CHARGING = 4

# 编辑模式
MODE_SHELF = "shelf"       # 货架区域（拖拽矩形）
MODE_WORKSTATION = "ws"    # 工作站
MODE_OBSTACLE = "obs"      # 障碍物
MODE_CHARGING = "charge"   # 充电站
MODE_ERASE = "erase"       # 橡皮擦（删除）

# 颜色定义
COLORS = {
    'empty': (255, 255, 255),
    'obstacle': (100, 100, 100),
    'shelf': (139, 69, 19),
    'workstation': (0, 100, 200),
    'charging': (0, 180, 80),
    'grid_line': (220, 220, 220),
    'panel_bg': (245, 245, 250),
    'button': (70, 130, 180),
    'button_hover': (100, 149, 237),
    'text': (30, 30, 30),
    'text_light': (100, 100, 100),
    'selection': (255, 200, 0),
}


def get_chinese_font(size: int = 18):
    """获取支持中文的 Pygame 字体"""
    if platform.system() == 'Windows':
        try:
            return pygame.font.SysFont('Microsoft YaHei', size)
        except Exception:
            pass
    try:
        return pygame.font.SysFont('microsoftyahei,simhei', size)
    except Exception:
        return pygame.font.Font(None, size)


class Button:
    """简单按钮组件"""
    def __init__(self, x: int, y: int, w: int, h: int, text: str, font):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.hover = False
    
    def draw(self, screen: pygame.Surface):
        color = COLORS['button_hover'] if self.hover else COLORS['button']
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, (60, 100, 140), self.rect, 1, border_radius=4)
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        tr = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, tr)
    
    def update(self, mouse_pos: Tuple[int, int]) -> bool:
        self.hover = self.rect.collidepoint(mouse_pos)
        return self.hover
    
    def clicked(self, mouse_pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)


class Slider:
    """简单滑块组件"""
    def __init__(self, x: int, y: int, w: int, label: str, min_val: int, max_val: int, 
                 default: int, font):
        self.rect = pygame.Rect(x, y, w, 24)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = default
        self.font = font
        self.dragging = False
    
    def draw(self, screen: pygame.Surface):
        # 标签
        text_surf = self.font.render(f"{self.label}: {self.value}", True, COLORS['text'])
        screen.blit(text_surf, (self.rect.x, self.rect.y - 20))
        # 轨道
        pygame.draw.rect(screen, (200, 200, 200), self.rect, border_radius=2)
        # 滑块
        t = (self.value - self.min_val) / max(1, self.max_val - self.min_val)
        knob_x = self.rect.x + int(t * (self.rect.w - 16))
        knob_rect = pygame.Rect(knob_x, self.rect.y - 2, 16, self.rect.h + 4)
        pygame.draw.rect(screen, COLORS['button'], knob_rect, border_radius=4)
    
    def handle_event(self, event: pygame.event.Event, mouse_pos: Tuple[int, int]) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                self.dragging = True
                self._set_value_from_x(mouse_pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_value_from_x(mouse_pos[0])
            return True
        return False
    
    def _set_value_from_x(self, x: int):
        rel = x - self.rect.x
        t = max(0, min(1, rel / max(1, self.rect.w)))
        self.value = int(self.min_val + t * (self.max_val - self.min_val))


class InteractiveConfigEditor:
    """可交互仓库配置编辑器"""
    
    def __init__(self, width: int = 1000, height: int = 700, cell_size: int = 14):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("仓库配置编辑器 - 交互式设计")
        
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # 配置数据（可直接用于 create_env）
        self.config_dict: Dict[str, Any] = self._get_default_config()
        self._sync_grid_from_config()
        
        # 编辑状态
        self.mode = MODE_SHELF
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None
        self.erase_dragging: bool = False
        
        # 字体
        self.font = get_chinese_font(16)
        self.small_font = get_chinese_font(14)
        
        # 布局
        self.grid_width_px = 0
        self.grid_height_px = 0
        self.panel_x = 0
        self._update_layout()
        
        # 创建 UI 组件
        self._create_ui_components()
        
        self.clock = pygame.time.Clock()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置（优先从 config.json 加载）"""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            try:
                cfg = load_config_from_json(config_path)
                if cfg and ('warehouse' in cfg or 'robot' in cfg):
                    return {
                        'warehouse': cfg.get('warehouse', get_empty_warehouse_config()),
                        'robot': cfg.get('robot', {'num_robots': 5, 'max_battery': 100.0,
                                    'battery_consumption_rate': 0.1, 'charging_rate': 2.0}),
                        'max_steps': cfg.get('max_steps', 300),
                    }
            except Exception:
                pass
        base = get_empty_warehouse_config()
        return {
            'warehouse': base,
            'robot': {'num_robots': 5, 'max_battery': 100.0,
                      'battery_consumption_rate': 0.1, 'charging_rate': 2.0},
            'max_steps': 300,
        }
    
    def _sync_grid_from_config(self):
        """从 config_dict 同步到内部 grid 表示（用于编辑时的可视化）"""
        wh = self.config_dict['warehouse']
        w = wh['width']
        h = wh['height']
        self.grid_w = w
        self.grid_h = h
        self.edit_grid = [[CELL_EMPTY for _ in range(w)] for _ in range(h)]
        
        # 货架区域
        for region in wh.get('shelf_regions', []):
            x1, y1, x2, y2 = region[0], region[1], region[2], region[3]
            for y in range(max(0, y1), min(h, y2 + 1)):
                for x in range(max(0, x1), min(w, x2 + 1)):
                    self.edit_grid[y][x] = CELL_SHELF
        
        # 工作站
        for pos in wh.get('workstation_positions', []):
            x, y = pos[0], pos[1]
            if 0 <= y < h and 0 <= x < w:
                self.edit_grid[y][x] = CELL_WORKSTATION
        
        # 障碍物
        for pos in wh.get('obstacle_positions', []):
            x, y = pos[0], pos[1]
            if 0 <= y < h and 0 <= x < w:
                self.edit_grid[y][x] = CELL_OBSTACLE
        
        # 充电站
        stations = wh.get('charging_stations')
        if stations:
            for pos in stations:
                x, y = pos[0], pos[1]
                if 0 <= y < h and 0 <= x < w:
                    self.edit_grid[y][x] = CELL_CHARGING
    
    def _sync_config_from_grid(self):
        """从 edit_grid 同步回 config_dict"""
        wh = self.config_dict['warehouse']
        w = self.grid_w
        h = self.grid_h
        
        # 重建货架区域（简单合并相邻货架为矩形区域）
        shelf_cells = set()
        for y in range(h):
            for x in range(w):
                if self.edit_grid[y][x] == CELL_SHELF:
                    shelf_cells.add((x, y))
        
        shelf_regions = self._merge_shelf_cells_to_regions(shelf_cells)
        wh['shelf_regions'] = shelf_regions
        
        workstation_positions = []
        obstacle_positions = []
        charging_positions = []
        for y in range(h):
            for x in range(w):
                v = self.edit_grid[y][x]
                if v == CELL_WORKSTATION:
                    workstation_positions.append([x, y])
                elif v == CELL_OBSTACLE:
                    obstacle_positions.append([x, y])
                elif v == CELL_CHARGING:
                    charging_positions.append([x, y])
        
        wh['workstation_positions'] = workstation_positions
        wh['obstacle_positions'] = obstacle_positions
        wh['charging_stations'] = charging_positions if charging_positions else None
    
    def _merge_shelf_cells_to_regions(self, cells: set) -> List[List[int]]:
        """将货架单元格合并为矩形区域（简化实现：每个连通分量取外接矩形）"""
        if not cells:
            return []
        # 简单策略：按行合并，找出连续段
        regions = []
        used = set()
        cells_list = sorted(cells)
        
        while cells_list:
            start = cells_list.pop(0)
            if start in used:
                continue
            x1, y1 = start
            x2, y2 = start
            stack = [start]
            used.add(start)
            while stack:
                x, y = stack.pop()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in cells and (nx, ny) not in used:
                        used.add((nx, ny))
                        stack.append((nx, ny))
                        x1 = min(x1, nx)
                        y1 = min(y1, ny)
                        x2 = max(x2, nx)
                        y2 = max(y2, ny)
            regions.append([x1, y1, x2, y2])
        
        return regions
    
    def _update_layout(self):
        """更新布局尺寸"""
        wh = self.config_dict['warehouse']
        gw = wh['width']
        gh = wh['height']
        self.grid_width_px = gw * self.cell_size
        self.grid_height_px = gh * self.cell_size
        self.panel_width = 260
        self.panel_x = min(self.grid_width_px + 20, self.width - self.panel_width - 20)
    
    def _create_ui_components(self):
        """创建按钮和滑块"""
        px = self.panel_x + 10
        py = 120
        btn_w = self.panel_width - 20
        btn_h = 32
        
        self.btn_shelf = Button(px, py, btn_w, btn_h, "货架区域(拖拽)", self.font)
        self.btn_workstation = Button(px, py + 40, btn_w, btn_h, "工作站(点击)", self.font)
        self.btn_obstacle = Button(px, py + 80, btn_w, btn_h, "障碍物(点击)", self.font)
        self.btn_charging = Button(px, py + 120, btn_w, btn_h, "充电站(点击)", self.font)
        self.btn_erase = Button(px, py + 160, btn_w, btn_h, "橡皮擦(拖拽)", self.font)
        
        self.btn_clear = Button(px, py + 220, btn_w, btn_h, "清空全部", self.font)
        self.btn_load = Button(px, py + 260, btn_w, btn_h, "加载配置", self.font)
        self.btn_save = Button(px, py + 300, btn_w, btn_h, "保存配置", self.font)
        self.btn_run = Button(px, py + 350, btn_w, btn_h, "运行推理", self.font)
        
        self.slider_width = Slider(px, py + 440, btn_w, "网格宽度", 10, 80, 
                                   self.config_dict['warehouse']['width'], self.font)
        self.slider_height = Slider(px, py + 490, btn_w, "网格高度", 10, 80,
                                    self.config_dict['warehouse']['height'], self.font)
        self.slider_robots = Slider(px, py + 540, btn_w, "机器人数", 1, 30,
                                    self.config_dict['robot']['num_robots'], self.font)
        
        self.buttons = [
            self.btn_shelf, self.btn_workstation, self.btn_obstacle,
            self.btn_charging, self.btn_erase,
            self.btn_clear, self.btn_load, self.btn_save, self.btn_run
        ]
        self.sliders = [self.slider_width, self.slider_height, self.slider_robots]
    
    def _screen_to_grid(self, sx: int, sy: int) -> Optional[Tuple[int, int]]:
        """屏幕坐标转网格坐标"""
        if sx < 0 or sy < 0:
            return None
        gx = sx // self.cell_size
        gy = sy // self.cell_size
        if gx >= self.grid_w or gy >= self.grid_h:
            return None
        return (gx, gy)
    
    def _get_cell_color(self, cell_value: int) -> Tuple[int, int, int]:
        if cell_value == CELL_EMPTY:
            return COLORS['empty']
        elif cell_value == CELL_OBSTACLE:
            return COLORS['obstacle']
        elif cell_value == CELL_SHELF:
            return COLORS['shelf']
        elif cell_value == CELL_WORKSTATION:
            return COLORS['workstation']
        elif cell_value == CELL_CHARGING:
            return COLORS['charging']
        return COLORS['empty']
    
    def _place_cell(self, gx: int, gy: int, value: int):
        """放置/覆盖单元格"""
        if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
            self.edit_grid[gy][gx] = value
            self._sync_config_from_grid()
    
    def _draw_grid(self):
        """绘制编辑网格"""
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                rect = pygame.Rect(
                    x * self.cell_size, y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                color = self._get_cell_color(self.edit_grid[y][x])
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLORS['grid_line'], rect, 1)
        
        # 绘制拖拽预览
        if self.mode == MODE_SHELF and self.drag_start and self.drag_end:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            rx1 = min(x1, x2) * self.cell_size
            ry1 = min(y1, y2) * self.cell_size
            rw = (abs(x2 - x1) + 1) * self.cell_size
            rh = (abs(y2 - y1) + 1) * self.cell_size
            s = pygame.Surface((rw, rh))
            s.set_alpha(120)
            s.fill(COLORS['shelf'])
            self.screen.blit(s, (rx1, ry1))
            pygame.draw.rect(self.screen, COLORS['selection'], (rx1, ry1, rw, rh), 2)
    
    def _draw_panel(self):
        """绘制右侧面板"""
        panel_rect = pygame.Rect(self.panel_x, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, COLORS['panel_bg'], panel_rect)
        pygame.draw.line(self.screen, (200, 200, 200), (self.panel_x, 0), (self.panel_x, self.height))
        
        title = self.font.render("仓库配置", True, COLORS['text'])
        self.screen.blit(title, (self.panel_x + 15, 20))
        
        mode_text = self.small_font.render(f"当前: {self._mode_name()}", True, COLORS['text_light'])
        self.screen.blit(mode_text, (self.panel_x + 15, 50))
        
        for btn in self.buttons:
            btn.draw(self.screen)
        for s in self.sliders:
            s.draw(self.screen)
        
        hint = self.small_font.render("右键: 删除  |  保存为 config.json", True, COLORS['text_light'])
        self.screen.blit(hint, (self.panel_x + 10, self.height - 30))
    
    def _mode_name(self) -> str:
        names = {
            MODE_SHELF: "货架区域",
            MODE_WORKSTATION: "工作站",
            MODE_OBSTACLE: "障碍物",
            MODE_CHARGING: "充电站",
            MODE_ERASE: "橡皮擦",
        }
        return names.get(self.mode, "")
    
    def _handle_grid_click(self, gx: int, gy: int, button: int):
        """处理网格点击"""
        if button == 3:  # 右键：删除
            self._place_cell(gx, gy, CELL_EMPTY)
            return
        
        if self.mode == MODE_SHELF:
            # 货架由拖拽处理
            pass
        elif self.mode == MODE_WORKSTATION:
            self._place_cell(gx, gy, CELL_WORKSTATION)
        elif self.mode == MODE_OBSTACLE:
            self._place_cell(gx, gy, CELL_OBSTACLE)
        elif self.mode == MODE_CHARGING:
            self._place_cell(gx, gy, CELL_CHARGING)
        elif self.mode == MODE_ERASE:
            self._place_cell(gx, gy, CELL_EMPTY)  # 点击或拖拽途经时擦除
    
    def _handle_shelf_drag_end(self):
        """处理货架区域拖拽结束"""
        if not self.drag_start or not self.drag_end:
            return
        x1, y1 = self.drag_start
        x2, y2 = self.drag_end
        rx1, rx2 = min(x1, x2), max(x1, x2)
        ry1, ry2 = min(y1, y2), max(y1, y2)
        for y in range(ry1, ry2 + 1):
            for x in range(rx1, rx2 + 1):
                if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                    self.edit_grid[y][x] = CELL_SHELF
        self._sync_config_from_grid()
        self.drag_start = None
        self.drag_end = None
    
    def _run_inference(self):
        """运行推理（使用当前配置）"""
        try:
            from inference import run_inference
            # 先保存配置，再运行
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            save_config_to_json(self.config_dict, config_path)
            run_inference(
                model_path=os.path.join(os.path.dirname(__file__), 'checkpoints', 'final_model.pth'),
                num_episodes=1,
                num_robots=self.config_dict['robot']['num_robots'],
                max_tasks=self.config_dict['warehouse'].get('max_tasks', 20),
                render=True,
                config_dict=self.config_dict,
            )
        except Exception as e:
            print(f"运行推理失败: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """主循环"""
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            in_grid = mouse_pos[0] < self.grid_width_px and mouse_pos[1] < self.grid_height_px
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self._update_layout()
                    self._create_ui_components()
                
                # 滑块
                for s in self.sliders:
                    if s.handle_event(event, mouse_pos):
                        break
                
                # 应用滑块值
                if self.slider_width.value != self.config_dict['warehouse']['width'] or \
                   self.slider_height.value != self.config_dict['warehouse']['height']:
                    old_w, old_h = self.config_dict['warehouse']['width'], self.config_dict['warehouse']['height']
                    new_w = self.slider_width.value
                    new_h = self.slider_height.value
                    if new_w != old_w or new_h != old_h:
                        self.config_dict['warehouse']['width'] = new_w
                        self.config_dict['warehouse']['height'] = new_h
                        self.config_dict['warehouse']['shelf_regions'] = []
                        self.config_dict['warehouse']['workstation_positions'] = []
                        self.config_dict['warehouse']['obstacle_positions'] = []
                        self.config_dict['warehouse']['charging_stations'] = None
                        self._sync_grid_from_config()
                        self._update_layout()
                        self._create_ui_components()
                
                self.config_dict['robot']['num_robots'] = self.slider_robots.value
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # 按钮点击
                    if self.btn_shelf.clicked(mouse_pos):
                        self.mode = MODE_SHELF
                    elif self.btn_workstation.clicked(mouse_pos):
                        self.mode = MODE_WORKSTATION
                    elif self.btn_obstacle.clicked(mouse_pos):
                        self.mode = MODE_OBSTACLE
                    elif self.btn_charging.clicked(mouse_pos):
                        self.mode = MODE_CHARGING
                    elif self.btn_erase.clicked(mouse_pos):
                        self.mode = MODE_ERASE
                    elif self.btn_clear.clicked(mouse_pos):
                        self.config_dict = {
                            'warehouse': get_empty_warehouse_config(),
                            'robot': {'num_robots': 5, 'max_battery': 100.0,
                                      'battery_consumption_rate': 0.1, 'charging_rate': 2.0},
                            'max_steps': 300,
                        }
                        self._sync_grid_from_config()
                        self._update_layout()
                        self._create_ui_components()
                    elif self.btn_load.clicked(mouse_pos):
                        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
                        if os.path.exists(config_path):
                            self.config_dict = load_config_from_json(config_path)
                            self._sync_grid_from_config()
                            self._update_layout()
                            self._create_ui_components()
                            print("已加载 config.json")
                        else:
                            print("未找到 config.json")
                    elif self.btn_save.clicked(mouse_pos):
                        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
                        save_config_to_json(self.config_dict, config_path)
                        print("已保存到 config.json")
                    elif self.btn_run.clicked(mouse_pos):
                        self._run_inference()
                    elif in_grid:
                        cell = self._screen_to_grid(mouse_pos[0], mouse_pos[1])
                        if cell:
                            if self.mode == MODE_SHELF and event.button == 1:
                                self.drag_start = cell
                                self.drag_end = cell
                            elif self.mode == MODE_ERASE and event.button == 1:
                                self.erase_dragging = True
                                self._place_cell(cell[0], cell[1], CELL_EMPTY)
                            else:
                                self._handle_grid_click(cell[0], cell[1], event.button)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        if self.mode == MODE_SHELF and self.drag_start:
                            cell = self._screen_to_grid(mouse_pos[0], mouse_pos[1]) if in_grid else None
                            if cell:
                                self.drag_end = cell
                            self._handle_shelf_drag_end()
                        elif self.mode == MODE_ERASE:
                            self.erase_dragging = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.mode == MODE_SHELF and self.drag_start and pygame.mouse.get_pressed()[0]:
                        cell = self._screen_to_grid(mouse_pos[0], mouse_pos[1]) if in_grid else None
                        if cell:
                            self.drag_end = cell
                    elif self.mode == MODE_ERASE and self.erase_dragging and pygame.mouse.get_pressed()[0]:
                        cell = self._screen_to_grid(mouse_pos[0], mouse_pos[1]) if in_grid else None
                        if cell:
                            self._place_cell(cell[0], cell[1], CELL_EMPTY)
                    
                    for btn in self.buttons:
                        btn.update(mouse_pos)
            
            # 绘制
            self.screen.fill((250, 250, 252))
            self._draw_grid()
            self._draw_panel()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    editor = InteractiveConfigEditor(width=1200, height=750, cell_size=16)
    editor.run()


if __name__ == '__main__':
    main()
