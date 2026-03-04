"""
路径可视化脚本：绘制机器人的移动轨迹
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# 配置中文字体支持
from utils.plot_utils import setup_chinese_font
setup_chinese_font()


def load_paths(json_file: str) -> dict:
    """加载路径JSON文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 转换回整数键和元组
    paths = {}
    for rid_str, traj in data.items():
        rid = int(rid_str)
        paths[rid] = [(int(x), int(y)) for x, y in traj]
    return paths


def plot_paths(paths_file: str, grid_file: str = None, 
               save_path: str = None, show_plot: bool = True,
               warehouse_width: int = 50, warehouse_height: int = 50):
    """
    绘制机器人路径
    
    Args:
        paths_file: 路径JSON文件路径
        grid_file: 网格配置文件（可选，用于显示货架、工作站等）
        save_path: 保存图片路径（可选）
        show_plot: 是否显示图片
        warehouse_width: 仓库宽度
        warehouse_height: 仓库高度
    """
    # 加载路径
    if not os.path.exists(paths_file):
        print(f"错误：路径文件不存在: {paths_file}")
        return
    
    paths = load_paths(paths_file)
    
    if not paths:
        print("错误：路径文件为空")
        return
    
    print(f"加载了 {len(paths)} 个机器人的路径")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制网格背景
    ax.set_xlim(-0.5, warehouse_width - 0.5)
    ax.set_ylim(-0.5, warehouse_height - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.set_title('机器人移动轨迹', fontsize=14, fontweight='bold')
    
    # 绘制网格（可选，如果网格文件存在）
    # 注意：grid_file 参数目前未实现，保留用于未来扩展
    
    # 为每个机器人分配不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
    
    # 绘制每个机器人的路径
    for robot_id, traj in paths.items():
        if len(traj) < 2:
            continue
        
        color = colors[robot_id % len(colors)]
        
        # 提取x和y坐标
        x_coords = [pos[0] for pos in traj]
        y_coords = [pos[1] for pos in traj]
        
        # 绘制路径线
        ax.plot(x_coords, y_coords, color=color, linewidth=2, 
               alpha=0.6, label=f'机器人 {robot_id}')
        
        # 标记起点（绿色圆圈）
        ax.plot(x_coords[0], y_coords[0], 'o', color='green', 
               markersize=10, markeredgecolor='black', markeredgewidth=1,
               label='起点' if robot_id == 0 else '')
        
        # 标记终点（红色方块）
        ax.plot(x_coords[-1], y_coords[-1], 's', color='red', 
               markersize=10, markeredgecolor='black', markeredgewidth=1,
               label='终点' if robot_id == 0 else '')
        
        # 在路径上标记一些关键点（每10步一个点）
        if len(traj) > 10:
            for i in range(0, len(traj), max(1, len(traj) // 10)):
                ax.plot(x_coords[i], y_coords[i], 'o', color=color, 
                       markersize=4, alpha=0.5)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    
    # 添加统计信息
    stats_text = f"总机器人数: {len(paths)}\n"
    total_steps = sum(len(traj) for traj in paths.values())
    stats_text += f"总步数: {total_steps}\n"
    avg_steps = total_steps / len(paths) if paths else 0
    stats_text += f"平均步数: {avg_steps:.1f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_paths_with_grid(paths_file: str, grid: np.ndarray,
                         save_path: str = None, show_plot: bool = True):
    """
    绘制机器人路径（带网格信息）
    
    Args:
        paths_file: 路径JSON文件路径
        grid: 网格数组 (height, width)，0=空地, 1=障碍物, 2=货架, 3=工作站, 4=充电站
        save_path: 保存图片路径
        show_plot: 是否显示图片
    """
    # 加载路径
    paths = load_paths(paths_file)
    
    height, width = grid.shape
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 绘制网格
    cmap = ListedColormap(['white', 'gray', 'brown', 'blue', 'green'])
    im = ax.imshow(grid, cmap=cmap, origin='upper', 
                   extent=[-0.5, width-0.5, height-0.5, -0.5],
                   alpha=0.3, interpolation='nearest')
    
    # 添加网格标签
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.set_title('机器人移动轨迹（带网格信息）', fontsize=14, fontweight='bold')
    
    # 添加颜色条说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', label='空地'),
        Patch(facecolor='gray', label='障碍物'),
        Patch(facecolor='brown', label='货架'),
        Patch(facecolor='blue', label='工作站'),
        Patch(facecolor='green', label='充电站')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # 为每个机器人分配不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
    
    # 绘制每个机器人的路径
    for robot_id, traj in paths.items():
        if len(traj) < 2:
            continue
        
        color = colors[robot_id % len(colors)]
        
        # 提取x和y坐标
        x_coords = [pos[0] for pos in traj]
        y_coords = [pos[1] for pos in traj]
        
        # 绘制路径线
        ax.plot(x_coords, y_coords, color=color, linewidth=2.5, 
               alpha=0.7, label=f'机器人 {robot_id}', zorder=5)
        
        # 标记起点
        ax.plot(x_coords[0], y_coords[0], 'o', color='green', 
               markersize=12, markeredgecolor='black', markeredgewidth=2,
               zorder=6, label='起点' if robot_id == 0 else '')
        
        # 标记终点
        ax.plot(x_coords[-1], y_coords[-1], 's', color='red', 
               markersize=12, markeredgecolor='black', markeredgewidth=2,
               zorder=6, label='终点' if robot_id == 0 else '')
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    
    # 添加统计信息
    stats_text = f"总机器人数: {len(paths)}\n"
    total_steps = sum(len(traj) for traj in paths.values())
    stats_text += f"总步数: {total_steps}\n"
    avg_steps = total_steps / len(paths) if paths else 0
    stats_text += f"平均步数: {avg_steps:.1f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化机器人移动轨迹')
    parser.add_argument('--paths_file', type=str, 
                      default='paths_episode_1.json',
                      help='路径JSON文件路径（默认: paths_episode_1.json）')
    parser.add_argument('--save_path', type=str, default=None,
                      help='保存图片路径（默认: 不保存）')
    parser.add_argument('--no_show', action='store_true',
                      help='不显示图片（仅保存）')
    parser.add_argument('--width', type=int, default=50,
                      help='仓库宽度（默认: 50）')
    parser.add_argument('--height', type=int, default=50,
                      help='仓库高度（默认: 50）')
    
    args = parser.parse_args()
    
    # 确定保存路径
    if args.save_path is None:
        base_name = os.path.splitext(args.paths_file)[0]
        args.save_path = f"{base_name}_visualization.png"
    
    # 绘制路径
    plot_paths(
        paths_file=args.paths_file,
        save_path=args.save_path,
        show_plot=not args.no_show,
        warehouse_width=args.width,
        warehouse_height=args.height
    )


if __name__ == '__main__':
    main()
