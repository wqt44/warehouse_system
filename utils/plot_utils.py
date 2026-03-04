"""
绘图工具：配置matplotlib中文字体支持
"""
import matplotlib.pyplot as plt
import matplotlib
import platform

# 全局变量，标记是否已配置字体
_font_configured = False


def setup_chinese_font(verbose: bool = True):
    """
    配置matplotlib支持中文显示
    
    在Windows系统上使用SimHei（黑体）或Microsoft YaHei（微软雅黑）
    在Linux/Mac系统上使用WenQuanYi Micro Hei或其他可用字体
    
    Args:
        verbose: 是否打印配置信息
    """
    global _font_configured
    
    # 如果已经配置过，直接返回
    if _font_configured:
        return True
    
    system = platform.system()
    
    # 尝试的中文字体列表（按优先级排序）
    chinese_fonts = []
    
    if system == 'Windows':
        # Windows系统常见中文字体
        chinese_fonts = [
            'Microsoft YaHei',      # 微软雅黑
            'SimHei',              # 黑体
            'SimSun',               # 宋体
            'KaiTi',                # 楷体
            'FangSong',             # 仿宋
        ]
    elif system == 'Darwin':  # macOS
        chinese_fonts = [
            'PingFang SC',          # 苹方
            'STHeiti',              # 华文黑体
            'Arial Unicode MS',     # Arial Unicode MS
        ]
    else:  # Linux
        chinese_fonts = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',     # 文泉驿正黑
            'Noto Sans CJK SC',      # Noto Sans CJK
            'Droid Sans Fallback',   # Droid Sans Fallback
        ]
    
    # 尝试设置字体
    font_set = False
    selected_font = None
    
    for font_name in chinese_fonts:
        try:
            # 设置matplotlib的默认字体
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            # 解决负号显示问题
            plt.rcParams['axes.unicode_minus'] = False
            
            # 验证字体是否可用（通过尝试创建FontProperties）
            from matplotlib.font_manager import FontProperties
            test_font = FontProperties(family=font_name)
            # 如果字体名称匹配或包含，则认为设置成功
            if font_name in test_font.get_name() or test_font.get_name() == font_name:
                font_set = True
                selected_font = font_name
                if verbose:
                    print(f"已设置中文字体: {font_name}")
                break
        except Exception:
            # 如果字体不可用，继续尝试下一个
            continue
    
    if not font_set:
        # 如果所有字体都不可用，尝试使用系统默认字体
        try:
            # 获取系统所有可用字体
            from matplotlib.font_manager import fontManager
            available_fonts = [f.name for f in fontManager.ttflist]
            
            # 查找包含中文字符的字体
            chinese_keywords = ['Chinese', 'CJK', 'Hei', 'Song', 'Kai', 'Ming', 'YaHei', 'SimHei']
            for keyword in chinese_keywords:
                for font in available_fonts:
                    if keyword.lower() in font.lower():
                        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                        plt.rcParams['axes.unicode_minus'] = False
                        font_set = True
                        selected_font = font
                        if verbose:
                            print(f"已设置中文字体: {font}")
                        break
                if font_set:
                    break
            
            if not font_set and verbose:
                print("警告: 未找到可用的中文字体，中文可能显示为方块")
                print("建议: 安装中文字体或使用英文标签")
        except Exception as e:
            if verbose:
                print(f"警告: 配置中文字体时出错: {e}")
                print("中文可能显示为方块")
    
    # 标记已配置
    if font_set:
        _font_configured = True
    
    return font_set
