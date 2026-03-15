"""
pytest 配置：确保从项目根或 warehouse_system 目录运行测试时都能正确导入模块
"""
import sys
import os

# 将 warehouse_system 目录加入 path，便于 import env / utils / config 等
_ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ws not in sys.path:
    sys.path.insert(0, _ws)
