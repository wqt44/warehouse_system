# 仓储系统多智能体强化学习仿真平台

基于 MAPPO 的多智能体仓储系统，支持机器人协同任务分配与路径规划。

## 核心特性

- **可配置仓库**：货架区、工作站、障碍物、充电站，默认配置 `config.json`
- **MAPPO 算法**：多智能体近端策略优化
- **课程学习**：从 2 机器人逐步过渡到 20 机器人
- **A* 寻路导航**：推理/评估阶段用 A* 寻路，准确进入目标货物所在通道
- **电量与充电**：低电量自动充电，充完电离开充电站
- **取货流程**：任务取货点在货架上，机器人进入货架取货后自动退回通道
- **礼让策略**：有货的机器人不动，没货的机器人让开；预测碰撞、提前礼让（同目标/对穿）

## 快速开始

### 安装

```bash
cd warehouse_system
pip install -r requirements.txt
```

依赖：Python 3.8+、PyTorch、gymnasium、numpy、pygame、matplotlib、scipy、tqdm

### 1. 配置仓库

使用图形界面编辑 `config.json`：

```bash
python interactive_config.py
```

可拖拽绘制货架、点击放置工作站/障碍物/充电站，保存后自动写入 `config.json`。

### 2. 运行示例

```bash
python example.py
```

### 3. 训练

```bash
# 基础训练（有 GPU 时会询问是否使用）
python train.py --device cpu

# 推荐：课程学习
python train.py --use_curriculum --device cpu

# 完整训练（含可视化）
python train.py --use_curriculum --use_learning_allocator --render --device cpu
```

### 4. 推理

```bash
python inference.py
```

自动加载 `config.json`，运行后会生成 `robot_paths.png` 路径图。

### 5. 评估

```bash
python evaluate.py --model_path ./checkpoints/final_model.pth --num_episodes 10
```

## 主要脚本

| 脚本 | 说明 |
|------|------|
| `interactive_config.py` | 图形化编辑仓库配置，保存到 config.json |
| `example.py` | 贪心分配 + 随机动作演示 |
| `train.py` | 训练 MAPPO 模型 |
| `inference.py` | 加载模型进行推理与可视化 |
| `evaluate.py` | 性能评估与算法对比 |
| `plot_paths.py` | 从路径 JSON 生成可视化图 |

## 配置说明

- 默认配置文件：`config.json`
- 训练/推理/评估均默认使用 `config.json`，可用 `--config 其他.json` 覆盖

### 配置结构

```json
{
  "warehouse": {
    "width": 50,
    "height": 50,
    "shelf_regions": [[x1, y1, x2, y2], ...],
    "workstation_positions": [[x, y], ...],
    "obstacle_positions": [[x, y], ...],
    "charging_stations": [[x, y], ...] 或 null,
    "task_spawn_rate": 0.1,
    "max_tasks": 20
  },
  "robot": {
    "num_robots": 5,
    "max_battery": 100.0,
    ...
  },
  "max_steps": 300
}
```

## 项目结构

```
warehouse_system/
├── agents/           # 机器人、任务定义
├── algorithms/       # MAPPO、课程学习
├── baselines/        # 贪心、匈牙利、遗传算法
├── env/              # 仓库环境、观察、奖励
├── models/           # 策略网络、任务分配网络
├── utils/            # 可视化、配置加载、A* 寻路等
├── config.json       # 默认仓库配置
├── train.py
├── inference.py
├── evaluate.py
├── example.py
├── interactive_config.py
└── requirements.txt
```

## 设计要点

- **移动约束**：机器人只能在空地、工作站、充电站移动，不能穿越障碍物；仅当「前往取货」时可进入目标货架格，取完货自动退回通道
- **任务**：取货点随机在货架上，送货点在工作站
- **动作**：上/右/下/左/等待 共 5 个离散动作
- **观察**：自身状态 + 11×11 局部网格 + 全局任务队列
- **礼让**：有货（有任务）的机器人不主动让路；没货的机器人遇冲突时让路；执行移动前预测同目标/对穿并提前让一方本步等待

## 常见问题

**训练过慢？** 减少 episode 数、用 `--device cuda`、不加 `--render`

**任务完成率低？** 用课程学习，检查 `config.json` 是否合理

**自定义布局？** 运行 `python interactive_config.py` 编辑并保存

## 许可证

MIT License
