# Multi-Agent Game Theory Research Framework

UESTC深度学习课程结课项目
多智能体博弈论研究框架。支持任意数量和类型的智能体、自定义奖励函数和市场机制。

## 快速开始

### 安装
```bash
pip install -r requirements.txt
```

### 运行政府干预实验
```bash
cd f:\projects\production_game
python examples/government_intervention/run_experiment.py
```

这会生成：
- `examples/government_intervention/results/experiment_results.json` - 实验数据
- `examples/government_intervention/results/*.png` - 6张可视化图表

### 生成可视化
```bash
python framework/utils/visualization.py
```

## 项目结构

```
framework/                           # 可复用的多智能体框架
├── core/
│   ├── environment.py              # 多智能体游戏环境
│   └── learning.py                 # DQN学习
└── utils/
    └── visualization.py            # 数据可视化

examples/government_intervention/   # 政府干预农业政策示例
├── scenario.py                    # 环境和奖励函数定义
├── run_experiment.py              # 实验脚本
├── EXPERIMENT_DESIGN.md           # 实验设计文档
├── ANALYSIS.md                    # 结果分析
└── results/                       # 输出目录
```

## 框架使用

### 1. 定义智能体和奖励
```python
from framework.core import AgentSpec, GameEnvironment

farmer_spec = AgentSpec(
    name='farmer',
    num_agents=3,
    obs_space=(4,),
    action_space=(1, 1),
    reward_fn=farmer_reward_fn,
)
```

### 2. 定义市场机制
```python
from framework.core import MarketSpec

market_spec = MarketSpec(
    clearing_fn=my_market_clearing_function
)
```

### 3. 创建环境和运行
```python
env = GameEnvironment(
    agents_specs={'farmer': farmer_spec, ...},
    market_spec=market_spec,
    time_steps=100,
)

obs = env.reset()
for step in range(100):
    actions = policy(obs)
    obs, rewards, done, info = env.step(actions)
```

## 政府干预实验

### 研究问题
政府价格保护政策对农民、消费者和市场均衡的影响

### 主要发现
- **农民收益**: 干预强度20%时下降20%
- **消费者收益**: 基本不变 (±0.6%)
- **整体福利**: 下降5.2%
- **市场价格**: 自然稳定，干预无额外效果

### 政策含义
价格保护政策成本高于收益，应考虑直接补贴

## 依赖

- PyTorch 2.0+
- NumPy 1.20+
- Matplotlib 3.5+
- Python 3.8+
│   │   ├── __init__.py
│   │   └── dqn_agent.py                 # DQN智能体实现
│   └── utils/
│       └── __init__.py
├── scripts/
│   ├── train.py                         # 训练脚本
│   ├── evaluate.py                      # 评估脚本
│   └── visualize.py                     # 可视化脚本
├── requirements.txt                     # 依赖包列表
├── setup.py                             # 项目配置
└── README.md                            # 本文件
```

## 安装与设置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 安装项目

```bash
pip install -e .
```

## 快速开始

### 训练模型

```bash
# 基础训练（默认参数）
python scripts/train.py

# 自定义参数
python scripts/train.py \
    --num-farmers 5 \
    --num-episodes 1000 \
    --max-steps 500 \
    --save-interval 100 \
    --save-dir ./models \
    --log-dir ./logs
```

训练会在`models/`目录下定期保存模型，在`logs/`目录下保存训练结果。

### 评估模型

```bash
# 评估指定的模型
python scripts/evaluate.py \
    --model-dir ./models/episode_1000 \
    --num-farmers 5 \
    --max-steps 500 \
    --num-episodes 10
```

### 可视化结果

```bash
# 生成各类图表
python scripts/visualize.py \
    --results-file ./logs/training_results.json \
    --trajectory-file ./logs/trajectory.json \
    --output-dir ./visualizations
```

## 核心组件说明

### 1. ProductionGameEnv

多智能体环境的实现，支持:
- 灵活的配置参数
- 动态的市场价格机制
- 完整的奖励计算
- Gym兼容的接口

```python
from production_game.envs.production_game_env import ProductionGameEnv

env = ProductionGameEnv(
    num_farmers=5,
    max_production=100.0,
    max_demand=100.0,
    max_price=10.0,
    time_steps=500,
)

obs = env.reset()
action = {
    'farmer': np.array([50, 45, 55, 40, 60]),  # 5个农民的产量决策
    'consumer': np.array([0.8]),                 # 消费者购买倾向
    'government': np.array([0.1, 0.05]),        # 税收率、补贴率
}
next_obs, rewards, done, info = env.step(action)
```

### 2. DQN智能体

基于深度Q学习的智能决策系统:
- **QNetwork**: 深度神经网络Q函数估计
- **DQNAgent**: 单个智能体的训练和推理
- **MultiAgentDQN**: 多智能体系统管理

主要特性:
- ε-贪心探索策略
- 经验回放机制
- 目标网络
- 梯度裁剪

```python
from production_game.agents.dqn_agent import DQNAgent

agent = DQNAgent(
    state_dim=10,
    action_dim=1,
    learning_rate=1e-3,
    gamma=0.99,
    epsilon=1.0,
)

# 选择动作
action = agent.act(state, training=True)

# 存储经验
agent.remember(state, action, reward, next_state, done)

# 经验回放学习
loss = agent.replay()
```

## 训练配置

### 推荐参数

对于快速测试:
```bash
python scripts/train.py --num-episodes 100 --max-steps 200
```

对于生产级训练:
```bash
python scripts/train.py \
    --num-farmers 10 \
    --num-episodes 5000 \
    --max-steps 1000 \
    --save-interval 500
```

### 超参数调整

在`dqn_agent.py`中修改DQNAgent初始化参数:
- `learning_rate`: 学习率（默认1e-3）
- `gamma`: 折扣因子（默认0.99）
- `epsilon`: 初始探索率（默认1.0）
- `epsilon_decay`: 探索率衰减（默认0.9995）

## 评估指标

训练和评估会生成以下指标:

1. **奖励**: 每个agent的累积奖励
2. **市场价格**: 供需平衡下的动态价格
3. **交易量**: 实际成交量
4. **满意度**: 消费者和政府的满意度指标
5. **政府预算**: 税收和补贴的财政影响

## 输出文件

### 训练输出 (`logs/`)
- `training_results.json`: 训练过程中的奖励曲线和配置信息

### 模型文件 (`models/`)
- `episode_*/farmer_*.pt`: 农民模型
- `episode_*/consumer.pt`: 消费者模型
- `episode_*/government.pt`: 政府模型

### 可视化输出 (`visualizations/`)
- `training_rewards.png`: 三个agent的奖励曲线
- `evaluation_trajectory.png`: 市场动态和满意度指标
- `comprehensive_analysis.png`: 综合分析图表

## 扩展与自定义

### 修改奖励函数

编辑`production_game_env.py`中的`_calculate_rewards`方法:

```python
def _calculate_rewards(self, ...):
    # 自定义农民奖励
    farmer_rewards = your_custom_reward_function()
    
    # 自定义消费者奖励
    consumer_reward = ...
    
    # 自定义政府奖励
    government_reward = ...
```

### 调整市场机制

修改`_calculate_market_price`和交易逻辑以实现不同的市场模型。

### 添加新的约束条件

在`step`方法中添加额外的物理约束或经济约束。

## 性能优化

1. **GPU加速**: 修改设备参数使用CUDA
2. **批处理**: 调整batch_size参数
3. **并行训练**: 使用多进程加速环境交互

## 参考资源

- [Deep Q-Networks (DQN)](https://arxiv.org/abs/1312.5602)
- [Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1908.03963)
- [OpenAI Gym Documentation](https://gym.openai.com/)
