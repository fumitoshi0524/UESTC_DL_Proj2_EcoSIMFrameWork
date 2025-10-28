## 一、研究问题（Research Questions）

1. 政府通过价格保护（价格下限）介入市场，对农民、消费者与政府自身福利的影响量化为何？
2. 干预强度（intervention strength）如何影响市场价格、供需平衡与系统收敛性？
3. 在给定市场机制下，干预是否提高总社会福利（social welfare）或仅改变福利分配？

## 二、总体设计要点（Design Overview）

- 模型类型：离散时间、多智能体、局部信息（agents observe其自身状态与可观测的市场信号）
- 智能体：农民（supply agents）、消费者（demand agents）、政府（policy agent）
- 干预机制：政府设定价格下限 $p_{floor}$，市场价格为 $p_t=\max(p^{eq}_t, p_{floor})$，其中 $p^{eq}_t$ 为市场均衡价
- 实验采样：5 种干预强度（0%, 5%, 10%, 15%, 20%），每种配置独立重复 N=50 次回合，回合长度 T=100 步

## 三、数学模型（Mathematical Model）

设 $i\in\{1,...,N_s\}$ 表示农民个体，$j\in\{1,...,N_d\}$ 表示消费者个体。简化情况下，本实验采用 $N_s=3,\ N_d=2$。

1) 农民决策与成本

农民 $i$ 在时刻 $t$ 选择产量 $q_{i,t}\ge0$。生产成本采用二次成本函数：
$$
c_i(q_{i,t}) = c_0 q_{i,t} + \frac{1}{2} c_1 q_{i,t}^2,
$$
其中 $c_0,c_1>0$。

农民当期利润（报酬）定义为：
$$
\pi_{i,t} = p_t q_{i,t} - c_i(q_{i,t}),
$$
其中 $p_t$ 为市场价格（见下式）。每个农民的长期目标是最大化折扣和 $\mathbb{E}[\sum_{t=0}^{T-1} \gamma^t \pi_{i,t}]$，此处 $\gamma$ 为折扣因子（实验中可设 $\gamma=0.99$ 或 1.0 以便评估平均收益）。

2) 消费者决策与效用

消费者 $j$ 选择购买量 $d_{j,t}\ge0$，其效用函数采用对数效用加价格支出：
$$
u_j(d_{j,t}) = a_j \ln(1 + d_{j,t}) - p_t d_{j,t},
$$
其中 $a_j>0$ 表示需求弹性或偏好强度。消费者目标为最大化折扣效用和 $\mathbb{E}[\sum_t \gamma^t u_j(d_{j,t})]$。

3) 市场清算与价格形成

令总供给 $S_t=\sum_i q_{i,t}$，总需求 $D_t=\sum_j d_{j,t}$。先计算无干预下的均衡价格 $p^{eq}_t$ 作为供需差异的线性函数：
$$
p^{eq}_t = p_0 + \alpha (D_t - S_t),
$$
其中 $p_0$ 为基准价格（可设定为 0.5），$\alpha>0$ 为价格弹性系数（例如 $\alpha=0.5$）。

政府介入后的实际市场价格为：
$$
p_t = \max\left(p^{eq}_t,\ p_{floor}\right),
$$
价格下限由干预强度 $\gamma$ 与基准价格决定：
$$
p_{floor} = p_0(1 + \gamma),\quad \gamma\in\{0,0.05,0.1,0.15,0.2\}.
$$

4) 政府目标函数

政府的即时收益（或目标函数）设计为价格稳定性与供需平衡的加权组合，并扣除干预成本 $C_{int}$：
$$
G_t = \lambda_1\cdot (1 - |p_t - p_0|/p_0) + \lambda_2\cdot\left(1 - \frac{|S_t - D_t|}{\max(S_t,D_t,\epsilon)}\right) - C_{int}(\gamma),
$$
其中 $\lambda_1,\lambda_2\ge0$ 为权重，$\epsilon$ 为避免除以零的正则项。干预成本可建模为二次项：
$$
C_{int}(\gamma) = k_0 \gamma + \frac{1}{2} k_1 \gamma^2.
$$

政府目标为最大化 $\mathbb{E}[\sum_t \gamma^t G_t]$。

5) 行为策略（policy）

在本实验中，为了清晰比较政策影响，我们采用启发式（heuristic）或随机化策略作为 baseline：

- 农民：产量 $q_{i,t}$ 基于前期价格与自身库存的线性规则 + 小幅随机扰动
- 消费者：购买量 $d_{j,t}$ 基于偏好系数 $a_j$ 与当前价格的下滞后函数
- 政府：固定报告干预强度 $\gamma$，不做动态调整（实验中仅比较不同静态 $\gamma$）

（后续可替换为 DQN 等学习策略；本设计先保证可比性与可解释性）

## 四、实验配置（Detailed Protocol）

1. 参数预设值（示例）

 - $p_0=0.5$（基准价格）
 - $c_0=0.2,\ c_1=0.3$（农民成本系数）
 - $a_j\sim\mathcal{U}(0.8,1.2)$（消费者偏好随机分配）
 - $\alpha=0.5,\ \gamma_{discount}=0.99$（价格弹性与折扣因子）
 - $k_0=0.1,\ k_1=0.2$（政府干预成本参数）

2. 场景（policy configs）

 - 五种 $\gamma$：0, 0.05, 0.10, 0.15, 0.20
 - 对每一种 $\gamma$：重复 $N=50$ 次独立回合（不同随机种子）
 - 每回合 $T=100$ 时步，记录每步的 $q_{i,t}, d_{j,t}, p_t, \pi_{i,t}, u_{j,t}, G_t$

3. 初始状态与随机性

 - 每次回合初始化农民与消费者的私有状态（例如库存、初始倾向）为随机值
 - 固定随机种子以便重复实验（但每次回合应使用不同种子以估计变异）

## 五、衡量指标的精确定义（Metrics with formulas）

1) 农民平均收益（per-config）
$$
\overline{\Pi}_{farmer} = \frac{1}{N \cdot N_s} \sum_{r=1}^N \sum_{i=1}^{N_s} \sum_{t=1}^T \pi^{(r)}_{i,t},
$$
其中 $r$ 为回合索引，$N$ 为回合数。

2) 农民收益标准差（波动性）
$$
\sigma_{farmer} = \sqrt{\frac{1}{N\cdot N_s\cdot T -1} \sum_{r,i,t} (\pi^{(r)}_{i,t} - \overline{\Pi}_{farmer})^2 }.
$$

3) 消费者平均效用
$$
\overline{U}_{consumer} = \frac{1}{N \cdot N_d} \sum_{r=1}^N \sum_{j=1}^{N_d} \sum_{t=1}^T u^{(r)}_{j,t}.
$$

4) 政府平均效能
$$
\overline{G} = \frac{1}{N} \sum_{r=1}^N \sum_{t=1}^T G^{(r)}_{t} / T.
$$

5) 市场价格均值与波动性
$$
\overline{p}=\frac{1}{N\cdot T} \sum_{r,t} p^{(r)}_t,\quad
\sigma_p = \sqrt{\frac{1}{N\cdot T -1}\sum_{r,t} (p^{(r)}_t - \overline{p})^2 }.
$$

6) 收敛性指标（最后 $L$ 步）
$$
Conv_{farmer} = \frac{1}{N\cdot N_s \cdot L} \sum_{r=1}^N \sum_{i=1}^{N_s} \sum_{t=T-L+1}^T \pi^{(r)}_{i,t}.
$$

建议采用 $L=25$（即最后 25 个时间步）或以回合分段（例如最后 20 个回合）计算收敛平均。

## 六、统计分析计划（Statistical Analysis Plan）

1) 描述性统计

 - 对每个配置计算均值、标准差、95% 置信区间
 - 绘制平均曲线与误差条（error-bar），以及箱线图（boxplots）与密度估计

2) 因子检验（Intervention effect）

 - 以干预强度 $\gamma$ 作为因子，对农民平均收益、消费者平均效用、政府效能进行单因素 ANOVA
 - 若 ANOVA 显著，进行事后检验（Tukey HSD）比较两两差异

3) 趋势检验

 - 使用线性回归检验干预强度对指标的趋势：
$$
Y = \beta_0 + \beta_1 \gamma + \varepsilon,
$$
其中 $Y$ 为某一聚合指标（例如 $\overline{\Pi}_{farmer}$）。检验 $\beta_1$ 是否显著。

4) 收敛性与时间序列分析

 - 对价格序列进行自相关检验（ACF/PACF）与波动率分析
 - 检验系统是否在 $T$ 步内达到稳定：使用单位根检验或方差趋于稳定的判据

5) 功效分析（Power analysis）

 - 在设计阶段假定最小可检测效应（MDE）为农民平均收益的 0.05（绝对值），在 $\alpha=0.05$、功效 $1-\beta=0.8$ 下，估计所需的回合数。若需要，可调整 $N$。

## 七、数据格式与可复现性（Data & Reproducibility）

1) 原始数据格式（每回合）

 JSON 结构（每个回合）：

```json
{
  "seed": 12345,
  "config": {"gamma": 0.05, "p0": 0.5, ...},
  "steps": [
     {"t":0, "p":0.51, "S":0.8, "D":0.7, "q": [0.2,0.3,0.3], "d": [0.35,0.35], "pi": [..], "u": [..], "G": 0.6},
     ...
  ],
  "summary": {"avg_pi_farmer": 0.23, "avg_u_consumer": 0.12, "avg_G": 0.58}
}
```

2) 聚合输出（各配置）

 - `experiment_results.json` 包含每个配置（gamma）下 N 个回合的聚合统计：均值、标准差、收敛值、价格指标

3) 版本控制与环境

 - 所有代码与文档放在同一 Git 仓库，commit 记录参数变更
 - 使用 `requirements.txt` 锁定 Python 包版本
 - 在实验报告中记录 Python 可执行文件路径与环境信息

## 八、可视化与报告（Visualization & Reporting）

建议生成下列图表：

1. 智能体平均收益随干预强度变化（误差条）
2. 不同配置的价格时间序列平均线
3. 收敛性比较（最后 $L$ 步的箱线图）
4. 政策成本-效益散点图（政府效能 vs 干预成本）
5. 各方福利分配堆叠图（stacked bar）

所有图保存为 PNG，并保存原始数据以便重绘与审查。

## 九、潜在风险与限制（Limitations & Risks）

1. 模型对现实简化过度，特别是行为规则与外生冲击未完全考虑
2. 参数敏感性高，需对关键参数（成本、偏好、价格弹性）做进一步敏感性分析
3. 当前设计以静态政策为主，未涵盖动态政策调整情形

## 十、实验流程（Step-by-step Protocol）

1. 在 `scenario.py` 中设置参数与 reward 函数，确保与文档一致
2. 使用 `run_experiment.py` 逐一运行五个配置（不同 gamma），每个配置 N=50
3. 将每回合 JSON 输出写入 `results/`，并在跑完后生成 `experiment_results.json` 聚合文件
4. 用 `framework/utils/visualization.py` 生成所有图表
5. 用 ANOVA 与回归检验对结果进行统计检验，保存 p 值与效应量

## 十一、附录 — 参考公式与参数建议（Appendix）

- 成本参数：$c_0=0.2, c_1=0.3$（可调）
- 价格偏好：$a_j\in[0.8,1.2]$ 随机扰动
- 置信度：95%（alpha=0.05）

---