# 实验1：峡谷漫步

本目录是峡谷漫步实验代码包，包含多种传统强化学习算法实现和实验报告。

## 目录

| 路径 | 说明 |
| --- | --- |
| `agent_dynamic_programming/` | 动态规划 agent。 |
| `agent_monte_carlo/` | 蒙特卡洛 agent。 |
| `agent_q_learning/` | Q-learning agent。 |
| `agent_sarsa/` | SARSA agent。 |
| `agent_diy/` | 自定义 agent。 |
| `conf/` | 实验配置。 |
| `docs/` | 实验报告、Word 版本和渲染输出。 |
| `train_test.py` | 训练测试入口。 |
| `kaiwu.json` | 开悟平台配置。 |

## 运行

在开悟运行环境或兼容容器中进入本目录：

```bash
python train_test.py
```

运行前修改 `train_test.py` 中的 `algorithm_name`，可选值包括：

- `dynamic_programming`
- `monte_carlo`
- `q_learning`
- `sarsa`
- `diy`

## 开发入口

每个 agent 的主要修改点：

- `agent.py`: 预测、训练、动作处理和模型接口。
- `feature/`: 数据结构、观测处理、奖励设计和样本转换。
- `algorithm/`: 算法更新逻辑。
- `workflow/train_workflow.py`: 环境交互与训练流程。
