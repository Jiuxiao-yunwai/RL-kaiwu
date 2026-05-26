# 项目结构说明

本仓库包含两个开悟强化学习实验工程。开悟平台依赖固定的 agent、conf、workflow 目录约定，因此结构整理以“明确边界、补齐文档、忽略运行产物”为主，不重排核心代码。

## 根目录

| 路径 | 说明 |
| --- | --- |
| `README.md` | 仓库入口、快速开始、实验导航。 |
| `requirements.txt` | Python 依赖清单。部分注释依赖需要在开悟平台或专用镜像中获得。 |
| `docs/` | 仓库级文档和框架说明。 |
| `scripts/` | 仓库维护脚本，目前包含 Markdown 实验报告转 Word 的辅助脚本。 |
| `exp1-gorge-work/` | 实验1：峡谷漫步。 |
| `exp2-back-to-the-realm/` | 实验2：重返秘境。 |

## 实验1：峡谷漫步

`exp1-gorge-work` 是较轻量的单实验代码包，核心内容是多种传统强化学习 agent 的实现：

| 路径 | 说明 |
| --- | --- |
| `agent_dynamic_programming/` | 动态规划 agent。 |
| `agent_monte_carlo/` | 蒙特卡洛 agent。 |
| `agent_q_learning/` | Q-learning agent。 |
| `agent_sarsa/` | SARSA agent。 |
| `agent_diy/` | 自定义 agent 模板或实验版本。 |
| `conf/` | 实验级配置。 |
| `docs/` | 实验报告、渲染稿和补充说明。 |
| `train_test.py` | 单步训练测试入口，运行前需选择 `algorithm_name`。 |

## 实验2：重返秘境

`exp2-back-to-the-realm` 包含 agent 实现、KaiWuDRL 框架代码、环境适配、协议文件和工具脚本：

| 路径 | 说明 |
| --- | --- |
| `agent_dqn/`, `agent_dqn_01/` | DQN agent 版本。 |
| `agent_target_dqn/`, `agent_target_dqn_01/`, `agent_target_dqn_02/` | Target DQN agent 版本。 |
| `agent_diy/`, `agent_diy_01/` | 自定义 agent 模板或实验版本。 |
| `conf/` | 开悟应用、算法和 KaiWuDRL 进程配置。 |
| `kaiwu_agent/` | 开悟 agent 开发套件与场景评估入口。 |
| `kaiwu_env/` | 环境配置、地图和环境通信封装。 |
| `kaiwudrl/` | 分布式训练框架代码。 |
| `arena_proto/` | 场景协议生成代码。 |
| `thirdparty/` | 第三方组件，如 model pool。 |
| `tools/` | 训练、进程、模型、配置切换和诊断脚本。 |
| `origin/` | 原始模板或基线代码备份。 |
| `docs/` | 实验报告和开发指南。 |
| `train_test.py` | 单步训练测试入口，运行前需选择 `algorithm_name`。 |

## Agent 标准结构

同一实验下的 agent 目录尽量保持一致：

| 路径 | 说明 |
| --- | --- |
| `agent.py` | agent 生命周期入口，包括预测、训练、模型保存和加载。 |
| `algorithm/algorithm.py` | 算法更新逻辑。 |
| `conf/conf.py` | agent 内部超参数。 |
| `conf/train_env_conf.toml` | 环境配置。 |
| `feature/definition.py` | `ObsData`、`ActData`、`SampleData` 等数据结构和样本转换。 |
| `feature/preprocessor.py` | 观测预处理、特征工程和辅助函数。 |
| `model/model.py` | 模型结构。 |
| `workflow/train_workflow.py` | 采样、样本处理、学习调用和监控上报流程。 |

## 维护约定

- 不移动 `agent_*`、`conf`、`kaiwu_agent`、`kaiwu_env`、`kaiwudrl` 等平台依赖目录。
- 新增实验报告时优先放入对应实验的 `docs/`，仓库级说明放入根 `docs/`。
- 新增 agent 时复用同实验现有 agent 结构，避免创建不兼容的目录层级。
- 训练日志、checkpoint、模型导出、TensorBoard 输出和文档渲染中间产物默认不提交。
