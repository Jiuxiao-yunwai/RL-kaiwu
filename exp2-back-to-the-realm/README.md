# 实验2：重返秘境

本目录是重返秘境实验代码包，包含 DQN、Target DQN、自定义 agent、KaiWuDRL 分布式训练框架、环境适配和工具脚本。

## 目录

| 路径 | 说明 |
| --- | --- |
| `agent_dqn/`, `agent_dqn_01/` | DQN agent 版本。 |
| `agent_target_dqn/`, `agent_target_dqn_01/`, `agent_target_dqn_02/` | Target DQN agent 版本。 |
| `agent_diy/`, `agent_diy_01/` | 自定义 agent。 |
| `conf/` | 应用、算法和 KaiWuDRL 进程配置。 |
| `kaiwu_agent/` | 开悟 agent 开发套件与评估入口。 |
| `kaiwu_env/` | 环境通信封装、地图和场景配置。 |
| `kaiwudrl/` | 分布式训练框架。 |
| `arena_proto/` | 场景协议生成代码。 |
| `thirdparty/` | 第三方组件。 |
| `tools/` | 训练、诊断、配置切换和模型工具。 |
| `origin/` | 原始模板或基线备份。 |
| `docs/` | 实验报告和开发指南。 |
| `train_test.py` | 训练测试入口。 |

## 运行

在开悟运行环境或兼容容器中进入本目录：

```bash
python train_test.py
```

运行前修改 `train_test.py` 中的 `algorithm_name`，可选值包括：

- `dqn`
- `target_dqn`
- `diy`

`train_test.py` 会调用 `tools/` 下的 shell 脚本，并依赖 `/data`、`/root/tools`、model pool 等平台运行时资源。普通 Windows 本地环境通常无法完整执行训练流程。

## 开发入口

推荐优先关注当前实验版本：

- `agent_target_dqn/`: Target DQN 主版本。
- `agent_dqn/`: DQN 主版本。
- `agent_diy_01/`: 自定义实验版本。
- `kaiwu_agent/back_to_the_realm/`: 评估工作流和场景侧特征处理。
