# RL-kaiwu

腾讯开悟强化学习实验仓库，包含两个独立实验工程：

- `exp1-gorge-work`: 峡谷漫步实验，覆盖动态规划、蒙特卡洛、Q-learning、SARSA 和自定义 agent。
- `exp2-back-to-the-realm`: 重返秘境实验，覆盖 DQN、Target DQN 和自定义 agent，并包含完整 KaiWuDRL 运行框架、环境适配与工具脚本。

仓库保留腾讯开悟平台要求的目录命名和代码包形态。结构优化集中在文档入口、目录说明和运行产物忽略规则上，避免随意移动 agent、conf、workflow 等平台敏感目录。

## 快速开始

### 1. 准备环境

建议使用 Python 虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

部分依赖或运行脚本依赖腾讯开悟平台镜像、Linux shell、`/data`、`/root/tools` 等运行时路径；在普通 Windows 本地环境中，适合做代码阅读、文档维护和静态检查，完整训练建议放到开悟平台或匹配的容器环境中执行。

### 2. 选择实验

峡谷漫步：

```powershell
cd exp1-gorge-work
python train_test.py
```

重返秘境：

```powershell
cd exp2-back-to-the-realm
python train_test.py
```

运行前按实验说明调整 `train_test.py` 中的 `algorithm_name`，并确认对应 agent 目录存在。

## 目录导航

```text
.
├── exp1-gorge-work/           # 实验1：峡谷漫步
├── exp2-back-to-the-realm/    # 实验2：重返秘境
├── docs/                      # 仓库级文档与框架说明
├── scripts/                   # 文档转换等辅助脚本
├── requirements.txt           # Python 依赖清单
└── README.md                  # 仓库入口
```

更完整的目录说明见 [docs/project-structure.md](docs/project-structure.md)。

## Agent 开发约定

每个 agent 目录通常遵循同一结构：

```text
agent_xxx/
├── agent.py                  # 智能体预测、训练、模型保存/加载入口
├── algorithm/algorithm.py    # 算法、loss、优化器逻辑
├── conf/                     # agent 训练参数
├── feature/                  # 数据结构、特征处理、奖励 shaping
├── model/model.py            # 网络或值函数模型
└── workflow/train_workflow.py
```

修改 agent 时优先保持这个结构，便于 `train_test.py`、平台打包流程和评估工作流按约定加载。

## 文档

- [docs/README.md](docs/README.md): 文档索引。
- [docs/intro.md](docs/intro.md): 腾讯开悟强化学习框架综述。
- [exp1-gorge-work/README.md](exp1-gorge-work/README.md): 实验1目录说明。
- [exp2-back-to-the-realm/README.md](exp2-back-to-the-realm/README.md): 实验2目录说明。

## 维护建议

- 训练产物、日志、checkpoint、模型导出文件不要提交到仓库。
- 新 agent 尽量复制同实验下现有 agent 骨架，再替换算法和特征逻辑。
- 实验报告和关键运行说明放在各实验的 `docs/` 下，仓库级说明放在根 `docs/` 下。
