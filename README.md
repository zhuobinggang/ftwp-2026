## 20261020: vanilla cogniTextwolrdAgent训练的三种方式

* 模型代码以`final_vanilla_agent.py`和`final_vanilla_agent_mlm.py`为主
* scripts以`final_full_train_script.py`和`final_full_train_script_small_lr.py`和`final_full_train_script_mlm.py`为主
* 注意： 训练sequence classify模型，高于lr = 2e-5，batch size = 16无法收敛
* prompt B: 将description替换成available actions

| 指标 / 超参数 | 实验组 1 | 实验组 2 | 实验组 3 | 实验组2 + naive ucb1 | 实验组2 + prompt B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **lr** | 2e-5 | 1e-5 | 2e-5 | 1e-5 | - |
| **batch size** | 16 | 16 | 16 | 16 | - |
| **bert** | sequence classify | sequence classify | mask language modeling | sequence classify | - |
| **loss e1** | 0.2703 | 0.2093 | 0.1922 | - | 0.2254 |
| **loss e2** | 0.1482 | 0.1025 | 0.0985 | -| 0.1339 |
| **loss e3** | 0.1065 | 0.0779 | 0.0804 | - | 0.1084 |
| **prec** | 0.7989 | 0.8869 | 0.8939 | - | - |
| **rec** | 0.8290 | 0.8316 | 0.8204 | - | - |
| **f1** | 0.8137 | 0.8584 | 0.8556 | - | - |
| **avg step** | 21.47 | 22.18 | 23.20 | 18.63 | 27.23 |
| **norm score** | 0.7208 | 0.8654 | 0.8387 | 0.7615 | 0.7729 |

结论

1. 学习率是对分类头模型最具破坏性的超参数
2. MLM / Cloze 范式表现出强大的先验鲁棒性与收敛速度
3. 离线指标（F1）与在线强化评估（Score / Step）的轻微解耦