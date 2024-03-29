# 飞桨训推一体全流程（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleOCR中所有模型的飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

[![img](https://github.com/PaddlePaddle/PaddleOCR/raw/dygraph/test_tipc/docs/guide.png)](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/test_tipc/docs/guide.png)

## 2. 测试工具简介

### 目录介绍

```
test_tipc
    |--configs                              # 配置目录
        |--SwinIR                           # 模型名称
            |--train_infer_python.txt       # 基础训练推理测试配置文件
    |--docs
        |--train_infer_python.md            # TIPC说明文档
    |--data                                 # 推理数据
    |--output                               # TIPC推理结果与日志
    |--test_train_inference_python.sh       # TIPC基础训练推理测试解析脚本
    |--common_func.sh                       # TIPC基础训练推理测试常用函数
    |--prepare.sh                           # 推理数据准备脚本
    |--readme.md                            # 使用文档
```

### 测试流程概述

使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程概括如下：

1. 运行prepare.sh准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_train_inference_python.sh`，产出log，由log可以看到不同配置是否运行成功；

测试单项功能仅需两行命令，**如需测试不同模型/功能，替换配置文件即可**，命令格式如下：

```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/prepare.sh  configs/[model_name]/[params_file_name]  [Mode]

# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh configs/[model_name]/[params_file_name]  [Mode]
```

例如，测试基本训练预测功能的`lite_train_lite_infer`模式，运行：

```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/prepare.sh ./test_tipc/configs/SwinIR/train_infer_python.txt 'lite_train_lite_infer'

# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/SwinIR/train_infer_python.txt 'lite_train_lite_infer'
```
