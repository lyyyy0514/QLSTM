# QLSTM
# 对比实验 — QWLSTM vs LSTM vs QRF

本项目旨在对比三种模型（QWLSTM、LSTM、QRF）在不同市场指数上的预测效果。

## 实验脚本

- `qlstm_train.py`：使用加权 LSTM（QWLSTM）模型进行训练与预测。
- `lstm_train.py`：使用标准 LSTM 模型进行训练与预测。
- `qrf_train.py`：使用分位数随机森林（QRF）模型进行训练与预测。

每个脚本均可直接运行，自动读取相应数据并完成训练、验证及评估。

## 数据处理模块

- `data_sp500.py`：原始 `S&P 500` 指数数据的下载与初步清洗。
- `data_shangzheng.py`：原始 `上证指数` 数据的下载与初步清洗。
- `dataloader_sp500.py`：将 `S&P 500` 清洗后数据转换为模型输入的 `DataLoader`。
- `dataloader_shangzheng.py`：将 `上证指数` 清洗后数据转换为模型输入的 `DataLoader`。

> 注意：数据文件名请与脚本内部调用保持一致。若使用其他文件名或路径，请自行修改脚本中的导入部分。

## 环境依赖

建议环境：

```bash
python==3.13.2
numpy==2.2.2
pandas==2.2.3
scikit-learn==1.61.1
torch>=1.13.0
bayes-opt>=1.2.0
```  
即使与上述版本略有出入，代码仍应能在常见 Python 环境中运行。

## 使用方法

1. 克隆或下载本项目：
2. 安装依赖：
   ```bash
pip install -r requirements.txt
```
3. 运行训练脚本：
   - 对上证指数：
     ```bash
python qlstm_train.py --market shangzheng
python lstm_train.py --market shangzheng
python qrf_train.py --market shangzheng
```
   - 对 S&P 500：
     ```bash
python qlstm_train.py --market sp500
python lstm_train.py --market sp500
python qrf_train.py --market sp500
```

## 结果评估

各脚本会输出模型在验证集或测试集上的关键指标（如 MSE、Kupiec 检验结果等），并将预测结果和日志保存在项目目录中。

## 日志记录

- 每次运行脚本将自动记录到 `training.log`，包含：
  - 运行时间戳
  - 当前使用的超参数
  - 验证/测试集损失
  - Kupiec 检验结果
- 可通过日志文件查看历史实验记录，方便对比与追踪。

## 最优参数

- 在 `bayesian_optimization` 完成后，最优超参数将保存在 `best_params.txt` 中，格式为 txt：
  ```txt例如
  {
    "dropout": 0.123,
    "hidden_size": 64,
    "num_layers": 2,
    "n_iter": 1000,
    "lr": 0.001,
    "batch_size": 32,
    "tol": 1e-05,
    "tau": 0.5,
    "n_estimators": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_depth": 10
  }
  ```
- 可直接加载该文件用于复现最佳实验结果：
  ```bash
python example_load_best.py
```

---

欢迎根据需要调整参数、批处理方式或数据来源，以完成更深入的对比分析。

