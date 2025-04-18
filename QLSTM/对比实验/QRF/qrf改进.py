import numpy as np
import torch
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from quantile_forest import RandomForestQuantileRegressor
from data1 import X, Y  
# 固定使用 CPU
device = torch.device("cpu")
print("Using device:", device)


def sliding_window(X: np.ndarray, Y: np.ndarray, step: int = 30):
    """
    使用滑动窗口将原始时间序列数据转换为监督学习样本，
    每个样本包含连续的 step 个时刻的数据，目标取第 step+1 个时刻的值。
    """
    if len(X) != len(Y):
        raise IndexError("X 和 Y 的长度不一致！")
    x_windows, y_windows = [], []
    for i in range(len(X) - step):
        x_windows.append(X[i:i + step, :])
        # 这里假设 Y 为二维数组，取第一列作为目标值（例如：价格等）
        y_windows.append(Y[i + step, 0])
    return np.array(x_windows), np.array(y_windows)

def flatten(X: np.ndarray):
    """
    将每个样本的多维数据展平为一维特征向量，
    最终输出形状为 (n_samples, n_features)
    """
    return X.reshape(X.shape[0], -1)

# 使用滑动窗口构造特征
X_window, Y_window = sliding_window(X, Y, step=30)

# 数据标准化：先将数据展平成二维，然后对每个特征进行标准化
X_window_flat = flatten(X_window)
scaler = StandardScaler()
X_window_scaled = scaler.fit_transform(X_window_flat)

# 两次拆分，得到 训练集、测试集、验证集（
X_train_np, X_temp_np, Y_train_np, Y_temp_np = train_test_split(X_window_scaled, Y_window, test_size=0.3, random_state=42)
X_test_np, X_val_np, Y_test_np, Y_val_np = train_test_split(X_temp_np, Y_temp_np, test_size=1/3, random_state=42)

# 若后续需要torch tensor，可以转换，不过 QRF 模型直接接受 numpy 数组
X_train = X_train_np.copy()
Y_train = Y_train_np.copy()
X_val   = X_val_np.copy()
Y_val   = Y_val_np.copy()
X_test  = X_test_np.copy()
Y_test  = Y_test_np.copy()

print("数据预处理完成：训练集、验证集、测试集尺寸分别为",
      X_train.shape, X_val.shape, X_test.shape)


quantile = 0.05  # 全局分位数

def violation(Y_true, Y_predict):
    """
    计算预测值低于真实值的个数
    """

    if isinstance(Y_true, torch.Tensor):
        Y_true = Y_true.cpu().numpy()
    if isinstance(Y_predict, torch.Tensor):
        Y_predict = Y_predict.cpu().numpy()
    return int(np.sum(Y_true < Y_predict))

def target_loss(Y_true: np.ndarray, Y_predict: np.ndarray, quantile: float = quantile):
    """
    目标损失计算：使得预测样本中低于真实值的比例尽可能接近指定的 quantile
    """
    yp_big_num = violation(Y_true, Y_predict)
    proportion = yp_big_num / len(Y_true)
    return abs(proportion - quantile)

def kupiec_test(violations, total_days, quantile, verbose: bool = True):
    """
    Kupiec 检验，用于衡量风险度量模型（预测值低于实际值）的合理性
    """
    confidence_intervals_map = {
        0.05: 3.841,
        0.1: 2.706,
        0.025: 5.024,
    }
    if quantile not in confidence_intervals_map:
        raise ValueError("Invalid quantile value, must be 0.05, 0.1, or 0.025")
    p_value = quantile
    kupiec_statistic = -2 * np.log(((1 - p_value) ** (total_days - violations)) * (p_value ** violations))
    result = kupiec_statistic > confidence_intervals_map[p_value]
    test = f"kupiec 检验：{'不拒绝' if result else '拒绝'}原假设, 违约次数: {violations}"
    print(kupiec_statistic)
    if verbose:
        print(test)
    return result



def calculate_loss(n_estimators, min_samples_split, min_samples_leaf, max_depth):
    """
    贝叶斯优化目标函数：
    构建随机森林分位回归模型，对训练集进行拟合，
    然后在验证集上预测指定分位数，计算目标损失
    """
    # 参数转换
    n_estimators = int(n_estimators)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    max_depth = int(max_depth)
    
    # 构建模型
    rf = RandomForestQuantileRegressor(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )
    # 使用训练集（此处数据均为 numpy 数组）
    rf.fit(X_train, Y_train.ravel())
    
    # 对验证集进行预测，使用 quantiles 参数传入列表
    Y_pred = rf.predict(X_val, quantiles=[quantile])
    if isinstance(Y_pred, np.ndarray) and Y_pred.ndim == 2:
        Y_pred = Y_pred[:, 0]
    
    # 输出 Kupiec 检验结果（仅作参考）
    yp_violations = violation(Y_val, Y_pred)
    kupiec_test(yp_violations, len(Y_pred), quantile=quantile)
    
    loss = -target_loss(Y_val, Y_pred, quantile=quantile)
    print(f"Loss: {loss} (n_estimators: {n_estimators}, min_samples_split: {min_samples_split}, "
          f"min_samples_leaf: {min_samples_leaf}, max_depth: {max_depth})")
    return loss

def bayesian_optimization(_iter: int = 100):
    """
    使用贝叶斯优化调节随机森林的参数，保存最优参数到文件
    """
    pbounds = {
        "n_estimators": (50, 200),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 15),
        "max_depth": (3, 20),
    }
    optimizer = BayesianOptimization(f=calculate_loss, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points=30, n_iter=_iter)  # 参数搜索
    best_params = optimizer.max["params"]
    print("最优参数：", optimizer.max)
    # 使用时间戳命名防止覆盖
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"best_params_{timestamp}.txt"
    with open("best_params.txt", "w") as f:
        json.dump(best_params, f)
    return best_params

def load_best_params():
    with open("best_params.txt", "r") as f:
        best_params = json.load(f)
    return {
        "n_estimators": int(best_params["n_estimators"]),
        "min_samples_split": int(best_params["min_samples_split"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
        "max_depth": int(best_params["max_depth"]),
    }

def train_model_1():
    """
    滚动向前预测训练：
    根据加载最优参数后，在训练集、验证集、测试集上进行滚动预测
    """
    try:
        best_params = load_best_params()
    except FileNotFoundError:
        print("best_params.txt 未找到，请先运行 bayesian_optimization()。")
        return
    
    # 合并全部数据（注意这里用的是已标准化且展平后的 numpy 数组）
    X_total = np.concatenate([X_train, X_val, X_test], axis=0)
    Y_total = np.concatenate([Y_train, Y_val, Y_test], axis=0)
    
    total_size = X_total.shape[0]
    input_size = int(total_size * 0.8)  # 例如 80% 用作训练
    
    Y_pred = np.empty(total_size - input_size)
    
    for i in range(total_size - input_size):
        slice_start = i
        slice_end = i + input_size
        
        _X_train = X_total[slice_start:slice_end]
        _Y_train = Y_total[slice_start:slice_end]
        
        rf = RandomForestQuantileRegressor(
            n_estimators=best_params["n_estimators"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_depth=best_params["max_depth"],
        )
        # 模型训练时注意将目标值转为1d数组
        rf.fit(_X_train, _Y_train.ravel())
        
        # 预测下一个时刻的值：这里用最后一个样本作为预测输入
        # 保证输入数据为二维数组（1, n_features）
        x_input = X_total[slice_end].reshape(1, -1)
        y = rf.predict(x_input, quantiles=[quantile])
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y[:, 0]
        Y_pred[i] = y[0]
        print(f"第 {i+1} 次预测完成!")
    
    print("滚动预测训练完成!")
    loss = target_loss(Y_total[input_size:], Y_pred, quantile=quantile)
    print("最终损失:", loss)
    
    yp_violations = violation(Y_total[input_size:], Y_pred)
    kupiec_result = kupiec_test(yp_violations, len(Y_pred), quantile=quantile)
    print("Kupiec 检验结果:", kupiec_result)
    


if __name__ == "__main__":
    # 首先进行贝叶斯优化，找出最优参数
    bayesian_optimization(_iter=100)
    # 根据最优参数进行滚动预测训练
    train_model_1()
