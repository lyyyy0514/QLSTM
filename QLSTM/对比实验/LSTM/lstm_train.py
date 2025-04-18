from dataloader import X_train, Y_train, X_test, Y_test, X_val, Y_val, device, flatten 
from QWLSTMModel import QWLSTMModel, get_rfweight
from quantile_forest import RandomForestQuantileRegressor
import numpy as np
from bayes_opt import BayesianOptimization
import json
import torch
import time

# 定义全局变量 tau
tau = 1
quantile = 0.05  # 全局分位数

def violation(Y_true, Y_predict):
    if isinstance(Y_true, torch.Tensor):
        Y_true = Y_true.cpu().numpy()

    if isinstance(Y_predict, torch.Tensor):
        Y_predict = Y_predict.cpu().numpy()

    return int(np.sum(Y_true < Y_predict))

# the param should be whether instance of ndarray or tensor
def target_loss(Y_true: np.ndarray, Y_predict: np.ndarray, quantile: float = quantile):
    """计算目标损失，用于贝叶斯优化

    Args:
        Y (np.ndarray): 实际值（标签）
        Y_predict (np.ndarray): 预测值
        quantile (float): 分位数

    Returns:
        float: 损失值
    """
    yp_big_num = violation(Y_true, Y_predict)
    proportion = yp_big_num / len(Y_true)

    return abs(proportion - quantile)


def kupiec_test(violations, total_days, quantile, verbose: bool = True):
    """Kupiec检验

    Args:
        violations (int): 违约次数
        total_days (int): 总天数
        significance_level (float): 显著性水平

    Returns:
        bool: 是否拒绝原假设
    """

    confidence_intervals_map = {
        0.05: 3.841,
        0.1: 2.706,
        0.025: 5.024,
    }

    if quantile not in confidence_intervals_map:
        raise ValueError("Invalid quantile value, must be 0.05, 0.1, or 0.025")

    p_value = quantile
    kupiec_statistic = -2 * np.log(
        ((1 - p_value) ** (total_days - violations)) * (p_value**violations)
    )

    result = kupiec_statistic > confidence_intervals_map[p_value]

    if result < confidence_intervals_map[quantile]:
        text = f"kupiec result: Reject the null hypothesis. Default count: {violations}"
    
    else:
        text = f"not reject the null hypothesis, Default count: {violations}"
    
    if verbose:
        print(text)

    return result < confidence_intervals_map[quantile]  # 95%置信区间的临界值为3.84


def calculate_loss(
    # Qmodel参数
    dropout,
    hidden_size,
    n_iter,
    lr,
    batch_size,
    tol,
    num_layers,
    # RF参数
    n_estimators,
    min_samples_split,
    min_samples_leaf,
    max_depth,
):
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    n_estimators = int(n_estimators)
    min_samples_leaf = int(min_samples_leaf)
    min_samples_split = int(min_samples_split)
    max_depth = int(max_depth)
    batch_size = int(batch_size)
    n_iter = int(n_iter)

    rf = RandomForestQuantileRegressor(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )
    qwlstm_model = QWLSTMModel(
        hs=hidden_size, quantile=quantile, dropout=dropout, num_layers=num_layers
    )

    rf.fit(flatten(X_train).cpu(), flatten(Y_train).cpu())
    mrfw, mrfwn = get_rfweight(rf, flatten(X_train).cpu())

    qwlstm_model.fit(  # 使用训练集调优超参数（70%）
        X_train,
        Y_train,
        mrfw,
        tau=tau,  # 使用全局变量 tau
        d=False,
        batch_size=batch_size,
        n_iter=n_iter,
        lr=lr,
        tol=tol,
        verbose=False,
    )

    # 预测结果
    Y_pred = qwlstm_model.predict(X_val)

    # kupiec检验
    yp_big_num = violation(Y_val, Y_pred)
    kupiec_test(yp_big_num, len(Y_pred), quantile=quantile)

    return -target_loss(Y_val.cpu().numpy(), Y_pred, quantile=quantile)


def bayesian_optimization(_iter: int = 500):
    pbounds = {
        "dropout": (0.0, 0.5),
        "hidden_size": (4, 128),
        "n_iter": (500, 2000),
        "lr": (1e-4, 1e-1),
        "batch_size": (4, 256),
        "tol": (1e-6, 1e-4),
        "num_layers": (1, 4),
        "n_estimators": (50, 200),
        "min_samples_leaf": (1, 15),
        "min_samples_split": (2, 20),
        "max_depth": (5, 30),
    }

    optimizer = BayesianOptimization(f=calculate_loss, pbounds=pbounds, random_state=1)

    optimizer.maximize(init_points=30, n_iter=_iter)  # 优化完成
    best_params = optimizer.max["params"]
    print(optimizer.max)
        # 使用时间戳为文件命名，避免覆盖
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"best_params_{timestamp}.txt"
    with open("best_params.txt", "w") as f:
        json.dump(best_params, f)  # 将参数保存到文件中


def load_best_params():
    with open("best_params.txt", "r") as f:
        best_params = json.load(f)

    best_params_inted = {
        "hidden_size": int(best_params["hidden_size"]),
        "num_layers": int(best_params["num_layers"]),
        "n_estimators": int(best_params["n_estimators"]),
        "min_samples_split": int(best_params["min_samples_split"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
        "max_depth": int(best_params["max_depth"]),
        "batch_size": int(best_params["batch_size"]),
        "n_iter": int(best_params["n_iter"]),
    }

    best_params_float = {
        "dropout": best_params["dropout"],
        "lr": best_params["lr"],
        "tol": best_params["tol"],
        # tau 参数不再通过优化器调整
    }

    return {**best_params_inted, **best_params_float}


def train_model_1():  # 滚动向前预测训练
    try:
        best_params = load_best_params()
    except FileNotFoundError:
        print("best_params.txt not found. Please run bayesian_optimization() first.")
        return

    # 处理训练数据
    X = torch.cat((X_train, X_val, X_test), dim=0)
    Y = torch.cat((Y_train, Y_val, Y_test), dim=0)
    input_size = int(X.shape[0] * 0.8)
    total_size = X.shape[0]

    # 创建模型
    rf = RandomForestQuantileRegressor(
        n_estimators=best_params["n_estimators"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_depth=best_params["max_depth"],
    )
    qwlstm_model = QWLSTMModel(
        hs=best_params["hidden_size"],
        quantile=quantile,
        dropout=best_params["dropout"],
        num_layers=best_params["num_layers"],
    )

    # 开始训练
    Y_pred = np.ndarray((total_size - input_size))  # 20%的预测值
    for i in range(0, total_size - input_size):

        slice_start = i
        slice_end = i + input_size

        _X_train = X[slice_start:slice_end]
        _Y_train = Y[slice_start:slice_end]

        rf.fit(flatten(_X_train).cpu(), flatten(_Y_train).cpu())
        mrfw, mrfwn = get_rfweight(rf, flatten(_X_train).cpu())

        qwlstm_model.fit(  # 使用训练集调优超参数（70%）
            _X_train,
            _Y_train,
            mrfw,
            tau=tau,  # 使用全局变量 tau
            d=False,
            batch_size=best_params["batch_size"],
            n_iter=best_params["n_iter"],
            lr=best_params["lr"],
            tol=best_params["tol"],
            verbose=False,
        )

        with torch.no_grad():
            y = qwlstm_model.predict(X[-1].unsqueeze(0))  # 预测最后一个值

        Y_pred[i] = y[0]
        print(
            f"{i} times prediction finished!"
        )

    # 计算损失
    print("model training finished!")
    loss = target_loss(Y[input_size:], Y_pred, quantile=quantile)
    print("model last loss: ", loss)

    # 计算Kupiec检验
    yp_big_num = violation(Y[input_size:], Y_pred)
    kupiec_test(yp_big_num, len(Y_pred), quantile=quantile)
    print(kupiec_test(yp_big_num, len(Y_pred), quantile=quantile))

if __name__ == "__main__":
    bayesian_optimization(100)
    train_model_1()
