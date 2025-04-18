# Quantile Regresson Forest Weight Long Short Term Memory（QWLSTM）
# 分位数回归森林加权长短期记忆网络模型 实现
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import warnings
from dataloader import device

warnings.filterwarnings("ignore")


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
    ):
        """lstm模型实现

        Args:
            input_size (int): 输入大小
            hidden_size (int): 隐藏层大小
            num_layers (int): 层数
            output_size (int): 输出大小
            dropout (float): 关闭神经元的概率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(  # lstm
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, output_size)  # 线性层

    def forward(self, x):
        # 初始化 LSTM 隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 通过 LSTM 层
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


def rf_graph(x):

    G = np.zeros((len(x), len(x)))

    for i in range(len(x)):

        G[i, np.where(x == x[i])[0]] = 1

    nodes = Counter(x)
    nodes_num = np.array([nodes[i] for i in x])

    return G, G / nodes_num.reshape(-1, 1)


def get_rfweight(rf, x):

    n = x.shape[0]

    leaf = rf.apply(x)
    ntrees = leaf.shape[1]
    G_unnorm = np.zeros((n, n))
    G_norm = np.zeros((n, n))

    for i in range(ntrees):

        tmp1, tmp2 = rf_graph(leaf[:, i])
        G_unnorm += tmp1
        G_norm += tmp2

    return G_unnorm / ntrees, G_norm / ntrees


def get_derivative_matrix(A, B, device=device):

    n = A.shape[0]
    p = A.shape[1]

    C = torch.zeros(n, n * p).to(device)

    row_idx = torch.arange(n).repeat((p, 1)).T.reshape(-1)
    col_idx = torch.arange(n * p)

    C[row_idx, col_idx] = A.reshape(-1)

    mB = torch.tile(B.T, (n, 1)) - torch.tile(B.reshape(-1, 1), (1, n))
    mB = mB.to(device)

    return C @ mB


class QWLSTMModel:

    def __init__(
        self,
        hs=64,
        quantile: float = 0.05,
        dropout: float = 0.1,
        num_layers: int = 3,
        device=device,
    ):
        """

        Args:
            hs (int, optional): 隐藏层. Defaults to 64.
            quantile (float, optional): 分位数. Defaults to 0.05.
            dropout (float, optional): 随机关闭神经元. Defaults to .1
            num_layers (int, optional): lstm隐藏层数量. Defaults to 3
            device (str, optional): 是否使用gpu加速. Defaults to 'cpu'.
        """
        self.hs = hs
        self.device = device
        self.q = 1 - quantile
        self.dropout = dropout
        self.num_layers = num_layers

    def fit(
        self,
        x,
        y,
        weight,
        tau=0.5,
        d=False,
        batch_size=100,
        n_iter=500,
        lr=1e-3,
        tol=1e-5,
        verbose=True,
    ):

        x, y = x.to(self.device), y.to(self.device)

        n = x.shape[0]
        p = x.shape[1]
        input_size = x.shape[2]

        self.fnet = LSTMModel(
            input_size=input_size,
            hidden_size=self.hs,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            [
                {"params": self.fnet.parameters()},
            ],
            lr=lr,
        )

        self.loss_count = []
        last_loss = np.inf
        flag = 0

        for i_iter in range(n_iter):

            csample = np.random.permutation(n)[:batch_size]
            tmp_x = x[csample].to(self.device)
            tmp_y = y[csample].to(self.device)
            tmp_w = (
                torch.Tensor(weight[
                    np.tile(csample, (batch_size, 1)).T.ravel(),
                    np.tile(csample, (1, batch_size)),
                ]
                .reshape(batch_size, -1)
                .T).to(self.device)
            )
            tmp_w = tmp_w if tau is None else tmp_w - torch.diag(torch.diag(tmp_w))
            tmp_fx = self.fnet(tmp_x)
            # tmp_my = torch.tile(tmp_y, (batch_size, 1 ))
            # tmp_mfx = torch.tile(tmp_fx, (1, batch_size))

            if tau is None:
                loss = self.quantile_loss(tmp_y, tmp_fx.ravel(), self.q).mean()
            else:
                loss1 = self.quantile_loss(tmp_y, tmp_fx.ravel(), self.q).mean()
                loss2 = (
                    torch.mean(
                        self.quantile_loss(tmp_y, tmp_fx.ravel(), self.q) * tmp_w.T
                    )
                    * n
                    / (n - 1)
                )
                loss = tau * loss1 + (1 - tau) * loss2

            self.loss_count.append(loss.data.cpu().tolist())

            if (np.abs(last_loss - loss.data.cpu().numpy()) <= tol) & (i_iter >= 100):

                if verbose:

                    print(
                        "Algorithm converges for RWN model at iter {}, loss: {}.".format(
                            i_iter, self.loss_count[-1]
                        )
                    )

                flag = 1
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.data.cpu().numpy()

        if (flag == 0) & verbose:

            print(
                "Algorithm may not converge for RWN model, loss: {}.".format(
                    self.loss_count[-1]
                )
            )

    def predict(self, x_new):

        x_new = x_new.to(self.device)
        x_new = x_new.reshape(-1, 1) if x_new.ndim == 1 else x_new

        self.fnet.to(self.device)

        with torch.no_grad():
            y_pred = self.fnet(x_new)

        return y_pred.cpu().numpy().ravel()

    def predict_derivative(self, x_new):

        x_new = x_new.to(self.device)
        x_new = x_new.reshape(-1, 1) if x_new.ndim == 1 else x_new

        return self.dnet.cpu()(x_new).data.numpy()

    def quantile_loss(self, y_pred, y_true, q):
        errors = y_true - y_pred.detach()
        max_errors = torch.max(q * errors, (q - 1) * errors)
        return max_errors
