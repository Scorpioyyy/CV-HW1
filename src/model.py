"""
模型定义模块
手动实现三层 MLP（两个隐藏层 + 输出层），支持 ReLU / Sigmoid / Tanh 激活函数
"""

import numpy as np


#  激活函数
class ReLU:
    """ReLU(x) = max(0, x)"""

    def forward(self, x):
        self._cache = x            # 缓存输入，供反向传播使用
        return np.maximum(0.0, x)

    def backward(self, grad):
        # dL/dx = dL/d(relu) * 1_{x>0}
        return grad * (self._cache > 0)


class Sigmoid:
    """Sigmoid(x) = 1 / (1 + exp(-x))"""

    def forward(self, x):
        # 数值裁剪防止 exp 溢出
        out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        self._cache = out          # 缓存 sigmoid 输出
        return out

    def backward(self, grad):
        # dL/dx = dL/d(sigmoid) * sigmoid * (1 - sigmoid)
        s = self._cache
        return grad * s * (1.0 - s)


class Tanh:
    """Tanh(x) = tanh(x)"""

    def forward(self, x):
        out = np.tanh(x)
        self._cache = out          # 缓存 tanh 输出
        return out

    def backward(self, grad):
        # dL/dx = dL/d(tanh) * (1 - tanh^2)
        return grad * (1.0 - self._cache ** 2)


def get_activation(name: str):
    """根据名称返回激活函数实例"""
    mapping = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}
    name = name.lower()
    if name not in mapping:
        raise ValueError(f'不支持的激活函数: {name}，可选: {list(mapping.keys())}')
    return mapping[name]()


#  线性层
class Linear:
    """全连接线性层: z = x @ W + b"""

    def __init__(self, in_features: int, out_features: int, activation: str = 'relu'):
        # He 初始化（适用于 ReLU 系激活函数，对 Sigmoid/Tanh 也足够）
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.b = np.zeros(out_features, dtype=np.float32)
        # 梯度
        self.dW: np.ndarray = None
        self.db: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x                        # 缓存输入，用于反向传播
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        参数:
            grad: dL/dz，形状 (batch, out_features)
        返回:
            dL/dx，形状 (batch, in_features)
        同时计算并保存 dL/dW 和 dL/db
        """
        x = self._cache
        self.dW = x.T @ grad                   # (in, out)
        self.db = grad.sum(axis=0)             # (out,)
        return grad @ self.W.T                 # (batch, in)


#  三层 MLP
class MLP:
    """
    三层多层感知机（两个隐藏层 + 输出层）

    结构:
        输入(784) → Linear1 → Act → Linear2 → Act → Linear3 → softmax → 预测

    参数:
        input_size   : 输入维度，默认 784（Fashion-MNIST 展平后）
        hidden_sizes : 两个隐藏层的神经元数，如 [256, 128]
        output_size  : 输出类别数，默认 10
        activation   : 激活函数名称，'relu' | 'sigmoid' | 'tanh'
        weight_decay : L2 正则化系数（lambda）
    """

    def __init__(
        self,
        input_size:   int       = 784,
        hidden_sizes: list      = None,
        output_size:  int       = 10,
        activation:   str       = 'relu',
        weight_decay: float     = 0.0,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.weight_decay = weight_decay
        self.activation_name = activation

        # 构建层（最后一层无激活函数，由 softmax + 交叉熵统一处理）
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.linears    = [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.activations = [get_activation(activation) for _ in hidden_sizes]

    # 前向传播
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
            x: 输入，形状 (batch, 784)
        返回:
            logits，形状 (batch, 10)（未经 softmax）
        """
        out = x
        for i, linear in enumerate(self.linears):
            out = linear.forward(out)
            if i < len(self.activations):          # 最后一层不加激活
                out = self.activations[i].forward(out)
        return out  # logits

    # 损失函数
    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """数值稳定的 softmax"""
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_x   = np.exp(shifted)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def loss(self, logits: np.ndarray, labels: np.ndarray):
        """
        交叉熵损失 + L2 正则化

        返回:
            (scalar_loss, probs)
        """
        n     = len(labels)
        probs = self.softmax(logits)

        # 交叉熵（取正确类别的对数概率）
        log_p       = np.log(probs[np.arange(n), labels] + 1e-12)
        ce_loss     = -log_p.mean()

        # L2 正则化（仅对权重 W，不对偏置 b）
        l2_loss = 0.0
        for lin in self.linears:
            l2_loss += np.sum(lin.W ** 2)
        l2_loss *= 0.5 * self.weight_decay

        return ce_loss + l2_loss, probs

    # 反向传播
    def backward(self, probs: np.ndarray, labels: np.ndarray):
        """
        手工反向传播，计算所有参数的梯度并存储在各层的 dW / db 中

        参数:
            probs : softmax 概率，形状 (batch, 10)
            labels: 真实标签，形状 (batch,)
        """
        n = len(labels)

        # softmax + 交叉熵组合梯度：dL/d(logits)
        # 对正确类别减 1，再除以 batch size
        grad = probs.copy()
        grad[np.arange(n), labels] -= 1.0
        grad /= n

        # 从输出层向输入层逐层反传
        # layers 顺序: [Linear1, Linear2, Linear3]
        # activations: [Act1,    Act2]
        for i in range(len(self.linears) - 1, -1, -1):
            # 经过线性层反传（同时计算 dW、db）
            grad = self.linears[i].backward(grad)
            # L2 正则化梯度：dL2/dW = lambda * W
            self.linears[i].dW += self.weight_decay * self.linears[i].W

            # 经过激活函数反传（最后一层 i=len-1 没有对应的激活函数）
            if i > 0:
                grad = self.activations[i - 1].backward(grad)

    # 预测
    def predict(self, x: np.ndarray) -> np.ndarray:
        """返回预测类别（argmax of logits），形状 (batch,)"""
        return np.argmax(self.forward(x), axis=1)

    # 参数保存与加载
    def get_params(self) -> dict:
        params = {}
        for i, lin in enumerate(self.linears):
            params[f'W{i}'] = lin.W.copy()
            params[f'b{i}'] = lin.b.copy()
        # 保存网络配置，方便加载时重建
        params['_config'] = {
            'input_size':   self.linears[0].W.shape[0],
            'hidden_sizes': [lin.W.shape[0] for lin in self.linears[1:]],
            'output_size':  self.linears[-1].W.shape[1],
            'activation':   self.activation_name,
            'weight_decay': self.weight_decay,
        }
        return params

    def set_params(self, params: dict):
        for i, lin in enumerate(self.linears):
            lin.W = params[f'W{i}'].copy()
            lin.b = params[f'b{i}'].copy()

    def save(self, filepath: str):
        """保存权重到 .npy 文件"""
        np.save(filepath, self.get_params(), allow_pickle=True)

    def load(self, filepath: str):
        """从 .npy 文件加载权重"""
        params = np.load(filepath, allow_pickle=True).item()
        self.set_params(params)

    @classmethod
    def from_file(cls, filepath: str) -> 'MLP':
        """从文件中重建模型（包含结构配置）"""
        params = np.load(filepath, allow_pickle=True).item()
        cfg    = params['_config']
        model  = cls(**cfg)
        model.set_params(params)
        return model
