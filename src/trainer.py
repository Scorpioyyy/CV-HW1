"""
训练循环模块
实现：SGD 优化器（带 Momentum）、学习率衰减、训练循环、最优模型自动保存
"""

import numpy as np
import os
import json
import time

from .data_loader import get_batches


#  SGD 优化器
class SGD:
    """
    随机梯度下降优化器，支持 Momentum

    参数:
        lr       : 初始学习率
        momentum : 动量系数（0 表示普通 SGD）
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr       = lr
        self.momentum = momentum
        self._velocity: dict = {}  # 每个参数对应的速度

    def step(self, model):
        """根据各层存储的梯度更新参数"""
        for i, lin in enumerate(model.linears):
            if lin.dW is None:
                continue

            # 初始化速度
            if i not in self._velocity:
                self._velocity[i] = {
                    'W': np.zeros_like(lin.W),
                    'b': np.zeros_like(lin.b),
                }

            vW = self._velocity[i]['W']
            vb = self._velocity[i]['b']

            # v = momentum * v - lr * grad
            vW = self.momentum * vW - self.lr * lin.dW
            vb = self.momentum * vb - self.lr * lin.db

            lin.W += vW
            lin.b += vb

            self._velocity[i]['W'] = vW
            self._velocity[i]['b'] = vb


#  学习率调度器
class StepLRScheduler:
    """
    阶梯式学习率衰减：每隔 step_size 个 epoch，lr 乘以 gamma

    参数:
        optimizer  : SGD 优化器实例
        step_size  : 衰减间隔（epoch 数）
        gamma      : 衰减系数（< 1）
    """

    def __init__(self, optimizer: SGD, step_size: int = 10, gamma: float = 0.5):
        self.optimizer  = optimizer
        self.step_size  = step_size
        self.gamma      = gamma
        self._epoch_cnt = 0

    def step(self):
        self._epoch_cnt += 1
        if self._epoch_cnt % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLRScheduler:
    """指数学习率衰减：每个 epoch 后 lr *= gamma"""

    def __init__(self, optimizer: SGD, gamma: float = 0.97):
        self.optimizer = optimizer
        self.gamma     = gamma

    def step(self):
        self.optimizer.lr *= self.gamma


#  训练器
class Trainer:
    """
    训练器，封装完整训练流程

    功能：
        - 按 epoch 训练，支持 mini-batch SGD
        - 每个 epoch 结束后在验证集上评估
        - 自动保存验证集上准确率最高的模型权重
        - 记录并返回训练历史（loss / accuracy）
    """

    def __init__(self, model, optimizer: SGD, scheduler=None, save_dir: str = './models'):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir  = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss':   [],
            'val_acc':    [],
            'lr':         [],
        }
        self.best_val_acc = 0.0

    # 单轮训练
    def _train_epoch(self, X_train, y_train, batch_size: int) -> float:
        """训练一个 epoch，返回平均训练损失"""
        total_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch in get_batches(X_train, y_train, batch_size, shuffle=True):
            # 前向传播
            logits = self.model.forward(X_batch)
            loss, probs = self.model.loss(logits, y_batch)

            # 反向传播
            self.model.backward(probs, y_batch)

            # 参数更新
            self.optimizer.step(self.model)

            total_loss += loss
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # 验证/测试评估
    def evaluate(self, X, y, batch_size: int = 512):
        """计算给定数据集上的平均损失和准确率"""
        total_loss = 0.0
        correct    = 0
        n_batches  = 0

        for X_batch, y_batch in get_batches(X, y, batch_size, shuffle=False):
            logits = self.model.forward(X_batch)
            loss, probs = self.model.loss(logits, y_batch)
            preds = np.argmax(probs, axis=1)

            total_loss += loss
            correct    += (preds == y_batch).sum()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / len(y)
        return avg_loss, accuracy

    # 完整训练流程
    def train(
        self,
        X_train, y_train,
        X_val,   y_val,
        epochs:     int = 50,
        batch_size: int = 128,
        verbose:    bool = True,
    ) -> dict:
        """
        完整训练过程

        返回:
            history dict，包含 train_loss / val_loss / val_acc / lr
        """
        print(f'\n{"="*60}')
        print(f'开始训练 | epochs={epochs} | batch_size={batch_size}')
        print(f'训练集: {len(X_train)} | 验证集: {len(X_val)}')
        print(f'初始学习率: {self.optimizer.lr:.5f}')
        print(f'{"="*60}')

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # 训练
            train_loss = self._train_epoch(X_train, y_train, batch_size)

            # 验证
            val_loss, val_acc = self.evaluate(X_val, y_val)

            # 记录历史
            self.history['train_loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_acc'].append(float(val_acc))
            self.history['lr'].append(float(self.optimizer.lr))

            # 保存最优模型
            flag = ''
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save(os.path.join(self.save_dir, 'best_model.npy'))
                flag = '  ← 保存最优'

            if verbose:
                elapsed = time.time() - t0
                print(
                    f'Epoch {epoch:3d}/{epochs} | '
                    f'Train Loss: {train_loss:.4f} | '
                    f'Val Loss: {val_loss:.4f} | '
                    f'Val Acc: {val_acc:.4f} | '
                    f'LR: {self.optimizer.lr:.6f} | '
                    f'{elapsed:.1f}s{flag}'
                )

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

        # 保存训练历史
        hist_path = os.path.join(self.save_dir, 'history.json')
        with open(hist_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        print(f'\n训练完成！最优验证集准确率: {self.best_val_acc:.4f}')
        print(f'最优模型已保存至: {os.path.join(self.save_dir, "best_model.npy")}')

        return self.history
