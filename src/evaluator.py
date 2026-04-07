"""
测试评估模块
加载训练好的最优模型，输出测试集准确率与混淆矩阵
"""

import numpy as np
from .data_loader import get_batches, FashionMNISTLoader


class Evaluator:
    """
    模型评估器

    功能：
        - 批量预测
        - 计算整体准确率
        - 计算并打印混淆矩阵
        - 返回分类错误的样本索引
    """

    CLASS_NAMES = FashionMNISTLoader.CLASS_NAMES
    N_CLASSES   = len(CLASS_NAMES)

    def __init__(self, model):
        self.model = model

    # 批量预测
    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """返回预测标签，形状 (N,)"""
        preds = []
        for X_batch, _ in get_batches(X, np.zeros(len(X)), batch_size, shuffle=False):
            preds.append(self.model.predict(X_batch))
        return np.concatenate(preds)

    # 准确率
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float((preds == y).mean())

    # 混淆矩阵
    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """返回形状 (N_CLASSES, N_CLASSES) 的混淆矩阵，行=真实，列=预测"""
        preds = self.predict(X)
        cm    = np.zeros((self.N_CLASSES, self.N_CLASSES), dtype=np.int64)
        for true, pred in zip(y, preds):
            cm[true, pred] += 1
        return cm

    def print_confusion_matrix(self, cm: np.ndarray):
        """格式化打印混淆矩阵及各类准确率"""
        names = [n[:4] for n in self.CLASS_NAMES]   # 截短名称
        col_w = 6

        print('\n' + '='*70)
        print('混淆矩阵（行=真实类别，列=预测类别）')
        print('='*70)

        # 表头
        header = f'{"":8s}' + ''.join(f'{n:>{col_w}}' for n in names)
        print(header)
        print('-' * len(header))

        for i, name in enumerate(names):
            row = f'{name:8s}' + ''.join(f'{cm[i, j]:>{col_w}}' for j in range(self.N_CLASSES))
            print(row)

        print('-' * len(header))

        # 每类准确率
        print('\n各类别准确率:')
        total_correct = 0
        for i, name in enumerate(self.CLASS_NAMES):
            n_total   = cm[i].sum()
            n_correct = cm[i, i]
            acc       = n_correct / n_total if n_total > 0 else 0.0
            total_correct += n_correct
            print(f'  {name:8s}: {acc:.4f}  ({n_correct}/{n_total})')

        overall = total_correct / cm.sum() if cm.sum() > 0 else 0.0
        print(f'\n  整体准确率: {overall:.4f}')

    # 错误样本分析
    def get_error_samples(self, X: np.ndarray, y: np.ndarray, n: int = 12, seed: int = 0):
        """
        返回被错误分类的样本子集

        返回:
            indices   : 原始数据集中的索引
            true_labels
            pred_labels
        """
        np.random.seed(seed)
        preds        = self.predict(X)
        error_mask   = (preds != y)
        error_idx    = np.where(error_mask)[0]

        if len(error_idx) > n:
            error_idx = np.random.choice(error_idx, n, replace=False)
            error_idx = np.sort(error_idx)

        return error_idx, y[error_idx], preds[error_idx]

    # 完整评估报告
    def full_report(self, X: np.ndarray, y: np.ndarray):
        """打印完整测试评估报告"""
        print('\n' + '='*70)
        print('测试集评估报告')
        print('='*70)

        acc = self.accuracy(X, y)
        print(f'测试集准确率: {acc:.4f} ({acc*100:.2f}%)')

        cm = self.confusion_matrix(X, y)
        self.print_confusion_matrix(cm)

        return acc, cm
