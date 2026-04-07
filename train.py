"""
训练脚本
用法:
    python train.py [--epochs 50] [--lr 0.05] [--hidden 256 128] [--activation relu]
                    [--weight_decay 0.001] [--batch_size 128] [--lr_decay 0.97]
                    [--data_dir ./data] [--save_dir ./models]
"""

import argparse
import sys
import os

# 确保可以导入 src 包
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.data_loader import FashionMNISTLoader
from src.model       import MLP
from src.trainer     import SGD, ExponentialLRScheduler, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Fashion-MNIST 三层 MLP 训练')
    parser.add_argument('--epochs',       type=int,   default=50,        help='训练轮数')
    parser.add_argument('--lr',           type=float, default=0.05,       help='初始学习率')
    parser.add_argument('--hidden',       type=int,   nargs='+', default=[256, 128], help='隐藏层大小')
    parser.add_argument('--activation',   type=str,   default='relu',    choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--weight_decay', type=float, default=0.001,     help='L2正则化系数')
    parser.add_argument('--batch_size',   type=int,   default=128,       help='批大小')
    parser.add_argument('--lr_decay',     type=float, default=0.97,      help='指数学习率衰减系数')
    parser.add_argument('--momentum',     type=float, default=0.9,       help='SGD动量')
    parser.add_argument('--data_dir',     type=str,   default='./data',  help='数据目录')
    parser.add_argument('--save_dir',     type=str,   default='./models', help='模型保存目录')
    parser.add_argument('--seed',         type=int,   default=42,        help='随机种子')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # 1. 数据加载
    loader = FashionMNISTLoader(data_dir=args.data_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_data(
        val_ratio=0.1, normalize=True, seed=args.seed
    )

    # 2. 模型定义
    model = MLP(
        input_size   = X_train.shape[1],  # 784
        hidden_sizes = args.hidden,
        output_size  = 10,
        activation   = args.activation,
        weight_decay = args.weight_decay,
    )
    print(f'\n模型结构: 784 → {" → ".join(map(str, args.hidden))} → 10')
    print(f'激活函数: {args.activation}')
    total_params = sum(lin.W.size + lin.b.size for lin in model.linears)
    print(f'总参数量: {total_params:,}')

    # 3. 优化器与学习率调度
    optimizer = SGD(lr=args.lr, momentum=args.momentum)
    scheduler = ExponentialLRScheduler(optimizer, gamma=args.lr_decay)

    # 4. 训练
    trainer = Trainer(model, optimizer, scheduler, save_dir=args.save_dir)
    history = trainer.train(
        X_train, y_train,
        X_val,   y_val,
        epochs     = args.epochs,
        batch_size = args.batch_size,
    )

    # 5. 最终测试集评估（加载最优权重）
    from src.evaluator import Evaluator
    best_path = os.path.join(args.save_dir, 'best_model.npy')
    model.load(best_path)
    evaluator = Evaluator(model)
    test_acc, cm = evaluator.full_report(X_test, y_test)

    return history, test_acc


if __name__ == '__main__':
    main()
