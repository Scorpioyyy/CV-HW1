"""
超参数搜索脚本
支持网格搜索和随机搜索两种模式
用法:
    python search.py [--mode grid|random] [--epochs 20] [--data_dir ./data]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.data_loader           import FashionMNISTLoader
from src.hyperparameter_search import GridSearch, RandomSearch


def parse_args():
    parser = argparse.ArgumentParser(description='超参数搜索')
    parser.add_argument('--mode',     type=str, default='grid', choices=['grid', 'random'], help='搜索模式')
    parser.add_argument('--epochs',   type=int, default=20,     help='每组参数的训练轮数')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./results/search')
    parser.add_argument('--n_iter',   type=int, default=20,     help='随机搜索迭代次数')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)

    # 数据加载
    loader = FashionMNISTLoader(data_dir=args.data_dir)
    (X_train, y_train), (X_val, y_val), _ = loader.load_data(val_ratio=0.1, normalize=True)

    if args.mode == 'grid':
        # 网格搜索配置
        param_grid = {
            'lr':           [0.01, 0.05, 0.1],
            'hidden_sizes': [[256, 128], [512, 256]],
            'weight_decay': [0.0, 0.001, 0.01],
            'activation':   ['relu', 'tanh'],
            'epochs':       [args.epochs],
            'batch_size':   [128],
            'lr_decay':     [0.97],
            'momentum':     [0.9],
        }
        searcher = GridSearch(param_grid, save_dir=args.save_dir)
        results  = searcher.run(X_train, y_train, X_val, y_val)

    else:
        # 随机搜索配置
        param_distributions = {
            'lr':           {'type': 'log_uniform', 'low': 0.005, 'high': 0.2},
            'hidden_sizes': [[256, 128], [512, 256], [512, 128], [256, 64]],
            'weight_decay': [0.0, 0.0001, 0.001, 0.01],
            'activation':   ['relu', 'tanh'],
            'epochs':       [args.epochs],
            'batch_size':   [64, 128, 256],
            'lr_decay':     [0.95, 0.97, 0.99],
            'momentum':     [0.9],
        }
        searcher = RandomSearch(param_distributions, n_iter=args.n_iter,
                                save_dir=args.save_dir)
        results  = searcher.run(X_train, y_train, X_val, y_val)

    # 输出最优参数
    best_params, best_acc = searcher.best_params()
    print(f'\n最优参数组合:')
    for k, v in best_params.items():
        print(f'  {k}: {v}')
    print(f'最优验证集准确率: {best_acc:.4f}')

    return results


if __name__ == '__main__':
    main()
