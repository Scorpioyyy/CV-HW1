"""
测试脚本
用法:
    python test.py [--model_path ./models/best_model.npy] [--data_dir ./data]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import FashionMNISTLoader
from src.model       import MLP
from src.evaluator   import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Fashion-MNIST 测试评估')
    parser.add_argument('--model_path', type=str, default='./models/best_model.npy', help='模型权重路径')
    parser.add_argument('--data_dir',   type=str, default='./data',                  help='数据目录')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载数据
    loader = FashionMNISTLoader(data_dir=args.data_dir)
    _, _, (X_test, y_test) = loader.load_data(val_ratio=0.1, normalize=True)

    # 2. 加载模型（从文件重建结构）
    if not os.path.exists(args.model_path):
        print(f'错误：未找到模型文件 {args.model_path}')
        print('请先运行 train.py 进行训练')
        sys.exit(1)

    print(f'加载模型: {args.model_path}')
    model = MLP.from_file(args.model_path)

    # 3. 评估
    evaluator = Evaluator(model)
    test_acc, cm = evaluator.full_report(X_test, y_test)

    return test_acc, cm


if __name__ == '__main__':
    main()
