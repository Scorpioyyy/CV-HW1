"""
数据加载与预处理模块
Fashion-MNIST 数据集：10类服装图像，每张 28×28 灰度图
"""

import numpy as np
import os
import gzip
import struct
import urllib.request


class FashionMNISTLoader:
    """Fashion-MNIST 数据集加载器"""

    # 官方数据下载地址
    URLS = {
        'train_images': 'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'train_labels': 'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'test_images':  'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'test_labels':  'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    }

    # 10 个类别名称
    CLASS_NAMES = [
        'T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
        '凉鞋', '衬衫', '运动鞋', '包', '短靴'
    ]
    CLASS_NAMES_EN = [
        'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
    ]

    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download(self):
        """
        下载 Fashion-MNIST 原始二进制文件

        此处仅使用 torchvision 完成数据下载，
        不使用其任何模型、自动微分或训练相关功能。
        所有后续数据读取均通过 NumPy 完成。
        """
        # 检查是否已有所有文件
        all_filenames = [url.split('/')[-1] for url in self.URLS.values()]
        if all(os.path.exists(os.path.join(self.data_dir, f)) for f in all_filenames):
            print('  数据文件已存在，跳过下载')
            return

        # 仅用于下载：使用 torchvision 下载 Fashion-MNIST ──
        try:
            import torchvision  # 仅用于数据下载，不用于模型训练
            print('  使用 torchvision 下载 Fashion-MNIST（仅下载功能）...')
            torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=True,  download=True
            )
            torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=False, download=True
            )
            # torchvision 下载到 FashionMNIST/raw/ 子目录，移动到 data_dir
            raw_dir = os.path.join(self.data_dir, 'FashionMNIST', 'raw')
            for fname in all_filenames:
                src = os.path.join(raw_dir, fname)
                dst = os.path.join(self.data_dir, fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)
                    print(f'  复制: {fname}')
            print('  下载完成')
        except ImportError:
            # torchvision 不可用时，回退到直接 HTTP 下载
            print('  torchvision 不可用，尝试直接下载...')
            for name, url in self.URLS.items():
                filename = url.split('/')[-1]
                filepath = os.path.join(self.data_dir, filename)
                if not os.path.exists(filepath):
                    print(f'  下载 {filename} ...')
                    urllib.request.urlretrieve(url, filepath)
                    print(f'  完成: {filename}')

    def _load_images(self, filename):
        """解析 IDX3 格式图像文件"""
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            assert magic == 2051, f'图像文件 magic number 错误: {magic}'
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num, rows * cols)  # 展平为 784 维向量
        return data

    def _load_labels(self, filename):
        """解析 IDX1 格式标签文件"""
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            assert magic == 2049, f'标签文件 magic number 错误: {magic}'
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def load_data(self, val_ratio=0.1, normalize=True, seed=42):
        """
        加载并预处理数据，划分训练/验证/测试集

        参数:
            val_ratio: 验证集占训练集的比例（默认 10%）
            normalize: 是否做均值-方差标准化
            seed: 随机种子，保证可复现

        返回:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
            X 形状: (N, 784)，dtype float32
            y 形状: (N,)，dtype int32，值域 [0, 9]
        """
        print('加载 Fashion-MNIST 数据集...')
        self.download()

        # 加载原始数据（uint8，[0, 255]）
        train_images = self._load_images('train-images-idx3-ubyte.gz')
        train_labels = self._load_labels('train-labels-idx1-ubyte.gz')
        test_images  = self._load_images('t10k-images-idx3-ubyte.gz')
        test_labels  = self._load_labels('t10k-labels-idx1-ubyte.gz')

        # 转为 float32 并归一化到 [0, 1]
        train_images = train_images.astype(np.float32) / 255.0
        test_images  = test_images.astype(np.float32)  / 255.0

        if normalize:
            # 使用训练集的均值和标准差做 z-score 标准化
            mean = train_images.mean()
            std  = train_images.std()
            train_images = (train_images - mean) / (std + 1e-8)
            test_images  = (test_images  - mean) / (std + 1e-8)
            self.mean, self.std = mean, std
            print(f'  标准化: mean={mean:.4f}, std={std:.4f}')

        # 随机划分验证集
        np.random.seed(seed)
        n_total = len(train_images)
        n_val   = int(n_total * val_ratio)
        perm    = np.random.permutation(n_total)
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        X_val, y_val     = train_images[val_idx],   train_labels[val_idx].astype(np.int32)
        X_train, y_train = train_images[train_idx], train_labels[train_idx].astype(np.int32)
        X_test,  y_test  = test_images,             test_labels.astype(np.int32)

        print(f'  训练集: {X_train.shape[0]} 样本')
        print(f'  验证集: {X_val.shape[0]} 样本')
        print(f'  测试集: {X_test.shape[0]} 样本')

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_batches(X, y, batch_size, shuffle=True):
    """生成 mini-batch 迭代器"""
    n = len(X)
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = indices[start: start + batch_size]
        yield X[idx], y[idx]
