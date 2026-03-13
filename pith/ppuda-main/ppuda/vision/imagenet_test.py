# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ImageNet wrapper.

"""
import hashlib
import time
import io
import numpy as np
from PIL import Image
import torchvision
import torchvision.datasets
from collections import defaultdict



def split_train_and_val(list_of_tups, num_val_per_class=50):

    class_dict = defaultdict(list)

    for item in list_of_tups:
        class_dict[item[1]].append(item)

    train_samples, val_samples = [], []

    # fix the class ordering
    for k in sorted(class_dict.keys()):
        v = class_dict[k]

        # last num_val_per_class will be the val samples
        train_samples.extend(v[:-num_val_per_class])
        val_samples.extend(v[-num_val_per_class:])

    return train_samples, val_samples




class ImageNetDataset(torchvision.datasets.ImageNet):
    """ImageNet数据集 with 性能分析"""

    def __init__(self, root, split, transform=None, has_validation=True, enable_timing=True):
        assert split in {"train", "val"}
        
        base_split = 'train' if (split == 'val' and has_validation) else split
        
        super().__init__(root, base_split, transform=transform)

        self.split = split

        if has_validation:
            train_samples, val_samples = split_train_and_val(
                self.samples, num_val_per_class=50)

            self.samples = train_samples if split == "train" else val_samples

        m = hashlib.sha256()
        m.update(str(self.samples).encode())
        self.checksum = int(m.hexdigest(), 16)
        
        # 🔥 性能分析
        self.enable_timing = enable_timing
        if enable_timing:
            self.io_times = []
            self.decode_times = []
            self.transform_times = []

    @property
    def num_examples(self):
        return len(self)
    
    def __getitem__(self, index):
        """带性能分析的数据加载"""
        path, target = self.samples[index]
        
        if not self.enable_timing:
            # 🔥 不计时的快速路径
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        
        # 🔥 带计时的路径
        # 1️⃣ 文件IO
        t0 = time.time()
        with open(path, 'rb') as f:
            img_bytes = f.read()
        io_time = (time.time() - t0) * 1000
        
        # 2️⃣ 图片解码
        t0 = time.time()
        sample = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        decode_time = (time.time() - t0) * 1000
        
        # 3️⃣ Transform
        t0 = time.time()
        if self.transform is not None:
            sample = self.transform(sample)
        transform_time = (time.time() - t0) * 1000
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # 记录
        self.io_times.append(io_time)
        self.decode_times.append(decode_time)
        self.transform_times.append(transform_time)
        
        # 每1000个样本打印
        if len(self.io_times) % 200 == 0:
            print(f"\n{'='*70}")
            print(f"📊 数据加载性能 (最近1000样本, split={self.split}):")
            print(f"{'='*70}")
            print(f"  文件IO:     {np.mean(self.io_times[-1000:]):7.2f}ms  (读JPEG文件)")
            print(f"  图片解码:   {np.mean(self.decode_times[-1000:]):7.2f}ms  (JPEG->RGB)")
            print(f"  Transform:  {np.mean(self.transform_times[-1000:]):7.2f}ms  (数据增强)")
            print(f"  " + "-"*66)
            total = sum([np.mean(self.io_times[-1000:]),
                       np.mean(self.decode_times[-1000:]),
                       np.mean(self.transform_times[-1000:])])
            print(f"  总计:       {total:7.2f}ms")
            print(f"{'='*70}\n")
        
        return sample, target