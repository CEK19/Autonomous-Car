import time
import numpy as np
import tensorflow as tf
import torch
import argparse


def parseargs():
    parser = argparse.ArgumentParser(usage='Test GPU transfer speed in TensorFlow(default) and Pytorch.')
    parser.add_argument('--pytorch', action='store_true', help='Use PyTorch instead of TensorFlow')
    args = parser.parse_args()
    return args


class TimingModelTF(tf.keras.Model):
    def __init__(self, ):
        super(TimingModelTF, self).__init__()

    @tf.function
    def call(self, x):
        return tf.cast(x, dtype=tf.float32)[0, 0]


class TimingModelTorch(torch.nn.Module):
    def __init__(self, ):
        super(TimingModelTorch, self).__init__()
        self.device = torch.device('cuda')

    def forward(self, x):
        with torch.no_grad():
            return torch.from_numpy(x).to(self.device).type(torch.float32)[0, 0].cpu()


if __name__ == '__main__':
    args = parseargs()
    width = 256
    height = 256
    channels = 3
    iterations = 100
    model = TimingModelTorch() if args.pytorch else TimingModelTF()

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        img = np.random.randint(5, size=(batch_size, height, width, channels), dtype=np.uint8)

        result = model(img)
        result.numpy()

        start = time.time()
        for i in range(iterations):
            result = model(img)
            result.numpy()
        batch_time = (time.time() - start) / iterations
        print(f'Batch size {batch_size}; Batch time {batch_time:.4f}; BPS {1 / batch_time:.1f}; FPS {(1 / batch_time) * batch_size:.1f}; MB/S {(((1 / batch_time) * batch_size) * 256 * 256 * 3) / 1000000:.1f}')
