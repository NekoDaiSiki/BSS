from PIL import ImageFilter
import random

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
	# 将一个图像的两个随机裁剪的图片作为“查询”和“键”
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
	# 『高斯模糊增强(Gaussian blur augmentation)』的原理详见 SimCLR
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x