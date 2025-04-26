import numpy as np
from PIL import Image,ImageFilter
import random
import torchvision.transforms as transforms

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=0.5):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
    
class Addblur(object):

    def __init__(self, p=0.5,blur="normal"):
        #         self.density = density
        self.p = p
        self.blur= blur

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: 
            if self.blur== "normal":
                img = img.filter(ImageFilter.BLUR)
                return img
            if self.blur== "Gaussian":
                img = img.filter(ImageFilter.GaussianBlur)
                return img
            if self.blur== "mean":
                img = img.filter(ImageFilter.BoxBlur)
                return img

        else:
            return img

class ReBlur(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_transform = transforms.Compose([
                            transforms.Resize([224, 56]),
                            transforms.Resize([224, 224]),])
            img = img_transform(img)
        return img
    
class RandomShift(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            if random.uniform(0, 1) < 0.5:
                img_transform = transforms.Compose([transforms.RandomCrop(224, padding=(32,0),padding_mode='reflect')])
            else:
                img_transform = transforms.Compose([transforms.RandomCrop(224, padding=(16,0),padding_mode='reflect')])
            img = img_transform(img)
        return img

class RandomSpi(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            finalImg = Image.new('RGB', (448, 224))
            finalImg.paste(img, (0, 0))
            finalImg.paste(img, (224, 0))
            img_transform = transforms.Compose([transforms.RandomCrop(224)])
            img = img_transform(finalImg)
        return img
    
class RandomCpr(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            if random.uniform(0, 1) < 0.5:
                img_transform = transforms.Compose([transforms.Pad(padding=(24,0),padding_mode='edge')])
            else:
                img_transform = transforms.Compose([transforms.Pad(padding=(48,0),padding_mode='edge')])
            img = img_transform(img)
        return img
    
class RandomComPre(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            rpadnum = random.randint(24, 112) 
            img_transform = transforms.Compose([transforms.Pad(padding=(0,rpadnum),padding_mode='reflect')])
            img = img_transform(img)
        return img