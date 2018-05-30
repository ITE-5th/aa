import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

from classes import classes

FGSM = 'FGSM'
ILLC = 'ILLC'


class FastGradientMethod:

    def __init__(self, model_name="resnet18"):
        self.model = getattr(models, model_name)(pretrained=True)
        self.model.eval().cuda()

    def modify_image(self, image_path, eps=10, method=FGSM):
        orig = cv2.imread(image_path)[..., ::-1]
        orig = cv2.resize(orig, (224, 224))
        img = orig.copy().astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img /= 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        inp = Variable(torch.from_numpy(img).cuda().float().unsqueeze(0), requires_grad=True)
        out = self.model(inp)
        original_pred = np.argmax(out.data.cpu().numpy())
        targeted_pred = np.argmin(out.data.cpu().numpy())
        original_class = classes[original_pred].split(',')[0]
        targeted_class = classes[targeted_pred].split(',')[0]
        if method == FGSM:
            if eps != 0:
                adv, perturbation, adv_class = self.modify(img, eps, original_pred, mean, std)
            else:
                adv_class = original_class
                while original_class == adv_class:
                    eps += 1
                    adv, perturbation, adv_class = self.modify(img, eps, original_pred, mean, std)
        elif method == ILLC:
            if eps != 0:
                adv, perturbation, adv_class = self.modify(img, eps, targeted_pred, mean, std, True)
            else:
                adv_class = original_class
                print(f'target is : {targeted_class}')
                while targeted_class != adv_class and eps < 50:
                    eps += 1
                    adv_class_prev = adv_class
                    adv, perturbation, adv_class = self.modify(img, eps, targeted_pred, mean, std, True)
                    if adv_class_prev != adv_class:
                        print(adv_class)
        adv_name = "adv.jpg"
        perturbation_name = "perturbation.jpg"
        cv2.imwrite(adv_name, adv)
        cv2.imwrite(perturbation_name, perturbation)
        return original_class, adv_class, adv_name, perturbation_name

    def modify(self, img, eps, original_pred, mean, std, targeted=False):
        criterion = nn.CrossEntropyLoss().cuda()
        inp = Variable(torch.from_numpy(img).cuda().float().unsqueeze(0), requires_grad=True)
        out = self.model(inp)
        loss = criterion(out, Variable(torch.Tensor([float(original_pred)]).cuda().long()))
        loss.backward()
        if targeted:
            inp.data = inp.data - ((eps / 255.0) * torch.sign(inp.grad.data))
        else:
            inp.data = inp.data + ((eps / 255.0) * torch.sign(inp.grad.data))
        inp.grad.data.zero_()
        adv_pred = np.argmax(self.model(inp).data.cpu().numpy())
        adv_class = classes[adv_pred].split(',')[0]
        adv = inp.data.cpu().numpy()[0]
        perturbation = (adv - img).transpose(1, 2, 0)
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = adv[..., ::-1]
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        return adv, perturbation, adv_class

    def predict_image(self, image_path):
        orig = cv2.imread(image_path)[..., ::-1]
        orig = cv2.resize(orig, (224, 224))
        img = orig.copy().astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img /= 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        inp = Variable(torch.from_numpy(img).cuda().float().unsqueeze(0), requires_grad=True)
        out = self.model(inp)
        original_pred = np.argmax(out.data.cpu().numpy())
        return classes[original_pred].split(',')[0]
