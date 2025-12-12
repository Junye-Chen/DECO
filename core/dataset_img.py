
import os
import json
import random
import glob
import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from core.utils import (create_random_shape_with_random_motion, Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip, GroupRandomHorizontalFlowFlip)
from core.utils import CreateWatermarkMask, draw_text


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args

        ILAWpath = args['ILAW']
        CLWDpath = args['CLWD']
        self.wmimg_path = sorted(list(glob.glob(os.path.join(ILAWpath, 'watermarked', '*.png'), recursive=True))) + \
                          sorted(list(glob.glob(os.path.join(CLWDpath, 'Watermarked_image', '*.jpg'), recursive=True)))
        self.mask_path = sorted(list(glob.glob(os.path.join(ILAWpath, 'alpha', '*.png'), recursive=True))) + \
                         sorted(list(glob.glob(os.path.join(CLWDpath, 'Alpha', '*.png'), recursive=True)))
        self.gt_path = sorted(list(glob.glob(os.path.join(ILAWpath, 'gt', '*.png'), recursive=True))) + \
                       sorted(list(glob.glob(os.path.join(CLWDpath, 'Watermark_free_image', '*.jpg'), recursive=True)))

        self.size = self.w, self.h = (args['w'], args['h'])

        self.totensor = transforms.ToTensor()

        # ! cannot do norm!
        self.transform_train = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize([self.h, self.w], antialias=None)
        ])

    def __getitem__(self, index):
        wm, alpha, gt = self.wmimg_path[index], self.mask_path[index], self.gt_path[index]
        wm = Image.open(wm).convert('RGB')
        gt = Image.open(gt).convert('RGB')
        alpha = Image.open(alpha).convert('L')

        if random.random() < 0.1 and np.max(alpha) > 0:
            alpha = Image.fromarray(np.uint8(np.asarray(alpha) / np.max(alpha) * 255 * random.uniform(0.98, 1.)))

        imageWidth, imageHeight = wm.size
        zero = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        clean = gt.copy()
        clean.paste(zero, (0, 0), alpha)

        mask = np.array(alpha)
        mask[mask > 1] = 255
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        wm = self.totensor(wm)
        gt = self.totensor(gt)
        mask = self.totensor(mask)
        clean = self.totensor(clean)

        img = torch.concat((wm, mask, gt, clean), dim=0)
        img = self.transform_train(img)
        wm = img[:3, ...] * 2.0 - 1.0  # img [-1,1]
        mask = img[3, ...].unsqueeze(dim=0)  # mask [0,1]
        gt = img[4:7, ...] * 2.0 - 1.0
        clean = img[7:, ...] * 2.0 - 1.0

        return wm, clean, mask, gt

    def __len__(self):
        return len(self.wmimg_path)

