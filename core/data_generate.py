# coding = gbk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision as tv
from PIL import Image
import random
import os
import torchvision.transforms.functional as TF
import math
import glob


def img_Resize(img, num=None, target_size=None):
    # img = Image.open(path)
    width, height = img.size
    if num is not None:
        target_size = (int(width * num), int(height * num))
    assert target_size is not None
    img = img.resize(target_size, Image.LANCZOS)

    return img


def ScaleRotateTranslate(image, angle, center=None, new_center=None, new_size=None, scale=None):
    if center is None:
        return image.rotate(angle)
    if new_size is None:
        new_size = image.size
    angle = -angle / 180.0 * math.pi
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = scale
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(new_size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)


class FlexRotateResizedCrop:
    def __init__(self, scale=(0.5, 2), ratio=(0.8, 1.2), size=(256, 256), area_max=0.25):

        self.size = size
        self.area_max = area_max
        self.scale = scale
        self.ratio = ratio

    def __call__(self, x):

        w, h = x.size
        h_valid = h
        w_valid = w

        if (h_valid / self.size[1]) > (w_valid / self.size[0]):
            if (self.scale[1] * h_valid) > self.size[1]:
                scale_max = min(self.size[1] * self.area_max / w_valid, self.scale[1])
            else:
                scale_max = self.scale[1]
            scale_min = scale_max / self.scale[1] * self.scale[0]

            scale_factor_w = random.uniform(scale_min, scale_max)
            scale_factor_h = max(min(scale_factor_w * random.uniform(self.ratio[0], self.ratio[1]), scale_max),
                                 scale_min)

            h_target = int(h_valid * scale_factor_h)
            w_target = int(w_valid * scale_factor_w)

            if h_target > self.size[1]:
                h_target = self.size[1]
                h_valid = int(h_target / scale_factor_h)
        else:
            if (self.scale[1] * w_valid) > self.size[0]:
                scale_max = min(self.size[0] * self.area_max / h_valid, self.scale[1])
            else:
                scale_max = self.scale[1]
            scale_min = scale_max / self.scale[1] * self.scale[0]
            scale_factor_h = random.uniform(scale_min, scale_max)
            scale_factor_w = max(min(scale_factor_h * random.uniform(self.ratio[0], self.ratio[1]), scale_max),
                                 scale_min)
            # print('scale_factor',scale_factor_w,scale_factor_h,scale_min,scale_max )
            h_target = int(h_valid * scale_factor_h)
            w_target = int(w_valid * scale_factor_w)
            if w_target > self.size[0]:
                w_target = self.size[0]
                w_valid = int(w_target / scale_factor_w)
        # print('size',x.size,(w_valid,h_valid),(w_target,h_target))


        center_x = w_valid // 2 + random.randint(0, w - w_valid)
        center_y = h_valid // 2 + random.randint(0, h - h_valid)
        theta = random.randint(0, 360)
        theta_n = theta / 180.0 * math.pi
        # nw_valid = int(w_valid*scale_factor_w*abs(math.cos(theta_n))+h_valid*scale_factor_h*abs(math.sin(theta_n)))
        # nh_valid = int(w_valid*scale_factor_w*abs(math.sin(theta_n))+h_valid*scale_factor_h*abs(math.cos(theta_n)))
        # print(w_valid*scale_factor_w,h_valid*scale_factor_h)

        dy = self.size[0] * abs(math.sin(theta_n)) + self.size[1] * abs(math.cos(theta_n))
        dx = self.size[0] * abs(math.cos(theta_n)) + self.size[1] * abs(math.sin(theta_n))
        rx = max((dx - w_valid * scale_factor_w) // 2, 0)
        ry = max((dy - h_valid * scale_factor_h) // 2, 0)
        # space_x = max(self.size[0]//2-nw_valid//2,0)
        # space_y = max(self.size[1]//2-nh_valid//2,0) random.randint(-rx,rx) random.randint(-ry,ry)
        # space_x = max(self.size[0]//2-nw_valid//2,0)
        shifty = random.uniform(-0.9, 0.9)
        shiftx = random.uniform(-0.9, 0.9)
        newcenter_x = self.size[0] // 2 + shifty * ry * math.sin(theta_n) + shiftx * rx * math.cos(theta_n)
        # random.randint(0, self.size[0]-w_target)+w_target//2
        newcenter_y = self.size[1] // 2 + shifty * ry * math.cos(theta_n) - shiftx * rx * math.sin(
            theta_n)  # -0 * space_y#random.randint(-space_y,space_y)#random.randint(0, self.size[1]-h_target)+
        # print(self.size,(dx,dy),(rx,ry),(nw_valid,nh_valid),(newcenter_x,newcenter_y))
        res = ScaleRotateTranslate(x, theta, center=(center_x, center_y), new_center=(newcenter_x, newcenter_y),
                                   new_size=self.size, scale=(scale_factor_w, scale_factor_h))  # random.randint(0,360)
        return res


class RotateResizedCrop:
    def __init__(self, scale=(0.5, 2), ratio=(0.8, 1.2), size=(256, 256)):

        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, x):
        w, h = x.size
        scale_factor_w = random.uniform(self.scale[0], self.scale[1])
        scale_factor_h = max(min(scale_factor_w * random.uniform(self.ratio[0], self.ratio[1]), self.scale[0]),
                             self.scale[1])
        roi_w = int(self.size[0] / scale_factor_w)
        roi_h = int(self.size[1] / scale_factor_h)
        if roi_w > w or roi_h > h:
            scale_factor = min(w / roi_w, h / roi_h)
            roi_w = int(roi_w * scale_factor)
            roi_h = int(roi_h * scale_factor)
        # print(w,roi_w)
        shift_x = random.randint(0, w - roi_w)
        shift_y = random.randint(0, h - roi_h)
        res = ScaleRotateTranslate(x, random.randint(0, 360), center=(shift_x + roi_w // 2, shift_y + roi_h // 2),
                                   new_center=(self.size[0] // 2, self.size[1] // 2), new_size=self.size,
                                   scale=(scale_factor_w, scale_factor_h))
        return res


class RandAlphaAdjuster:
    def __init__(self, gamma=(0.9, 1.1), gain=(0.5, 1)):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, x):
        gamma = random.uniform(self.gamma[0], self.gamma[1])
        gain = random.uniform(self.gain[0], self.gain[1])
        return TF.adjust_gamma(x, gamma, gain=gain)



def degrade_scale(im_ori, im_wm_l, im_wm_h, scale_rng=[0.1, 1]):
    size0 = im_ori.size
    scale_factor = random.uniform(scale_rng[0], scale_rng[1])
    im_ori = img_Resize(img_Resize(im_ori, scale_factor), target_size=size0)
    im_wm_l = img_Resize(img_Resize(im_wm_l, scale_factor), target_size=size0)
    im_wm_h = img_Resize(img_Resize(im_wm_h, scale_factor), target_size=size0)
    return np.asarray(im_ori), np.asarray(im_wm_l), np.asarray(im_wm_h)


def degrade_compression(im_ori, im_wm_l, im_wm_h, rate_rng=[70, 100]):
    size0 = im_ori.size
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(rate_rng[0], rate_rng[1])]
    im_ori = cv2.imdecode(cv2.imencode('.jpg', np.asarray(im_ori), encode_param)[1], 1)
    im_wm_l = cv2.imdecode(cv2.imencode('.jpg', np.asarray(im_wm_l), encode_param)[1], 1)
    im_wm_h = cv2.imdecode(cv2.imencode('.jpg', np.asarray(im_wm_h), encode_param)[1], 1)
    return im_ori, im_wm_l, im_wm_h


def watermarking(ori_Img_tmp, wm_ImgC_tmp, wm_ImgA_tmp, scale_rng=[0.5, 1], rate_rng=[70, 100]):

    # ori_Img_tmp = cropped_img.copy()
    # wm_ImgC_tmp = cropped_wmRGB.copy()
    # wm_ImgA_tmp = cropped_wmA.copy()
    assert ori_Img_tmp.size[0] == wm_ImgC_tmp.size[0]
    assert ori_Img_tmp.size[1] == wm_ImgC_tmp.size[1]


    im_wm_l = ori_Img_tmp.copy()
    im_wm_h = ori_Img_tmp.copy()
    im_wm_l.paste(wm_ImgC_tmp, (0, 0), wm_ImgA_tmp[0])
    im_wm_h.paste(wm_ImgC_tmp, (0, 0), wm_ImgA_tmp[1])

    # im_clean = ori_Img_tmp.copy()
    # im_clean.paste(Image.new('L', ori_Img_tmp.size), (0, 0), wm_ImgA_tmp[0])

    degrade_type = np.random.randint(0, 3)
    if degrade_type == 0:
        arr_im_gt = np.asarray(ori_Img_tmp)
        arr_im_wm_l = np.asarray(im_wm_l)
        arr_im_wm_h = np.asarray(im_wm_h)
        # arr_im_clean = np.asarray(im_clean)
    elif degrade_type == 1:
        # arr_im_gt, arr_im_wm, arr_im_clean = degrade_scale(ori_Img_tmp, im_wm, im_clean, scale_rng=scale_rng)
        arr_im_gt, arr_im_wm_l, arr_im_wm_h = degrade_scale(ori_Img_tmp, im_wm_l, im_wm_h, scale_rng=scale_rng)
    else:
        # arr_im_gt, arr_im_wm, arr_im_clean = degrade_compression(ori_Img_tmp, im_wm, im_clean, rate_rng=rate_rng)
        arr_im_gt, arr_im_wm_l, arr_im_wm_h = degrade_compression(ori_Img_tmp, im_wm_l, im_wm_h, rate_rng=rate_rng)

    # arr_alpha = np.asarray(wm_ImgA_tmp)
    # mask = (np.sum(abs(arr_im_wm - arr_im_gt), 2) > 3).astype(np.uint8)
    # mask0 = (np.sum(abs(arr_im_wm_h - arr_im_gt), 2) > 3).astype(np.uint8)
    wm_mask = np.asarray(wm_ImgA_tmp[0])
    mask = np.zeros_like(wm_ImgA_tmp[0])
    mask[wm_mask > 0] = 1
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.dilate((np.sum(abs(arr_im_wm_h - arr_im_gt), 2) > 3).astype(np.uint8), kernel, iterations=1)
    # return arr_im_wm, arr_im_gt, arr_im_clean, mask, arr_alpha
    return arr_im_wm_l, arr_im_wm_h, arr_im_gt, mask


def watermarking_large(ori_Img_tmp, wm_ImgC_tmp, wm_ImgA_tmp, scale_rng=[0.5, 1]):

    scale_factor = random.uniform(scale_rng[0], scale_rng[1])
    ori_Img1 = ori_Img_tmp
    w, h = ori_Img1.size


    res_Img1 = ori_Img1.copy()
    res_Img2 = ori_Img1.copy()
    mark_Img1 = img_Resize(wm_ImgC_tmp, scale_factor)  # wm_ImgC_tmp.copy()
    # alpha1 = img_Resize(wm_ImgA_tmp, scale_factor)  # wm_ImgA_tmp.copy()
    alpha1_l = img_Resize(wm_ImgA_tmp[0], scale_factor)  # wm_ImgA_tmp.copy()
    alpha1_h = img_Resize(wm_ImgA_tmp[1], scale_factor)  # wm_ImgA_tmp.copy()

    mark_w, mark_h = mark_Img1.size
    proportion = mark_w / mark_h
    mark_w = w
    mark_h = int(mark_w / proportion)
    mark_Img1 = img_Resize(mark_Img1, target_size=(mark_w, mark_h))
    # alpha1 = img_Resize(alpha1, target_size=(mark_w, mark_h))
    alpha1_l = img_Resize(alpha1_l, target_size=(mark_w, mark_h))
    alpha1_h = img_Resize(alpha1_h, target_size=(mark_w, mark_h))

    x = 0
    y = random.randint(int(h / 10), int(9 * h / 10 - mark_h))
    # ori_w, ori_h = ori_Img1.size
    # mark_w, mark_h = mark_Img1.size
    res_Img1.paste(mark_Img1, (x, y), alpha1_l)
    res_Img2.paste(mark_Img1, (x, y), alpha1_h)

    # res_Img2 = ori_Img1.copy()

    # res_Img2.paste(Image.new('L', mark_Img1.size), (x, y), alpha1)
    # 好像也不对。。。
    # new_alpha = Image.fromarray(255 - np.array(alpha1))
    # res_Img2.paste(Image.new('L', mark_Img1.size), (x, y), new_alpha)

    # background = Image.fromarray(np.zeros((h, w, 3), np.uint8)).convert("L")
    # background.paste(alpha1, (x, y), alpha1)
    # alpha1.save('4_alpha.png')
    # ori_Img1 = img_Resize(ori_Img1, target_size=ori_Img_tmp.size)
    # res_Img1 = img_Resize(res_Img1, target_size=ori_Img_tmp.size)
    # res_Img2 = img_Resize(res_Img2, target_size=ori_Img_tmp.size)
    # alpha1 = img_Resize(alpha1, target_size=ori_Img_tmp.size)
    arr_im_wm_l = np.asarray(res_Img1)
    arr_im_wm_h = np.asarray(res_Img2)
    # arr_ori_comp = np.asarray(res_Img2)
    # arr_alpha1 = np.asarray(background)
    ori_Img1 = np.asarray(ori_Img1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate((np.sum(abs(arr_im_wm_l - ori_Img1), 2) > 3).astype(np.uint8), kernel, iterations=1)
    # wm_mask = np.asarray(wm_ImgA_tmp[0])
    # print(np.unique(wm_mask))
    # mask = np.zeros_like(wm_ImgA_tmp[0])
    # mask[wm_mask > 0] = 1
    return arr_im_wm_l, arr_im_wm_h, ori_Img1, mask



