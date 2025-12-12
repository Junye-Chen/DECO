
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

from utilss.file_client import FileClient
from utilss.img_util import imfrombytes
from core.utils import (create_random_shape_with_random_motion, Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip, GroupRandomHorizontalFlowFlip)
from core.utils import CreateWatermarkMask, draw_text


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, args: dict):
        self.args = args
        self.video_root = args['video_root']
        self.flow_root = args['flow_root']
        self.mask_root = args['wm_root']
        self.num_local_frames = args['num_local_frames']  # 10
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])
        self.is_train = args['is_train']
        if self.is_train:
            print('regularization training...')

        if self.load_flow:
            assert os.path.exists(self.flow_root)

        json_path = os.path.join('./datasets', args['name'], 'train.json')
        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict.keys()))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted([f for f in os.listdir(os.path.join(self.video_root, v)) if not f.endswith('.zip')])
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list

        self.video_names = list(self.video_dict.keys())  # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

        self.create_wm = CreateWatermarkMask(self.mask_root, self.h, self.w)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))  # [0, 1, ... , length-1]
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]

        wms, alphas, all_masks, weight = self.create_wm.create_wm_mask_with_random_motion(self.video_dict[video_name], training=self.is_train)

        selected_index = self._sample_index(self.video_dict[video_name], self.num_local_frames, self.num_ref_frames)

        # read video frames
        frames = []
        wmimgs = []
        masks = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img.copy())  # src Image

            img.paste(wms[idx], (0, 0), alphas[idx])
            wmimgs.append(img)
            masks.append(all_masks[idx])  # Image

            # load flow
            if len(frames) <= self.num_local_frames - 1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx + 1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if len(frames) == self.num_local_frames:
                if random.random() < 0.5:
                    frames.reverse()
                    wmimgs.reverse()
                    masks.reverse()
                    if self.load_flow:
                        flows_f.reverse()
                        flows_b.reverse()
                        flows_ = flows_f
                        flows_f = flows_b
                        flows_b = flows_

        if self.load_flow:
            frames, flows_f, flows_b, masks = GroupRandomHorizontalFlowFlip()(frames, flows_f, flows_b, masks)
        else:
            frames, wmimgs, masks = GroupRandomHorizontalFlip()(frames, wmimgs, masks)

        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0  # img [-1,1]
        wmimg_tensors = self._to_tensors(wmimgs) * 2.0 - 1.0  # img [-1,1]
        mask_tensors = self._to_tensors(masks)  # mask [0,1]

        # img [-1,1] mask [0,1]
        if self.load_flow:
            return frame_tensors, wmimg_tensors, mask_tensors, flows_f, flows_b, video_name
        else:
            return frame_tensors, wmimg_tensors, mask_tensors, 'None', 'None', video_name, weight

    def __len__(self):
        return len(self.video_names)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = (432, 240)

        self.video_root = args['video_root']
        self.mask_root = "./videowm/"
        self.dataset_name = args['name']

        self.video_dict = {}
        self.frame_dict = {}

        json_path = os.path.join('./datasets', self.dataset_name, 'test.json')
        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict.keys()))

        for v in self.video_names:
            frame_list = sorted([f for f in os.listdir(os.path.join(self.video_root, v)) if not f.endswith('.zip')])
            # frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            self.video_dict[v] = v_len
            self.frame_dict[v] = frame_list

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        selected_index = list(range(self.video_dict[video_name]))

        wms_path = list(glob.glob(os.path.join(self.mask_root, self.dataset_name, video_name, 'wm', '*')))
        alphas_path = list(glob.glob(os.path.join(self.mask_root, self.dataset_name, video_name, 'alpha', '*')))
        masks_path = list(glob.glob(os.path.join(self.mask_root, self.dataset_name, video_name, 'mask', '*')))

        # read video frames
        frames = []
        wms = []
        masks = []

        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            frame_path = os.path.join(self.video_root, video_name, frame_list[idx])

            img_bytes = self.file_client.get(frame_path, 'input')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            wm = Image.open(wms_path[idx]).convert('RGB')
            alpha = Image.open(alphas_path[idx]).convert('L')
            mask = Image.open(masks_path[idx]).convert('L')

            frames.append(img.copy())

            img.paste(wm, (0, 0), alpha)
            wms.append(img)

            masks.append(mask)

        # normalize, to tensors
        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        wm_tensors = self._to_tensors(wms) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)

        return frame_tensors, wm_tensors, mask_tensors, video_name, frames_PIL

