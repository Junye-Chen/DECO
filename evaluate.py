import sys

sys.path.append(".")

import os
import cv2
import numpy as np
import argparse
from PIL import Image
import torch.nn.functional as F
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from torchvision import transforms
import imageio
from model.propainter import InpaintGenerator2
from model.CPN import Refinement
from model.image_model import ImageGeneratorVQ

from core.dataset import TestDataset
from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model, calc_rmse, cal_lpips
from lpips import LPIPS
from time import time
import warnings

warnings.filterwarnings("ignore")


# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length, ref_stride=10):
    ref_index = []
    for i in range(0, length, ref_stride):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


def main_worker(args):
    args.size = (args.width, args.height)
    w, h = args.size
    # set up datasets and data loader
    assert (args.dataset == 'davis') or args.dataset == 'youtube-vos', \
        f"{args.dataset} dataset is not supported"
    test_dataset = TestDataset(vars(args))
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    # set up models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # CPN
    cpn = Refinement(in_channels=4, out_channels=3, n_skips=0).to(device)
    cpn.load_state_dict(torch.load("weights/ibip.pth"))
    cpn.eval()
    for p0 in cpn.parameters():
        p0.requires_grad = False
    # ---------------------------------------

    # Image model
    imagemodel = ImageGeneratorVQ(args={}, train=False).to(device)
    imagemodel.load_state_dict(torch.load("weights/img.pth"), strict=False)
    imagemodel.eval()
    for p1 in imagemodel.parameters():
        p1.requires_grad = False

    # MSVT
    model = InpaintGenerator2(model_path=args.pretrain).to(device)
    model.eval()

    lpips = LPIPS(net='vgg').to(device).eval()

    # evaluate
    time_all = []
    print('Start evaluation ...')
    if args.task == 'video_completion':
        if args.dataset == 'youtube-vos':
            result_path = os.path.join(f'results_eval',
                                       f'{args.dataset}_rs_{args.ref_stride}_nl_{args.neighbor_length}_video_completion')
        else:
            result_path = os.path.join(f'results_eval',
                                       f'{args.dataset}_rs_{args.ref_stride}_nl_{args.neighbor_length}_video_completion')
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        eval_summary = open(os.path.join(result_path, f"{args.dataset}_metrics.txt"), "w")
        total_frame_psnr = []
        total_frame_ssim = []
        total_frame_lpips = []
        total_frame_rmse = []

        output_i3d_activations = []
        real_i3d_activations = []
        i3d_model = init_i3d_model('weights/i3d_rgb_imagenet.pt')
    else:
        result_path = os.path.join(f'results_eval',
                                   f'{args.dataset}_rs_{args.ref_stride}_nl_{args.neighbor_length}_object_removal')
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for index, items in enumerate(test_loader):
        torch.cuda.empty_cache()

        _, frames, masks, video_name, frames_PIL = items

        video_name = video_name[0]
        video_length = frames.size(1)
        print('Processing:', video_name, f'len:{video_length}')

        b, t, c, h, w = frames.size()
        wm, masks = frames.to(device), masks.to(device)

        torch.cuda.synchronize()
        time_start = time()
        comp_frames = [None] * video_length

        neighbor_stride = args.neighbor_length // 2
        if frames.size(1) > neighbor_stride:
            for f in range(0, video_length, neighbor_stride):
                neighbor_ids = [
                    i for i in range(max(0, f - neighbor_stride),
                                     min(video_length, f + neighbor_stride + 1))
                ]

                ref_ids = get_ref_index(neighbor_ids, video_length, args.ref_stride)
                selected_wms = wm[:, neighbor_ids + ref_ids, :, :, :]
                selected_masks = masks[:, neighbor_ids + ref_ids, :, :, :]
                t0 = len(neighbor_ids + ref_ids)
                l_t = len(neighbor_ids)

                with torch.no_grad():
                    # step 1: pred clean component
                    clean, wm_feat = cpn(selected_wms.view(b * t0, c, h, w), selected_masks.view(b * t0, 1, h, w))
                    clean = clean.view(b, t0, c, h, w)
                    clean_local = clean[:, :l_t, ...]
                    wm_feat_local = wm_feat.view(b, t0, 256, h // 8, w // 8)[:, :l_t, ...]

                    # step 2: compute image feat
                    local_masks = selected_masks[:, :l_t, ...].contiguous()
                    _, _, _, out_feat, _, _ = imagemodel(clean_local.reshape(b * l_t, c, h, w), local_masks.reshape(b * l_t, 1, h, w))

                    # step 3: inject feat to restore clean image
                    pred_img = model(clean, selected_masks, l_t, wm_feat_local.reshape(b * l_t, 256, h // 8, w // 8), out_feat[0])
                    pred_img = pred_img.view(b, -1, c, h, w)

                    comp_imgs = selected_wms[:, :l_t, ...] * (1. - local_masks) + pred_img * local_masks  # 这里按照 mask 得到修复好的图片

                pred_imgs = comp_imgs.view(-1, 3, h, w)
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    if comp_frames[idx] is None:
                        comp_frames[idx] = pred_imgs[i]
        else:
            with torch.no_grad():
                # step 1: pred clean component
                clean, wm_feat = cpn(wm.view(b * t, c, h, w), masks.view(b * t, 1, h, w))
                # step 2: compute image feat
                _, _, _, out_feat, _, _ = imagemodel(clean, masks.view(b * t, 1, h, w))
                clean = clean.view(b, t, c, h, w)

                # step 3: inject feat to restore clean image
                pred_imgs = model(clean, masks, t, wm_feat, out_feat[0])
                pred_imgs = pred_imgs.view(b, -1, c, h, w)

                comp_imgs = wm * (1. - masks) + pred_imgs * masks

                pred_imgs = comp_imgs.view(-1, 3, h, w)
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

                for i in range(video_length):
                    if comp_frames[i] is None:
                        comp_frames[i] = pred_imgs[i]

        ori_frames = frames_PIL
        ori_frames = [
            ori_frames[i].squeeze().cpu().numpy() for i in range(video_length)
        ]
        masks = masks.view(b * t, 1, h, w)
        masks = [masks[i].cpu().numpy() for i in range(masks.shape[0])]

        wms = [(wm[:, i].squeeze().cpu().numpy() + 1) / 2 * 255 for i in range(wm.view(b * t, c, h, w).shape[0])]

        torch.cuda.synchronize()
        time_i = time() - time_start
        time_i = time_i * 1.0 / video_length
        time_all.append(time_i)

        if args.task == 'video_completion':
            # calculate metrics
            cur_video_psnr = []
            cur_video_ssim = []
            cur_video_lpips = []
            cur_video_rmse = []
            comp_PIL = []  # to calculate VFID
            frames_PIL = []
            for ori, comp, mask in zip(ori_frames, comp_frames, masks):
                # print('ori', ori.shape, comp.shape)
                psnr, ssim = calc_psnr_and_ssim(ori, comp)
                mlpips = lpips(transforms.ToTensor()(ori).cuda(), transforms.ToTensor()(comp).cuda()).mean().item()
                rmse = calc_rmse(ori, comp, mask)

                cur_video_psnr.append(psnr)
                cur_video_ssim.append(ssim)
                cur_video_lpips.append(mlpips)
                cur_video_rmse.append(rmse)

                total_frame_psnr.append(psnr)
                total_frame_ssim.append(ssim)
                total_frame_lpips.append(mlpips)
                total_frame_rmse.append(rmse)

                frames_PIL.append(Image.fromarray(ori.astype(np.uint8)))
                comp_PIL.append(Image.fromarray(comp.astype(np.uint8)))

            # saving i3d activations
            frames_i3d, comp_i3d = calculate_i3d_activations(frames_PIL, comp_PIL, i3d_model, device=device)
            real_i3d_activations.append(frames_i3d)
            output_i3d_activations.append(comp_i3d)

            cur_psnr = sum(cur_video_psnr) / len(cur_video_psnr)
            cur_ssim = sum(cur_video_ssim) / len(cur_video_ssim)
            cur_lpips = sum(cur_video_lpips) / len(cur_video_lpips)
            cur_rmse = sum(cur_video_rmse) / len(cur_video_rmse)

            avg_psnr = sum(total_frame_psnr) / len(total_frame_psnr)
            avg_ssim = sum(total_frame_ssim) / len(total_frame_ssim)
            avg_lpips = sum(total_frame_lpips) / len(total_frame_lpips)
            avg_rmse = sum(total_frame_rmse) / len(total_frame_rmse)

            avg_time = sum(time_all) / len(time_all)
            print(
                f'[{index + 1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM/LPIPS/RMSEw: {cur_psnr:.4f}/{cur_ssim:.4f}/{cur_lpips:.4f}'
                f'/{cur_rmse:.4f} | Avg PSNR/SSIM/LPIPS/RMSEw: {avg_psnr:.4f}/{avg_ssim:.4f}/{avg_lpips:.4f}/{avg_rmse:.4f} | Time: {avg_time:.4f}'
            )
            eval_summary.write(
                f'[{index + 1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM/LPIPS/RMSEw: {cur_psnr:.4f}/{cur_ssim:.4f}/{cur_lpips:.4f}'
                f'/{cur_rmse:.4f} | Avg PSNR/SSIM/LPIPS/RMSEw: {avg_psnr:.4f}/{avg_ssim:.4f}/{avg_lpips:.4f}/{avg_rmse:.4f} | Time: {avg_time:.4f}\n'
            )
        else:
            avg_time = sum(time_all) / len(time_all)
            print(
                f'[{index + 1:3}/{len(test_loader)}] Name: {str(video_name):25} | Time: {avg_time:.4f}'
            )

        if args.save_video and args.name == 'davis':
            os.makedirs(os.path.join(result_path, video_name), exist_ok=True)
            comp_frames = [f.astype(np.uint8) for f in comp_frames]
            wms_frames = [f.astype(np.uint8).transpose(1, 2, 0) for f in wms]
            imageio.mimwrite(os.path.join(result_path, video_name, 'output.mp4'), comp_frames, codec='h264', fps=25, quality=10)
            imageio.mimwrite(os.path.join(result_path, video_name, 'watermark.mp4'), wms_frames, codec='h264', fps=25, quality=10)

        elif args.save_video and args.name == 'youtube-vos':
            os.makedirs(os.path.join(result_path, video_name), exist_ok=True)
            comp_frames = [f.astype(np.uint8) for f in comp_frames]
            wms_frames = [f.astype(np.uint8).transpose(1, 2, 0) for f in wms]
            imageio.mimwrite(os.path.join(result_path, video_name, 'output.mp4'), comp_frames, codec='h264', fps=25, quality=10)
            imageio.mimwrite(os.path.join(result_path, video_name, 'watermark.mp4'), wms_frames, codec='h264', fps=25, quality=10)

        # saving images for evaluating warping errors
        if args.save_results:
            save_frame_path = os.path.join(result_path, video_name)
            if not os.path.exists(save_frame_path):
                os.makedirs(save_frame_path, exist_ok=False)

            for i, frame in enumerate(comp_frames):
                cv2.imwrite(
                    os.path.join(save_frame_path,
                                 str(i).zfill(5) + '.png'),
                    cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))

        del wm, ori_frames, comp_frames, masks, frames_PIL, wms
        torch.cuda.empty_cache()

    if args.task == 'video_completion':
        avg_frame_psnr = sum(total_frame_psnr) / len(total_frame_psnr)
        avg_frame_ssim = sum(total_frame_ssim) / len(total_frame_ssim)
        avg_frame_lpips = sum(total_frame_lpips) / len(total_frame_lpips)
        avg_frame_rmse = sum(total_frame_rmse) / len(total_frame_rmse)

        fid_score = calculate_vfid(real_i3d_activations, output_i3d_activations)

        print('Finish evaluation... Average Frame PSNR/SSIM/LPIPS/RMSE/VFID: '
              f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{avg_frame_lpips:.4f}/{avg_frame_rmse:.4f}/{fid_score:.3f} | Time: {avg_time:.4f}')
        eval_summary.write(
            'Finish evaluation... Average Frame PSNR/SSIM/LPIPS/RMSE/VFID: '
            f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{avg_frame_lpips:.4f}/{avg_frame_rmse:.4f}/{fid_score:.3f} | Time: {avg_time:.4f}')
        eval_summary.close()
    else:
        print('Finish evaluation... Time: {avg_time:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--width', type=int, default=432)
    parser.add_argument("--ref_stride", type=int, default=10)
    parser.add_argument("--neighbor_length", type=int, default=50)
    parser.add_argument('--task', default='video_completion', choices=['object_removal', 'video_completion'])
    parser.add_argument('--dataset', choices=['davis', 'youtube-vos'], type=str)
    parser.add_argument('--video_root', default='dataset_root', type=str)
    parser.add_argument('--mask_root', default='mask_root', type=str)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num', default=10, type=int)
    parser.add_argument('--name', choices=['davis', 'youtube-vos'], type=str)
    parser.add_argument('--pretrain', '-pw', required=True, type=str)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.cuda.current_device())
    args = parser.parse_args()
    main_worker(args)
