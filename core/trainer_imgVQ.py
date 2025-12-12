import os
import glob
import logging
import importlib
import time

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.prefetch_dataloader import PrefetchDataLoader, CPUPrefetcher
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model.femasr_arch import FeMaSRNet
from model.propainter import Discriminator_2D

from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.loss import AdversarialLoss, PerceptualLoss, LPIPSLoss, make_r1_gp, SelfPerceptualLoss
from core.dataset_img import TrainDataset

from torchvision.utils import save_image
from utilss.metrics import PSNR, SSIM

torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.num_local_frames = config['train_data_loader']['num_local_frames']
        self.num_ref_frames = config['train_data_loader']['num_ref_frames']  #
        self.gp_coef = 0.001
        self.WM_stage = config['trainer']['WM_stage']
        if self.WM_stage:
            print('train wm matching...')

        # ---- metric ----
        self.psnr = PSNR()
        self.ssim = SSIM()

        # setup data set and data loader
        self.train_dataset = TrainDataset(config['train_data_loader'])
        print(self.train_dataset.__len__())

        if not os.path.isdir(os.path.join(self.config['save_dir'], 'Imgs')):
            os.makedirs(os.path.join(self.config['save_dir'], 'Imgs'), exist_ok=True)

        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        dataloader_args = dict(
            dataset=self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler,
            drop_last=True)

        self.train_loader = PrefetchDataLoader(self.train_args['num_prefetch_queue'], **dataloader_args)
        self.prefetcher = CPUPrefetcher(self.train_loader)

        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'], eps=0.1)  # soft label
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM()

        if self.config['losses']['perceptual_weight'] > 0:
            self.perc_loss = LPIPSLoss(use_input_norm=True, range_norm=True).to(self.config['device'])

        if not self.WM_stage:
            self.netG = FeMaSRNet(WM_stage=self.WM_stage).to(self.config['device'])
        else:
            load_path = self.train_args['pretrained']
            self.pre_net = FeMaSRNet(WM_stage=False).to(self.config['device'])
            self.pre_net.load_state_dict(torch.load(load_path, map_location=self.config['device']))

            self.netG = FeMaSRNet(WM_stage=self.WM_stage).to(self.config['device'])
            self.netG.load_state_dict(torch.load(load_path, map_location=self.config['device']), strict=False)
            frozen_module_keywords = ['quantize', 'decoder', 'after_quant_group', 'out_conv']
            if frozen_module_keywords is not None:
                for name, module in self.netG.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        if not self.config['model'].get('no_dis', False):
            self.netD = Discriminator_2D(
                in_channels=3,
                use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')

            self.netD = self.netD.to(self.config['device'])

        self.interp_mode = self.config['model']['interp_mode']
        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=True)

            self.netD = DDP(self.netD,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            if not self.config['model']['no_dis']:
                self.dis_writer = SummaryWriter(
                    os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    def metrics(self, pred, gt):
        psnr = self.psnr(pred.unsqueeze(0), gt.unsqueeze(0))
        ssim = self.ssim(pred.unsqueeze(0), gt.unsqueeze(0))
        return psnr, ssim

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        for name, param in self.netG.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
            else:
                print(f'Params {name} will not be optimized.')

        optim_params = [
            {
                'params': backbone_params
            },
        ]

        self.optimG = torch.optim.Adam(optim_params, lr=self.config['trainer']['glr'])

        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(self.netD.parameters(), lr=self.config['trainer']['dlr'],
                                           betas=(self.config['trainer']['beta1'],
                                                  self.config['trainer']['beta2']))

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(
                self.optimG,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
            if not self.config['model']['no_dis']:
                self.scheD = MultiStepRestartLR(
                    self.optimD,
                    milestones=scheduler_opt['milestones'],
                    gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'],
                eta_min=scheduler_opt['eta_min'])
            if not self.config['model']['no_dis']:
                self.scheD = CosineAnnealingRestartLR(
                    self.optimD,
                    periods=scheduler_opt['periods'],
                    restart_weights=scheduler_opt['restart_weights'],
                    eta_min=scheduler_opt['eta_min'])
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        if not self.config['model']['no_dis']:
            self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimG.param_groups[0]['lr']

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        n = self.train_args['log_freq']
        if writer is not None and self.iteration % n == 0:
            writer.add_scalar(name, self.summary[name] / n, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        # TODO: add resume name
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'),
                                'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1][4:] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path,
                                    f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path,
                                    f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,
                                    f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis'] and self.config['model']['load_d']:
                dataD = torch.load(dis_path, map_location=self.config['device'])
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])

            self.scheG.load_state_dict(data_opt['scheG'])
            if not self.config['model']['no_dis'] and self.config['model']['load_d']:
                self.optimD.load_state_dict(data_opt['optimD'])
                # self.scheD.load_state_dict(data_opt['scheD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']
        else:
            gen_path = self.config['trainer'].get('gen_path', None)
            dis_path = self.config['trainer'].get('dis_path', None)
            opt_path = self.config['trainer'].get('opt_path', None)
            if gen_path is not None:
                if self.config['global_rank'] == 0:
                    print(f'Loading Gen-Net from {gen_path}...')
                dataG = torch.load(gen_path, map_location=self.config['device'])
                self.netG.load_state_dict(dataG)

                if dis_path is not None and not self.config['model']['no_dis'] and self.config['model']['load_d']:
                    if self.config['global_rank'] == 0:
                        print(f'Loading Dis-Net from {dis_path}...')
                    dataD = torch.load(dis_path, map_location=self.config['device'])
                    self.netD.load_state_dict(dataD)

                if opt_path is not None:
                    data_opt = torch.load(opt_path, map_location=self.config['device'])
                    self.optimG.load_state_dict(data_opt['optimG'])
                    self.scheG.load_state_dict(data_opt['scheG'])
                    if not self.config['model']['no_dis'] and self.config['model']['load_d']:
                        self.optimD.load_state_dict(data_opt['optimD'])
                        self.scheD.load_state_dict(data_opt['scheD'])
            else:
                if self.config['global_rank'] == 0:
                    print('Warnning: There is no trained model found.'
                          'An initialized model will be used.')

    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'],
                                    f'dis_{it:06d}.pth')

            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                torch.save(netD.state_dict(), dis_path)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict(),
                        'scheG': self.scheG.state_dict(),
                        'scheD': self.scheD.state_dict()
                    }, opt_path)
            else:
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'scheG': self.scheG.state_dict()
                    }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
                   "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')

        self.netG.train()
        while True:
            self.epoch += 1
            self.prefetcher.reset()
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)
            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch"""
        device = self.config['device']
        train_data = self.prefetcher.next()
        while train_data is not None:
            # wm, clean, mask, gt
            wm, _, mask, gt = train_data
            wm, mask, gt = wm.to(device), mask.to(device).float(), gt.to(device)
            empty = torch.zeros(mask.shape).to(device)

            if self.WM_stage:
                with torch.no_grad():
                    _, _, _, gt_indices, _ = self.pre_net(torch.concat((gt, empty), dim=1))
                output, l_codebook, l_semantic, _, _ = self.netG(torch.concat((wm, mask), dim=1), gt_indices)
            else:
                output, l_codebook, l_semantic, _, _ = self.netG(torch.concat((gt, empty), dim=1))

            n, c, h, w = output.shape

            gen_loss = 0
            # optimize net_g
            if not self.config['model']['no_dis']:
                for p in self.netD.parameters():
                    p.requires_grad = False
            self.optimG.zero_grad()

            # codebook loss
            l_codebook *= self.config['losses']['codebook_weight']
            gen_loss += l_codebook.mean()

            # semantic cluster loss
            l_semantic *= self.config['losses']['semantic_weight']
            l_semantic = l_semantic.mean()
            gen_loss += l_semantic

            # l1 loss
            hole_loss = self.l1_loss(output * mask, gt * mask)
            hole_loss = hole_loss / (torch.mean(mask) + 1e-6)
            gen_loss += hole_loss
            valid_loss = self.l1_loss(output * (1 - mask), gt * (1 - mask))
            valid_loss = valid_loss / (torch.mean(1 - mask) + 1e-6)
            gen_loss += valid_loss
            l1_loss = hole_loss + valid_loss

            # SSIM
            l_ssim = (1 - self.ssim_loss(output, gt)) * 0.2
            gen_loss += l_ssim

            # perceptual loss
            perc_loss = self.perc_loss(output.view(-1, 3, h, w), gt.view(-1, 3, h, w))[0] * self.config['losses']['perceptual_weight']
            gen_loss += perc_loss

            # generator adversarial loss
            gen_clip = self.netD(output.view(n, 1, c, h, w))
            gan_loss = self.adversarial_loss(gen_clip, True, False)
            gan_loss = gan_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_loss
            self.add_summary(self.gen_writer, 'loss/gen', gen_loss.item())

            gen_loss.backward()
            self.optimG.step()

            # -------------------------------
            dis_loss = 0
            # optimize net_d
            for p in self.netD.parameters():
                p.requires_grad = True
            self.optimD.zero_grad()

            # discriminator adversarial loss
            real_clip = self.netD(gt.view(n, 1, c, h, w))
            fake_clip = self.netD(output.detach().view(n, 1, c, h, w))
            dis_real_loss = self.adversarial_loss(real_clip, True, True)
            dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            self.add_summary(self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
            self.add_summary(self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())

            # add grad_penalty
            grad_penalty = make_r1_gp(real_clip, gt.view(n, 1, c, h, w)) * self.gp_coef
            dis_loss += grad_penalty

            dis_loss.backward()
            self.optimD.step()

            self.update_learning_rate()

            # write image to tensorboard
            if self.iteration % 1000 == 0:
                # img to cpu
                t = 0
                wm_cpu = ((wm.view(n, -1, 3, h, w) + 1) / 2.0).cpu()
                gt_cpu = ((gt.view(n, -1, 3, h, w) + 1) / 2.0).cpu()
                output_cpu = ((output.view(n, -1, 3, h, w) + 1) / 2.0).cpu()
                img_results = torch.cat([wm_cpu[0][t], gt_cpu[0][t], output_cpu[0][t]], 1)
                img_results = torchvision.utils.make_grid(img_results, nrow=1, normalize=True)
                if self.gen_writer is not None:
                    self.gen_writer.add_image(f'img/img:{t}', img_results, self.iteration)

                # metric
                psnr, ssim = self.metrics(output_cpu[0][t], gt_cpu[0][t])
                print(f'fine: PSNR: {psnr:.2f}', " ", f'SSIM: {ssim:.4f}')

                # save image to logs
                if self.iteration % 2000 == 0:
                    save_image(img_results, os.path.join(self.config['save_dir'], 'Imgs', 'img {}.jpg'.format(self.iteration // 2000)))

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((f"l1: {l1_loss.item():.3f}; "
                                      f"ssim: {l_ssim.item():.3f}; "
                                      f"codebook: {l_codebook.item():.3f}; "
                                      f"semantic: {l_semantic.item():.3f}; "
                                      f"perc: {perc_loss.item():.3f}; "
                                      f"gan: {gan_loss.item():.3f}; "
                                      f"gen: {gen_loss.item():.3f}; "
                                      f"dis: {dis_loss.item():.3f}"))

                if self.iteration % self.train_args['log_freq'] == 0:
                    logging.info(f"[Iter {self.iteration}] "
                                 f"gen: {gen_loss.item():.3f}; "
                                 f"dis: {dis_loss.item():.3f}")

            self.iteration += 1
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break

            train_data = self.prefetcher.next()
