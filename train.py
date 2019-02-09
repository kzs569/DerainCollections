import os
import cv2
import argparse
import numpy as np

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets.rescan import TrainValDataset
from models.rescan import RESCAN
from losses.cal_ssim import SSIM
from utils.setting import *
from utils.logger import *


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class Session:
    def __init__(self, setting, logger):
        self.setting = setting
        self.model_dir = setting.checkpoint_dir
        ensure_dir(setting.checkpoint_dir)
        ##待修改
        self.net = RESCAN().cuda()
        self.crit = MSELoss().cuda()
        self.ssim = SSIM().cuda()

        self.step = 0
        self.save_steps = setting.save_steps
        self.num_workers = setting.num_workers
        self.batch_size = setting.batch_size
        self.dataloaders = {}

        self.opt = Adam(self.net.parameters(), lr=setting.lr)
        self.sche = MultiStepLR(self.opt, milestones=[15000, 17500], gamma=0.1)

    def get_dataloader(self, dataset, dtype):
        dataset = TrainValDataset(setting, dataset, dtype)
        if not dataset in self.dataloaders:
            self.dataloaders[dataset] = \
                DataLoader(dataset, batch_size=self.batch_size,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset])

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock': self.step,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.sche.last_epoch = self.step

    def inf_batch(self, name, batch):
        if name == 'train':
            self.net.zero_grad()

        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        R = O - B

        O_Rs = self.net(O)
        loss_list = [self.crit(O_R, R) for O_R in O_Rs]
        ssim_list = [self.ssim(O - O_R, O - R) for O_R in O_Rs]

        if name == 'train':
            sum(loss_list).backward()
            self.opt.step()

        losses = {
            'loss%d' % i: loss.item()
            for i, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)
        self.write(name, losses)

        return O - O_Rs[-1]

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        h, w = pred.shape[-2:]

        gen_num = (6, 2)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row + h, col: col + w] = tmp

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)


def run_train_val(setting, args, logger):
    sess = Session(setting, logger)

    if not args.retrain:
        lastest_model_path = os.path.join(setting.checkpoint_dir, 'lastest.ptb')
        sess.load_checkpoints(lastest_model_path)

    dt_train = sess.get_dataloader(args.dataset, 'train')
    dt_val = sess.get_dataloader(args.dataset, 'val')

    while sess.step < 20000:
        sess.sche.step()
        sess.net.train()

        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train')
            batch_t = next(dt_train)
        pred_t = sess.inf_batch('train', batch_t)

        if sess.step % 4 == 0:
            sess.net.eval()
            try:
                batch_v = next(dt_val)
            except StopIteration:
                dt_val = sess.get_dataloader('val')
                batch_v = next(dt_val)
            pred_v = sess.inf_batch('val', batch_v)

        if sess.step % int(sess.save_steps / 16) == 0:
            sess.save_checkpoints('latest')
        if sess.step % int(sess.save_steps / 2) == 0:
            sess.save_image('train', [batch_t['O'], pred_t, batch_t['B']])
            if sess.step % 4 == 0:
                sess.save_image('val', [batch_v['O'], pred_v, batch_v['B']])
            logger.info('save image as step_%d' % sess.step)
        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints('{}_step_{}'.format(setting.dataset_name, sess.step))
            logger.info('save model as step_%d' % sess.step)
        sess.step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-re', '--retrain', default=True)
    parser.add_argument('-m', '--model', required=True, choices=['rescan'])
    parser.add_argument('-d', '--dataset', required=True, choices=['rescan_rain', 'jorder_rain100l', 'jorder_rain100h'])
    args = parser.parse_args()

    setting = get_setting(args)

    torch.cuda.manual_seed_all(66)
    torch.manual_seed(66)
    torch.cuda.set_device(setting.device_id)

    logger = get_logger()

    run_train_val(setting, args, logger)
