import os
import random
import shutil

import cv2
import argparse
import numpy as np
import yaml

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import get_loader
from models import get_model
from losses.cal_ssim import SSIM
from utils.visualize import Visualizer
from utils.logger import get_logger
from utils import clean_dir, ensure_dir, save_image, save_checkpoints
from optimizers import get_optimizer
from losses import get_critical

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg, logger, vis):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        split=cfg["data"]["train_split"],
        patch_size=cfg['data']['patch_size'],
        augmentation=cfg['data']['aug_data']
    )

    v_loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
    )

    trainloader = DataLoader(
        t_loader,
        batch_size=cfg["batch_size"],
        num_workers=cfg["n_workers"],
        shuffle=True,
    )

    valloader = DataLoader(
        v_loader, batch_size=cfg["batch_size"], num_workers=cfg["n_workers"]
    )

    # Setup model, optimizer and loss function
    model_cls = get_model(cfg['model'])
    model = model_cls(cfg).to(device)

    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    scheduler = MultiStepLR(optimizer, milestones=[15000, 17500], gamma=0.1)

    crit = get_critical(cfg['critical'])().to(device)
    ssim = SSIM().to(device)

    step = 0

    if cfg['resume'] is not None:
        pass

    while step < cfg['max_iters']:
        scheduler.step()
        model.train()

        if cfg['model'] == 'rescan':
            O, B, prediciton = inference_rescan(model=model, optimizer=optimizer, dataloader=trainloader,
                                                critical=crit, ssim=ssim,
                                                step=step, vis=vis)
        if cfg['model'] == 'did_mdn':
            O, B, prediciton, label = inference_didmdn(model=model, optimizer=optimizer, dataloader=trainloader,
                                                       critical=crit, ssim=ssim,
                                                       step=step, vis=vis)

        if step % 10 == 0:
            model.eval()
            if cfg['model'] == 'rescan':
                O, B, prediciton_v = inference_rescan(model=model, optimizer=optimizer, dataloader=valloader,
                                                      critical=crit, ssim=ssim,
                                                      step=step, vis=vis)
            if cfg['model'] == 'did_mdn':
                O, B, prediciton, label = inference_didmdn(model=model, optimizer=optimizer,
                                                           dataloader=valloader,
                                                           critical=crit, ssim=ssim,
                                                           step=step, vis=vis)

        if step % int(cfg['save_steps'] / 16) == 0:
            save_checkpoints(model, step, optimizer, cfg['checkpoint_dir'], 'latest')
        if step % int(cfg['save_steps'] / 2) == 0:
            save_image('train', [O.cpu(), prediciton.cpu(), B.cpu()], cfg['checkpoint_dir'], step, cfg['batch_size'])
            if step % 10 == 0:
                save_image('val', [O.cpu(), prediciton.cpu(), B.cpu()], cfg['checkpoint_dir'], step, cfg['batch_size'])
            logger.info('save image as step_%d' % step)
        if step % cfg['save_steps'] == 0:
            save_checkpoints(model=model,
                             step=step,
                             optim=optimizer,
                             model_dir=cfg['checkpoint_dir'],
                             name='{}_step_{}'.format(cfg['model'] + cfg['data']['dataset'], step))
            logger.info('save model as step_%d' % step)
        step += 1


def inference_rescan(model, optimizer, dataloader, critical, ssim, step, vis):
    try:
        O, B = next(iter(dataloader))
    except StopIteration:
        O, B = next(iter(dataloader))

    if step % 10 != 0:
        model.zero_grad()

    O, B = O.to(device), B.to(device)
    O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
    R = O - B

    O_Rs = model(O)
    loss_list = [critical(O_R, R) for O_R in O_Rs]
    ssim_list = [ssim(O - O_R, O - R) for O_R in O_Rs]

    if step % 10 != 0:
        sum(loss_list).backward()
        optimizer.step()

    losses = {
        'loss%d' % i: loss.item()
        for i, loss in enumerate(loss_list)
    }
    ssimes = {
        'ssim%d' % i: ssim.item()
        for i, ssim in enumerate(ssim_list)
    }
    losses.update(ssimes)

    losses['lr'] = optimizer.param_groups[0]['lr']
    losses['step'] = step
    outputs = [
        "{}:{:.4g}".format(k, v)
        for k, v in losses.items()
    ]
    logger.info('train' + '--' + ' '.join(outputs))

    prediction = O - O_Rs[-1]

    if vis is not None:
        for k, v in losses.items():
            vis.plot(k, v)
        vis.images(np.clip((prediction.detach() * 255).data.cpu().numpy(), 0, 255)[:64], win='pred')
        vis.images(O.data.cpu().numpy()[:64], win='input')
        vis.images(B.data.cpu().numpy()[:64], win='groundtruth')

    return O, B, prediction


def inference_didmdn(model, optimizer, dataloader, critical, ssim, step, vis):
    try:
        O, B, label = next(iter(dataloader))
    except StopIteration:
        O, B, label = next(iter(dataloader))

    if step % 10 != 0:
        model.zero_grad()

    O, B, label = O.to(device), B.to(device), label.to(device)
    O, B, label = Variable(O, requires_grad=False), Variable(B, requires_grad=False), Variable(label.float(),
                                                                                               requires_grad=False)
    R = O - B
    O_R, prediction = model(O, label)

    loss = critical(B, prediction)
    ssims = ssim(O - O_R, O - R)
    losses = {
        'loss : ': loss.item()
    }
    losses.update({
        'ssim : ': ssims.item()
    })

    if step % 10 != 0:
        loss.backward()
        optimizer.step()

    outputs = [
        "{}:{:.4g}".format(k, v)
        for k, v in losses.items()
    ]
    logger.info('train' + '--' + ' '.join(outputs))

    if vis is not None:
        for k, v in losses.items():
            vis.plot(k, v)
        vis.images(np.clip((prediction.detach().data * 255).cpu().numpy(), 0, 255), win='pred')
        vis.images(O.data.cpu().numpy(), win='input')
        vis.images(B.data.cpu().numpy(), win='groundtruth')
        vis.images(R.data.cpu().numpy(), win='raindrop')

    return O, B, prediction, label


def train_gan(cfg, logger, vis):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        split=cfg["data"]["train_split"],
        patch_size=cfg['data']['patch_size'],
        augmentation=cfg['data']['aug_data']
    )

    train_loader = DataLoader(
        t_loader,
        batch_size=cfg["batch_size"],
        num_workers=cfg["n_workers"],
        shuffle=True,
    )

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    ndf = cfg['ndf']
    ngf = cfg['ngf']
    nc = 3

    netD_cls = get_model(cfg['netd'])
    netG_cls = get_model(cfg['netg'])

    netD = netD_cls(nc, cfg['output_nc'], ndf).to(device)
    netG = netG_cls(cfg['input_nc'], cfg['output_nc'], ngf).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)
    logger.info(netD)
    logger.info(netG)

    ###########   LOSS & OPTIMIZER   ##########
    criterion = torch.nn.BCELoss()
    criterionL1 = torch.nn.L1Loss()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=cfg['optimizer']['lr'],
                                  betas=(cfg['optimizer']['beta1'], 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=cfg['optimizer']['lr'],
                                  betas=(cfg['optimizer']['beta1'], 0.999))

    ###########   GLOBAL VARIABLES   ###########
    input_nc = cfg['input_nc']
    output_nc = cfg['output_nc']
    fineSize = cfg['data']['patch_size']

    real_A = Variable(torch.FloatTensor(cfg['batch_size'], input_nc, fineSize, fineSize), requires_grad=False).to(
        device)
    real_B = Variable(torch.FloatTensor(cfg['batch_size'], output_nc, fineSize, fineSize), requires_grad=False).to(
        device)
    label = Variable(torch.FloatTensor(cfg['batch_size']), requires_grad=False).to(device)

    real_label = 1
    fake_label = 0

    ########### Training   ###########
    netD.train()
    netG.train()
    for epoch in range(1, cfg['max_iters'] + 1):
        for i, image in enumerate(train_loader):
            ########### fDx ###########
            netD.zero_grad()
            if cfg['direction'] == 'OtoB':
                imgA = image[1]
                imgB = image[0]
            else:
                imgA = image[0]
                imgB = image[1]

            # train with real data
            real_A.data.resize_(imgA.size()).copy_(imgA)
            real_B.data.resize_(imgB.size()).copy_(imgB)
            real_AB = torch.cat((real_A, real_B), 1)

            output = netD(real_AB)
            label.data.resize_(output.size())
            label.data.fill_(real_label)
            errD_real = criterion(output, label)
            errD_real.backward()

            # train with fake
            fake_B = netG(real_A)
            label.data.fill_(fake_label)

            fake_AB = torch.cat((real_A, fake_B), 1)
            output = netD(fake_AB.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()

            errD = (errD_fake + errD_real) / 2
            optimizerD.step()

            ########### fGx ###########
            netG.zero_grad()
            label.data.fill_(real_label)
            output = netD(fake_AB)
            errGAN = criterion(output, label)
            errL1 = criterionL1(fake_B, real_B)
            errG = errGAN + cfg['lamb'] * errL1

            errG.backward()
            optimizerG.step()

            ########### Logging ##########
            if i % 50 == 0:
                logger.info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_L1: %.4f'
                            % (epoch, cfg['max_iters'], i, len(train_loader),
                               errD.item(), errGAN.item(), errL1.item()))

            if cfg['vis']['use'] and (i % 50 == 0):
                fake_B = netG(real_A)
                vis.images(real_A.data.cpu().numpy(), win='real_A')
                vis.images(fake_B.detach().cpu().numpy(), win='fake_B')
                vis.images(real_B.data.cpu().numpy(), win='real_B')
                vis.plot('error_d', errD.item())
                vis.plot('error_g', errGAN.item())
                vis.plot('error_L1', errL1.item())

        if epoch % 20 == 0:
            save_image(
                name='train',
                img_lists=[real_A.data.cpu(), fake_B.data.cpu(), real_B.data.cpu()],
                path='%s/fake_samples_epoch_%03d.png' % (cfg['checkpoint_dir'], epoch),
                step=epoch,
                batch_size=cfg['batch_size']
            )
            save_checkpoints(model=netG,
                             step=epoch,
                             optim=optimizerG,
                             model_dir=cfg['checkpoint_dir'],
                             name='{}_step_{}'.format(cfg['netg'] + cfg['data']['dataset'], epoch))
            save_checkpoints(model=netD,
                             step=epoch,
                             optim=optimizerD,
                             model_dir=cfg['checkpoint_dir'],
                             name='{}_step_{}'.format(cfg['netd'] + cfg['data']['dataset'], epoch))


if __name__ == '__main__':

    # Load the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./configs/pix2pix_jorder_rain100l.yaml')
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    if cfg['resume']:
        clean_dir(cfg['checkpoint_dir'])

    print(cfg)

    # Setup the log
    run_id = random.randint(1, 100000)
    logdir = os.path.join(cfg['checkpoint_dir'], os.path.basename(args.config)[:-4] + str(run_id))
    ensure_dir(logdir)
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)
    logger = get_logger(logdir)
    logger.info("Let the games begin")

    # Setup the Visualizer
    if cfg['vis']['use']:
        vis = Visualizer(cfg['vis']['env'])
    else:
        vis = None

    torch.multiprocessing.freeze_support()
    if cfg['model'] in ['rescan', 'did_mdn']:
        train(cfg, logger, vis)
    elif cfg['model'] in ['pix2pix']:
        train_gan(cfg, logger, vis)

