import os
import argparse
import yaml
import random
import shutil
import numpy as np

import torch
from torch.autograd import Variable
from datasets import get_loader
from models import get_model
from losses import get_critical
from optimizers import get_optimizer
from losses.cal_ssim import SSIM
from utils import ensure_dir
from utils.visualize import Visualizer
from utils.logger import get_logger
from torch.utils.data import DataLoader

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoints(model, optim, model_dir, name='lastest'):
    ckp_path = os.path.join(model_dir, name)
    try:
        print('Load checkpoint %s' % ckp_path)
        obj = torch.load(ckp_path)
    except FileNotFoundError:
        print('No checkpoint %s!!' % ckp_path)
        return False, None
    model.load_state_dict(obj['net'])
    optim.load_state_dict(obj['opt'])
    step = obj['clock']
    return True, step


def test(cfg, logger, vis):
    torch.cuda.manual_seed_all(66)
    torch.manual_seed(66)

    # Setup model, optimizer and loss function
    model_cls = get_model(cfg['model'])
    model = model_cls(cfg).to(device)

    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    crit = get_critical(cfg['critical'])().to(device)
    ssim = SSIM().to(device)

    model.eval()
    _, step = load_checkpoints(model, optimizer, cfg['checkpoint_dir'], name='latest')

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    test_loader = data_loader(
        data_path,
        split=cfg["data"]["test_split"],
        patch_size=cfg['data']['patch_size'],
        augmentation=cfg['data']['aug_data']
    )

    testloader = DataLoader(
        test_loader,
        batch_size=cfg["batch_size"],
        num_workers=cfg["n_workers"],
        shuffle=True,
    )

    all_num = 0
    all_losses = {}
    for i, batch in enumerate(testloader):

        O, B = batch
        O, B = Variable(O.to(device), requires_grad=False), Variable(B.to(device), requires_grad=False)
        R = O - B

        with torch.no_grad():
            O_Rs = model(O)
        loss_list = [crit(O_R, R) for O_R in O_Rs]
        ssim_list = [ssim(O - O_R, O - R) for O_R in O_Rs]

        losses = {
            'loss%d' % i: loss.item()
            for i, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)

        prediction = O - O_Rs[-1]

        batch_size = O.size(0)

        all_num += batch_size
        for key, val in losses.items():
            if i == 0:
                all_losses[key] = 0.
            all_losses[key] += val * batch_size
            logger.info('batch %d mse %s: %f' % (i, key, val))

        if vis is not None:
            for k, v in losses.items():
                vis.plot(k, v)
            vis.images(np.clip((prediction.detach().data * 255).cpu().numpy(), 0, 255), win='pred')
            vis.images(O.data.cpu().numpy(), win='input')
            vis.images(B.data.cpu().numpy(), win='groundtruth')

    for key, val in all_losses.items():
        logger.info('total mse %s: %f' % (key, val / all_num))


if __name__ == '__main__':
    # Load the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./configs/rescan_rescan_rain.yaml')
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    print(cfg)

    # Setup the log
    run_id = random.randint(1, 100000)
    logdir = os.path.join(cfg['checkpoint_dir'], os.path.basename(args.config)[:-4] + str(run_id))
    ensure_dir(logdir)
    print("LOGDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)
    logger = get_logger(logdir)
    logger.info("-------------------------------------Let the games begin----------------------------------")

    # Setup the Visualizer
    if cfg['vis']['use']:
        vis = Visualizer(cfg['vis']['env'] + '_test')
    else:
        vis = None

    test(cfg, logger, vis)
