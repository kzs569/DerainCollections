import os
import numpy as np
import cv2
import torch

def clean_dir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def save_image(name, img_lists, path, step, batch_size):
    data, pred, label = img_lists
    data, pred, label = data * 255, np.clip(pred.data * 255, 0, 255), label * 255

    height, width = pred.shape[-2:]

    if batch_size > 16:
        gen_num = (6, 2)
    else:
        gen_num = (1, 1)

    img = np.zeros((gen_num[0] * height, gen_num[1] * 3 * width, 3))
    for i in range(gen_num[0]):
        row = i * height
        for j in range(gen_num[1]):
            idx = i * gen_num[1] + j
            tmp_list = [data[idx], pred[idx], label[idx]]
            for k in range(3):
                col = (j * 3 + k) * width
                tmp = np.transpose(tmp_list[k], (1, 2, 0))
                img[row: row + height, col: col + width] = tmp

    img_file = os.path.join(path, '%d_%s.jpg' % (step, name))
    cv2.imwrite(img_file, img)


def save_checkpoints(model, step, optim, model_dir, name='lastest'):
    ckp_path = os.path.join(model_dir, name)
    obj = {
        'net': model.state_dict(),
        'clock': step,
        'opt': optim.state_dict(),
    }
    torch.save(obj, ckp_path)


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
