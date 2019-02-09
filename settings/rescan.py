import os
import logging

channel = 24
stage_num = 4
depth = 7  # >=3
use_se = True
uint = 'GRU'  # ['Conv', 'RNN', 'GRU', 'LSTM']
frame = 'Full'  # ['Add', 'Full']

aug_data = False  # Set as False for fair comparison

batch_size = 64
patch_size = 64
lr = 5e-3

dataset_name = 'rain100l'
data_dir = os.path.join('../datasets/', dataset_name)
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'

log_level = 'info'
model_path = os.path.join(model_dir, dataset_name + '_latest')
save_steps = 1000

num_workers = 8
num_GPU = 1
device_id = 0

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
