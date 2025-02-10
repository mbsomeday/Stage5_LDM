import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config



from torch.utils.data import DataLoader, Dataset
from ldm.data.ww_dataset import my_dataset
from ldm.models.autoencoder import AutoencoderKL

from paths_dict import lca_dataset_dict, lca_autoencoder_ckpt_dict, local_dataset_dict, local_autoencoder_ckpt_dict, kaggle_dataset_dict, kaggle_autoencoder_ckpt_dict

cwd = os.getcwd()

if 'my_phd' in cwd:
    print(f'Run on local -- {cwd}')
    dataset_dict = local_dataset_dict
    autoencoder_ckpt_dict = local_autoencoder_ckpt_dict
elif 'veracruz' in cwd:
    print(f'Run on LCA -- {cwd}')
    dataset_dict = lca_dataset_dict
    autoencoder_ckpt_dict = lca_autoencoder_ckpt_dict
elif 'kaggle' in cwd:
    print(f'Run on kaggle -- {cwd}')
    dataset_dict = kaggle_dataset_dict
    autoencoder_ckpt_dict = kaggle_autoencoder_ckpt_dict
else:
    raise Exception('运行平台未知，需配置路径!')

ds_name = 'D3'
txt_name = 'val.txt'

autoencoder_ckpt = autoencoder_ckpt_dict[ds_name]
ds_dir = dataset_dict[ds_name]

latent_name = str(ds_name) + '_' + str(txt_name[:-4]) + '_LatentSpace.pt'

# '/kaggle/input/stage4-d1-ecpdaytime-7augs/Stage4_D1_ECPDaytime_7Augs/dataset_txt/augmentation_train.txt'

ddconfig = {
    'double_z': True,
    'z_channels': 4,
    'resolution': 256,
    'in_channels': 3,
    'out_ch': 3,
    'ch': 128,
    'ch_mult': [1, 2, 4, 4],  # num_down = len(ch_mult)-1
    'num_res_blocks': 2,
    'attn_resolutions': [],
    'dropout': 0.0
}

lossconfig = {
    'target': 'ldm.modules.losses.LPIPSWithDiscriminator',
    'params' : {
        'disc_start': 50001,
        'kl_weight': 0.000001,
        'disc_weight': 0.5,
    }
}
model = AutoencoderKL(ddconfig=ddconfig,
                      lossconfig=lossconfig,
                      embed_dim=4,
                      ckpt_path=autoencoder_ckpt
                      )
model.eval()
model = model.to('cuda')
for param in model.parameters():
    param.requires_grad = False

augTrain_data = my_dataset(ds_dir=ds_dir, txt_name=txt_name)
augTrain_loader = DataLoader(augTrain_data, batch_size=32)

print('Information:')
print('autoencoder_ckpt:',autoencoder_ckpt)
print('ds_dir:', ds_dir)
print('ds_name:', ds_name)
print('latent_name:', latent_name)
print('txt_name:', txt_name)

saved_tensor = None

from tqdm import tqdm

with torch.no_grad():
    for idx, image_dict in enumerate(tqdm(augTrain_loader)):
        image = image_dict['image']
        image = image.to('cuda')
        latent_space = model.encoder(image).detach()

        if saved_tensor is None:
            saved_tensor = latent_space
        else:
            saved_tensor = torch.cat((saved_tensor, latent_space), 0)

    torch.save(saved_tensor, latent_name)

print('读取保存的tensor')
load_torch = torch.load(latent_name)
print(load_torch.size())

















