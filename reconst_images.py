'''
    使用训练好的 auto encoder 重建图片
'''

import os, torch, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import utils as vutils

from ldm.data.ww_dataset import my_dataset
from ldm.models.autoencoder import AutoencoderKL
from paths_dict import lca_dataset_dict, lca_autoencoder_ckpt_dict, local_dataset_dict, local_autoencoder_ckpt_dict, kaggle_dataset_dict, kaggle_autoencoder_ckpt_dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--ds_name', type=str)
    parse.add_argument('--txt_name', type=str)
    parse.add_argument('--batch_size', default=32, type=int)
    parse.add_argument('--AE_ds', default=None, type=str)

    args = parse.parse_args()
    return args

# 通过传入参数进行配置
args = parse_args()

ds_name = args.ds_name
txt_name = args.txt_name
batch_size = args.batch_size

cwd = os.getcwd()

print('*' * 60)
dataset_dict = ''
autoencoder_ckpt_dict = ''
if 'veracruz' in cwd:
    print(f'Run on LCA -- {cwd}')
    dataset_dict = lca_dataset_dict
    autoencoder_ckpt_dict = lca_autoencoder_ckpt_dict
elif 'my_phd' in cwd:
    print(f'Run on local -- {cwd}')
    dataset_dict = local_dataset_dict
    autoencoder_ckpt_dict = local_autoencoder_ckpt_dict
elif 'kaggle' in cwd:
    print(f'Run on kaggle -- {cwd}')
    dataset_dict = kaggle_dataset_dict
    autoencoder_ckpt_dict = kaggle_autoencoder_ckpt_dict
else:
    raise Exception('运行平台未知，需配置路径!')

AE_ds = args.AE_ds
if AE_ds is not None and AE_ds != ds_name:
    AE_ckpt = autoencoder_ckpt_dict[AE_ds]
    print(f'当前是输入数据集与训练AE的数据集不一致的情况\n模型权重为:{AE_ckpt}')
else:
    AE_ckpt = autoencoder_ckpt_dict[ds_name]
    print(f'Auto Encoder的模型权重为:{AE_ckpt}')

ds_path = dataset_dict[ds_name]
print(f'Input dataset: {ds_name} - txt: {txt_name} \nDataset Path:{ds_path}')
print(f'Batch size: {batch_size}')

# model
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
                      ckpt_path=AE_ckpt
                      )
model.eval()
model = model.to(DEVICE)
for param in model.parameters():
    param.requires_grad = False

# dataset
cur_data = my_dataset(ds_dir=ds_path, txt_name=txt_name)
cur_loader = DataLoader(cur_data, batch_size=batch_size, shuffle=False)

tensorDict_save_dir = os.path.join(cwd, ds_name, txt_name[:-4])
if not os.path.exists(tensorDict_save_dir):
    os.makedirs(tensorDict_save_dir)

rec_dict = {}
print('Num of images:', len(cur_data))
print('Num of batch:', len(cur_loader))
print(fr'Reconstructed images will save to {tensorDict_save_dir}')
print('*' * 60)

with torch.no_grad():
    for idx, image_dict in enumerate(tqdm(cur_loader)):
        images = image_dict['image'].to(DEVICE)
        dec, posterior = model(images)
        image_paths = image_dict['file_path']

        for img_idx, img_path in enumerate(image_paths):
            image_save_dir_part = img_path.split(ds_path.split(os.sep)[-1])[-1][1:]
            rec_dict.update({image_save_dir_part: dec[img_idx]})

        if idx != 0 and ((idx+1) % int(2000/batch_size) == 0 or (idx+1) == len(cur_loader)):
            tensorDict_save_name = ds_name + '_' + str(txt_name[:-4]) + '_' + str(idx + 1) + '.pt'
            tensorDict_save_path = os.path.join(cwd, ds_name, txt_name[:-4], tensorDict_save_name)
            torch.save(rec_dict, tensorDict_save_path)
            print(f'.pt file save to {tensorDict_save_path}')
            # 重置字典
            rec_dict = {}





















