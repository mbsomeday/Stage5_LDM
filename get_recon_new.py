import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import argparse
from torchvision import utils as vutils


from ldm.data.ww_dataset import my_dataset
from ldm.models.autoencoder import AutoencoderKL
from paths_dict import lca_dataset_dict, lca_autoencoder_ckpt_dict


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # 复制一份
    input_tensor = input_tensor.clone()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


def my_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Create {path}')

def create_dirs(base_path, ds_name):
    name_list = ['nonPedestrian', 'pedestrian']
    if ds_name == 'D4':
        time_list = ['daytime', 'dusk', 'night']
    else:
        time_list = ['']
    aug_dir = os.path.join(base_path, 'augmentation_train')
    my_make_dir(aug_dir)
    for n in name_list:
        for t in time_list:
            my_make_dir(os.path.join(base_path, n, t))


# configs
ds_name = 'D1'
txt_name = 'val.txt'
save_base = r'/veracruz/home/j/jwang/scripts/Stage5_LDM'

# 根据configuration拿到的变量
ds_dir = lca_dataset_dict[ds_name]
save_dataset_dir = os.path.join(save_base, ds_name)

# 打印输出信息
print('*' * 60)
print('Confige Information:')
print(f'dataset: {ds_name} -- dataset path: {ds_dir} -- {txt_name}')
print(f'Save to {save_dataset_dir}')
print('*' * 60)


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
                      ckpt_path=lca_autoencoder_ckpt_dict[ds_name]
                      )
model.eval()
for param in model.parameters():
    param.requires_grad = False

# dataset
cur_data = my_dataset(ds_dir=ds_dir, txt_name=txt_name)
cur_loader = DataLoader(cur_data, batch_size=1, shuffle=False)


for idx, image_dict in enumerate(tqdm(cur_loader)):
    image = image_dict['image']
    image_name = image_dict['image_name'][0]
    image_path = image_dict['file_path'][0]
    label = int(image_dict['label'])

    ds_dir_name = ds_dir.split('\\')[-1]
    part_path = image_path.split(ds_dir_name)[-1][1:]
    image_save_path = os.path.join(save_dataset_dir, part_path)
    dec, posterior = model(image)
    save_image_tensor(dec[0], image_save_path)



























