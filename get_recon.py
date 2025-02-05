'''
    获取 reconstruction images
'''
import torch.cuda
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import argparse

from ldm.data.ww_dataset import my_dataset
from ldm.models.autoencoder import AutoencoderKL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


dataset_dict = {
    'D1': r'/kaggle/input/stage4-d1-ecpdaytime-7augs/Stage4_D1_ECPDaytime_7Augs',
    'D2': r'/kaggle/input/stage4-d2-citypersons-7augs/Stage4_D2_CityPersons_7Augs',
    'D3': r'/kaggle/input/stage4-d3-ecpnight-7augs',
    'D4': r'/kaggle/input/stage4-d4-7augs'
}

autoencoder_ckpt_dict = {
    'D1': r'/kaggle/input/stage5-weights-ldm-d1/D1_epo26_00894.ckpt',
    'D2': r'/kaggle/input/stage5-weights-ldm-d2/D2_epo59_01239.ckpt',
    'D3': r'/kaggle/input/stage5-weights-ldm-d3/D3_epo49_01236.ckpt',
    'D4': r'/kaggle/input/stage5-weights-ldm-d4/D4_epo34_01236.ckpt'
}

parser = argparse.ArgumentParser()
parser.add_argument( "--ds_name", type=str)
parser.add_argument( "--txt_name", type=str)
args = parser.parse_args()


ds_name = args.ds_name
ds_dir = dataset_dict[ds_name]
txt_name = args.txt_name
autoencoder_ckpt = autoencoder_ckpt_dict[ds_name]
recon_name = ds_name + '_' + txt_name[:-4] +'_ReconstructedImage.pt'

print('*' * 50)
print(f'Dataset: {ds_dir} - {txt_name}\npath:{ds_dir}')
print(f'latent_name: {recon_name}')
print('*' * 50)

cur_data = my_dataset(ds_dir=ds_dir, txt_name=txt_name)
cur_loader = DataLoader(cur_data, batch_size=3)

# ddconfig = {
#     'double_z': True,
#     'z_channels': 4,
#     'resolution': 256,
#     'in_channels': 3,
#     'out_ch': 3,
#     'ch': 128,
#     'ch_mult': [1, 2, 4, 4],  # num_down = len(ch_mult)-1
#     'num_res_blocks': 2,
#     'attn_resolutions': [],
#     'dropout': 0.0
# }
#
# lossconfig = {
#     'target': 'ldm.modules.losses.LPIPSWithDiscriminator',
#     'params' : {
#         'disc_start': 50001,
#         'kl_weight': 0.000001,
#         'disc_weight': 0.5,
#     }
# }
# model = AutoencoderKL(ddconfig=ddconfig,
#                       lossconfig=lossconfig,
#                       embed_dim=4,
#                       ckpt_path=autoencoder_ckpt
#                       )
# model.eval()
# model = model.to(DEVICE)
# for param in model.parameters():
#     param.requires_grad = False
#
# saved_tensor = None
name_list = []

for idx, image_dict in enumerate(tqdm(cur_loader)):
    image = image_dict['image']
    image = image.to(DEVICE)

    image_names = image_dict['image_name']
    name_list.append(image_names)

    if idx == 3:
        break

print(name_list)
print(len(name_list))
#     dec, posterior = model(image)
#
#     if saved_tensor is None:
#         saved_tensor = dec
#     else:
#         saved_tensor = torch.cat((saved_tensor, dec), 0)
#
#
# torch.save(saved_tensor, recon_name)
#
# print('读取保存的tensor')
# load_torch = torch.load(recon_name)
# print(load_torch.size())



















