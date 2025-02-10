import argparse, os
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader, Dataset
from ldm.data.ww_dataset import my_dataset
from ldm.models.autoencoder import AutoencoderKL

from paths_dict import lca_dataset_dict, lca_autoencoder_ckpt_dict, local_dataset_dict, local_autoencoder_ckpt_dict, kaggle_dataset_dict, kaggle_autoencoder_ckpt_dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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

def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ds_name', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    return args

args = get_parser()
# ds_name = args.ds_name
batch_size = args.batch_size

txt_list = ['augmentation_train.txt', 'val.txt', 'test.txt']

def get_model(autoencoder_ckpt):
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
        'params': {
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

    return model


def get_latentSpace(ds_name):
    autoencoder_ckpt = autoencoder_ckpt_dict[ds_name]
    ds_dir = dataset_dict[ds_name]


    model = get_model(autoencoder_ckpt)

    for txt_name in txt_list:
        latent_name = str(ds_name) + '_' + str(txt_name[:-4]) + '_LatentSpace.pt'
        latentspace_data = my_dataset(ds_dir=ds_dir, txt_name=txt_name)
        latentspace_loader = DataLoader(latentspace_data, batch_size=batch_size)

        print('*' * 80)
        print('Information:')
        print('autoencoder_ckpt:', autoencoder_ckpt)
        print(f'ds name: {ds_name} - ds dir: {ds_dir} - txt name: {txt_name}')
        print('save latent_name:', latent_name)

        saved_tensor = None

        with torch.no_grad():
            for idx, image_dict in enumerate(tqdm(latentspace_loader)):
                image = image_dict['image']
                image = image.to(DEVICE)
                latent_space = model.encoder(image).detach()

                if saved_tensor is None:
                    saved_tensor = latent_space
                else:
                    saved_tensor = torch.cat((saved_tensor, latent_space), 0)

            torch.save(saved_tensor, latent_name)

        # print('读取保存的tensor')
        # load_torch = torch.load(latent_name)
        # print(load_torch.size())


for ds_name in ['D1', 'D2', 'D3', 'D4']:
    get_latentSpace(ds_name)





















