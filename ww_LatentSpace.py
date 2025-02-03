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


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))




now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

# add cwd for convenience and to make classes in this file available when
# running as `python main.py`
# (in particular `main.DataModuleFromConfig`)
sys.path.append(os.getcwd())

parser = get_parser()
parser = Trainer.add_argparse_args(parser)

opt, unknown = parser.parse_known_args()
if opt.name and opt.resume:
    raise ValueError(
        "-n/--name and -r/--resume cannot be specified both."
        "If you want to resume training in a new log folder, "
        "use -n/--name in combination with --resume_from_checkpoint"
    )
if opt.resume:
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        paths = opt.resume.split("/")
        # idx = len(paths)-paths[::-1].index("logs")+1
        # logdir = "/".join(paths[:idx])
        logdir = "/".join(paths[:-2])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), opt.resume
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

    opt.resume_from_checkpoint = ckpt
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    opt.base = base_configs + opt.base
    _tmp = logdir.split("/")
    nowname = _tmp[-1]
else:
    if opt.name:
        name = "_" + opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = ""
    nowname = now + name + opt.postfix
    logdir = os.path.join(opt.logdir, nowname)

ckptdir = os.path.join(logdir, "checkpoints")
cfgdir = os.path.join(logdir, "configs")
seed_everything(opt.seed)

configs = [OmegaConf.load(cfg) for cfg in opt.base]
cli = OmegaConf.from_dotlist(unknown)
config = OmegaConf.merge(*configs, cli)
lightning_config = config.pop("lightning", OmegaConf.create())
# merge trainer cli with config
trainer_config = lightning_config.get("trainer", OmegaConf.create())
# default to ddp
trainer_config["accelerator"] = "ddp"
for k in nondefault_trainer_args(opt):
    trainer_config[k] = getattr(opt, k)
if not "gpus" in trainer_config:
    del trainer_config["accelerator"]
    cpu = True
else:
    gpuinfo = trainer_config["gpus"]
    print(f"Running on GPUs {gpuinfo}")
    cpu = False
trainer_opt = argparse.Namespace(**trainer_config)
lightning_config.trainer = trainer_config

# model
# model = instantiate_from_config(config.model)

trainer_kwargs = dict()

# data
data = instantiate_from_config(config.data)
# NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
# calling these ourselves should not be necessary but it is.
# lightning still takes care of proper multiprocessing though

data.prepare_data()
data.setup()
print("#### Data #####")
for k in data.datasets:
    print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
# print('Now is training')
# trainer.fit(model, data)

from torch.utils.data import DataLoader, Dataset
from ldm.data.ww_dataset import my_dataset
from ldm.models.autoencoder import AutoencoderKL

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
                      )

test_data = my_dataset(ds_dir=r'/kaggle/input/stage4-d4-7augs', txt_name='test.txt')
test_loader = DataLoader(test_data, batch_size=3)

saved_tensor = None


for idx, image_dict in enumerate(test_loader):
    image = image_dict['image']
    latent_space = model.encode(image).detach()
    print('laten size:', latent_space.size())

    if saved_tensor is None:
        saved_tensor = latent_space
    else:
        saved_tensor = torch.cat((saved_tensor, latent_space), 0)
        print('拼接的：', saved_tensor.size())
    if idx == 3:
        torch.save(saved_tensor, 'test_save_tensor.pt')
        break


print('读取保存的tensor')
load_torch = torch.load('test_save_tensor.pt')
print(load_torch.size())



















