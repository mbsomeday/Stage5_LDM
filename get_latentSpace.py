import torch

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
print(torch.cuda.is_available())
model.to('cuda')
torch.save(model.state_dict(), r'test.pth')

# model.eval()
# x = torch.rand((1, 3, 224, 224))
#
# pos = model.encode(x)
# z = pos.sample().detach()
# print(z.size())
# print(z.requires_grad)

# import random
#
# random.seed(2)
# a = [1,2,3,4,5,6,7,8,9]
# b = [9,8,7,6,5]
# random.shuffle(a)
# random.shuffle(b)
#
# print(a)
# print(b)












