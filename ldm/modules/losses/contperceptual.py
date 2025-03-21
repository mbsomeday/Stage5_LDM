import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_weights(model, weights):
    ckpt = torch.load(weights, weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    return model

class Att_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # ds_weights = r'D:\chrom_download\EfficientB0_dsCls-028-0.991572.pth'
        self.ds_model = models.efficientnet_b0(weights='IMAGENET1K_V1', progress=True)
        new_classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=4)
        )
        self.ds_model.classifier = new_classifier
        self.ds_model.eval()
        self.ds_model.to(DEVICE)

        self.feed_forward_features = None
        self.backward_features = None
        self.grad_layer = 'features'

        self._register_hooks(self.ds_model, grad_layer='features')

    def _register_hooks(self, model, grad_layer):
        '''
            注册钩子函数
        '''
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)
                print(f"Register forward hook and backward hook! Hooked layer: {grad_layer}")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def calc_cam(self, x):
        '''
            x是单张image
        '''
        logits = self.ds_model(x)
        pred = torch.argmax(logits, dim=1)

        self.ds_model.zero_grad()

        grad_yc = logits[0, pred]
        grad_yc.requires_grad = True
        print(f'grad_yc: {grad_yc} - {grad_yc.requires_grad}')
        grad_yc.backward()
        print(f'backward features after backward: {self.backward_features}')
        print('flag 11111')

        w = F.adaptive_avg_pool2d(self.backward_features, 1)    # shape: (batch_size, 1280, 1, 1)
        print(f'wwwww: {w.shape}')
        # print(f'w: {w.shape}')
        temp_w = w[0].unsqueeze(0)
        print('flag 222')

        temp_fl = self.feed_forward_features[0].unsqueeze(0)

        ac = F.conv2d(temp_fl, temp_w)
        print('flag 333')

        ac = F.relu(ac)

        Ac = F.interpolate(ac, (224, 224))
        print('flag 444')

        heatmap = Ac

        # 获取mask
        Ac_min = Ac.min()
        Ac_max = Ac.max()
        print(f'Attention map diff: {Ac_max - Ac_min}')

        mask = heatmap.detach().clone()
        mask.requires_grad = False
        mask[mask<Ac_max] = 0
        masked_image = x - x * mask
        print('flag 555')

        return heatmap, mask, masked_image

    def forward(self, x):
        print('forward of attenLoss!!')
        out = self.ds_model(x)
        print(f'forward feature after forward: {self.feed_forward_features.shape}')
        ds_cam, ds_mask, ds_masked_image = self.calc_cam(x)
        return ds_cam, ds_mask, ds_masked_image



class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        # ds model attention mao的损失函数
        self.attloss = Att_Loss()


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight



    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # 将attention map也用kl_weight作为权重
        masked_images = np.ones(shape=inputs.shape)
        for img_idx, image in enumerate(inputs):
            image = torch.unsqueeze(image, dim=0)
            print(f'image: {image.shape}')
            heatmap, mask, masked_image = self.attloss(image)
            print('flag after mask computing')
            masked_images[img_idx] = masked_image
        masked_images = torch.tensor(masked_images)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                # logits_fake = self.discriminator(reconstructions.contiguous()) + self.discriminator(masked_images)
                logits_fake = self.discriminator(reconstructions.contiguous())

            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss



            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),

                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

