"""
wild mixture of
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import logging
import os
import random
from contextlib import contextmanager
from functools import partial

import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm

mainlogger = logging.getLogger("mainlogger")

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision.utils import make_grid
from lvdm.modules.lora import inject_trainable_lora,tune_lora_scale,load_lora_state_dict,collapse_lora,monkeypatch_remove_lora

from lvdm.modules.utils import (
    disabled_train,
    default,
    exists,
    extract_into_tensor,
    noise_like,
)
from lvdm.distributions import DiagonalGaussianDistribution, normal_kl
from lvdm.ema import LitEma
from lvdm.samplers.ddim import DDIMSampler
from lvdm.models.utils_diffusion import make_beta_schedule
from utils.common_utils import instantiate_from_config

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}

import peft
            

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor=None,
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        beta_dpo=5000.0, # dpo config 
        dupbeta=1, # dpo config 
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        mainlogger.info(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = unet_config.params.temporal_length
        self.image_size = image_size  # try conv?
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        # count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        # add loss dpo
        if self.loss_type == "dpo":
            self.ref_model = DiffusionWrapper(unet_config, conditioning_key)
            # freeze all
            for param in self.ref_model.diffusion_model.parameters():
                param.requires_grad = False
            self.beta_dpo = beta_dpo

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    mainlogger.info(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        mainlogger.info(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            mainlogger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            mainlogger.info(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            return_intermediates=return_intermediates,
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def dpo_loss(self, target, pred):
        model_losses = (pred - target).pow(2).mean(dim=[1, 2, 3])
        model_losses_w, model_losses_l = model_losses.chunk(2)
        raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
        model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)
        with torch.no_grad():  # Get the reference policy (unet) prediction
            ref_pred = self.ref_pred
            ref_losses = (ref_pred - target).pow(2).mean(dim=[1, 2, 3])
            ref_losses_w, ref_losses_l = ref_losses.chunk(2)
            ref_diff = ref_losses_w - ref_losses_l
            raw_ref_loss = ref_losses.mean()
        scale_term = -0.5 * self.beta_dpo
        inside_term = scale_term * (model_diff - ref_diff)
        # print()
        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        # log implicit acc for 
        self.log(
            "train/implicit_acc",implicit_acc,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        factor = self.dupfactor
        loss = ( -1 * factor * F.logsigmoid(inside_term)).mean()
        self.log(      "train/factor",float(factor.clone().mean().detach().cpu()),
            prog_bar=False,logger=True,
            on_step=True,on_epoch=False,
        )
        return loss

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        elif self.loss_type == "dpo":
            loss = self.dpo_loss(target, pred)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)
        # since we can't pass x_noisy and t to get loss
        # we have to use self.ref_pred to pass parameter
        if self.loss_type == "dpo":
            self.ref_pred = self.ref_model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported"
            )

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        """
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        """
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        print("shared step:",x.shape)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )
        torch.cuda.empty_cache()
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        _, loss_dict_no_ema = self.shared_step(batch, random_uncond=False)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, random_uncond=False)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(
            loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        print("log_images")
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="caption",
        cond_stage_trainable=False,
        cond_stage_forward=None,
        conditioning_key=None,
        uncond_prob=0.2,
        uncond_type="empty_seq",
        scale_factor=1.0,
        scale_by_std=False,
        # added for LVDM
        encoder_type="2d",
        frame_cond=None,
        only_model=False,
        use_scale=False,
        scale_a=1,
        scale_b=0.3,
        mid_step=400,
        fix_scale_bug=False,
        logdir=None,
        empty_params_only=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, "crossattn")
        lora_args = kwargs.pop("lora_args", []) # lora args should be poped bofere super.init
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.empty_params_only = empty_params_only

        # scale factor
        self.use_scale = use_scale
        if self.use_scale:
            self.scale_a = scale_a
            self.scale_b = scale_b
            if fix_scale_bug:
                scale_step = self.num_timesteps - mid_step
            else:  # bug
                scale_step = self.num_timesteps

            scale_arr1 = np.linspace(scale_a, scale_b, mid_step)
            scale_arr2 = np.full(scale_step, scale_b)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            scale_arr_prev = np.append(scale_a, scale_arr[:-1])
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer("scale_arr", to_torch(scale_arr))

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert encoder_type in ["2d", "3d"]
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert uncond_type in ["zero_embed", "empty_seq"]
        self.uncond_type = uncond_type

        ## future frame prediction
        self.frame_cond = frame_cond
        if self.frame_cond:
            # frame_len = self.model.diffusion_model.temporal_length
            frame_len = self.temporal_length
            cond_mask = torch.zeros(frame_len, dtype=torch.float32)
            cond_mask[: self.frame_cond] = 1.0
            ## b,c,t,h,w
            self.cond_mask = cond_mask[None, None, :, None, None]
            mainlogger.info(
                "---training for %d-frame conditoning T2V" % (self.frame_cond)
            )
        else:
            self.cond_mask = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True
                
        self.logdir = logdir
        # lora related args
        self.lora_implementation = getattr(lora_args, "lora_implementation", "peft") # default as peft 
        self.inject_unet = getattr(lora_args, "inject_unet", False)
        self.inject_clip = getattr(lora_args, "inject_clip", False)
        self.inject_unet_key_word = getattr(lora_args, "inject_unet_key_word", None)
        self.inject_clip_key_word = getattr(lora_args, "inject_clip_key_word", None)
        self.lora_rank = getattr(lora_args, "lora_rank", 4)
        self.lora_args = lora_args
        self.lora_scale =getattr(lora_args, "lora_scale", 4)
        self.save_lora = False
        if len(lora_args) != 0:
            if hasattr(self, 'ref_model'):
                # Unload the reference model to free up memory
                mainlogger.info("----removing ref_model when use lora")
                del self.ref_model
            self.save_lora = True
            pass
    def shared_ref_pred(self,x_noisy, t, cond, kwargs):
        if hasattr(self, 'ref_model'):
            # import pdb; pdb.set_trace();
            ref_pred =  self.ref_model(x_noisy, t, **cond, **kwargs).detach() 
        else:
            with torch.no_grad():
                ## set lora scale to 0 to ban lora layer
                tune_lora_scale(model=self.model,alpha=0)
                ref_pred = self.model(x_noisy, t, **cond, **kwargs).detach() 
                ## reset lora scale to origin value
                tune_lora_scale(model=self.model,alpha=self.lora_scale)
        return ref_pred
    def _freeze_model(self):
        for name, para in self.model.diffusion_model.named_parameters():
            para.requires_grad = False
            
    def load_lora_from_ckpt(self,path):
        # if self.lora_implementation == "peft":
        #     self.model = peft.get_peft_model(self.model, self.lora_default_config)
        #     peft.set_peft_model_state_dict(self.model, torch.load(path))
        #     # print(torch.load(path)['state_dict'].keys());exit();
        # else:
        load_lora_state_dict(self.model,path)
    def merge_lora(self):
        # if not called, the result is correct 
        # if called, the results seems wrong 
        collapse_lora(self.model)
        monkeypatch_remove_lora(self.model)
    def inject_lora(self):
        lora_scale=self.lora_scale
        lora_implementation =  self.lora_implementation #"peft" # peft / else 
        print(f"=======Inject Lora Use {lora_implementation} ==============")
        if self.lora_implementation == "peft":
            self.lora_default_config = peft.LoraConfig(
                r=self.lora_rank,
                target_modules=["to_k", "to_v", "to_q","proj"],        # only diffusion_model has these modules
                lora_dropout=0.0,
            )
            self.model = peft.get_peft_model(self.model, self.lora_default_config)
            self.model.print_trainable_parameters()
            # self.cond_stage_model = peft.get_peft_model(self.c
            # ond_stage_model, lora_default_config)
            # self.cond_stage_model.print_trainable_parameters()
        else:
            self._freeze_model()
            if self.inject_unet:
                self.lora_require_grad_params, self.lora_names = \
                    inject_trainable_lora(self.model, self.inject_unet_key_word, 
                                            r=self.lora_rank,
                                            scale=lora_scale,
                                            verbose=True)# set to True for debug 
                                            # module_child_name=inject_unet_child_name
                                            # )
                print(f'inject unet: {self.lora_names}')
            if self.inject_clip:
                self.lora_require_grad_params_clip, self.lora_names_clip = \
                    inject_trainable_lora(self.cond_stage_model, self.inject_clip_key_word, 
                                            r=self.lora_rank,
                                            scale=lora_scale,
                                            verbose=True,
                                            )
                print(f'inject clip: {self.lora_names_clip}')
        
    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if self.use_scale:
            return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
                * x_start
                * extract_into_tensor(self.scale_arr, t, x_start.shape)
                + extract_into_tensor(
                    self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
                )
                * noise
            )
        else:
            return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
                * x_start
                + extract_into_tensor(
                    self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
                )
                * noise
            )



    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # only for very first batch, reset the self.scale_factor
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            assert (
                self.scale_factor == 1.0
            ), "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            mainlogger.info("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            mainlogger.info(f"setting self.scale_factor to {self.scale_factor}")
            mainlogger.info("### USING STD-RESCALING ###")
            mainlogger.info(f"std={z.flatten().std()}")

    # def on_fit_start(self):
    #     checkpoint_path = '/home/rliuay/haoyu/research/LVDM-UST-VideoCrafterft-master/init_ckpt/VideoCrafter1-Image2Video-512/0000.ckpt'
    #     self.trainer.save_checkpoint(checkpoint_path)
    #     print(f'Initial checkpoint saved at {checkpoint_path}')

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    @torch.no_grad()
    def encode_first_stage(self, x):
        # print("in encode first stage ",x.shape)
        if self.encoder_type == "2d" and x.dim() == 5:
            return self.encode_first_stage_2DAE(x)
        encoder_posterior = self.first_stage_model.encode(x)
        print("    pencoder post",encoder_posterior.shape)
        results = self.get_first_stage_encoding(encoder_posterior).detach()
        print("    result",results.shape)
        return results

    def encode_first_stage_2DAE(self, x):
        """encode frame by frame"""
        b, _, t, _, _ = x.shape
        results = torch.cat(
            [
                self.get_first_stage_encoding(self.first_stage_model.encode(x[:, :, i]))
                .detach()
                .unsqueeze(2)
                for i in range(t)
            ],
            dim=2,
        )
        return results
    
    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            z = rearrange(z, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
            
        z = 1. / self.scale_factor * z

        results = self.first_stage_model.decode(z, **kwargs)
            
        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    def differentiable_decode_first_stage(self, z, **kwargs):
        """same as decode_first_stage but without decorator"""
        return self.decode_core(z, **kwargs)

    @torch.no_grad()
    def get_batch_input(
        self,
        batch,
        random_uncond,
        return_first_stage_outputs=False,
        return_original_cond=False,
        is_imgbatch=False,
    ):
        ## image/video shape: b, c, t, h, w
        data_key = "jpg" if is_imgbatch else self.first_stage_key
        x = super().get_input(batch, data_key)
        if self.loss_type == "dpo":
            # print("in get batch input prechunk ",x.shape)
            x = torch.cat(x.chunk(2, dim=1))  # feed pixel values
            # print("after chunk",x.shape)
        # print(x.shape);exit()
        if is_imgbatch:
            ## pack image as video
            # x = x[:,:,None,:,:]
            # print("================in image_batch======================")
            b = x.shape[0] // self.temporal_length
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=self.temporal_length)
        x_ori = x
        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)

        ## get caption condition
        cond_key = "txt" if is_imgbatch else self.cond_stage_key
        cond = batch[cond_key]
        # why notice: this is classifier-free guidance implementation
        if random_uncond and self.uncond_type == "empty_seq":
            # print(type(cond),cond)
            for i, ci in enumerate(cond):
                # print(len(cond))
                if random.random() < self.uncond_prob:
                    cond[i] = ""
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))
        if random_uncond and self.uncond_type == "zero_embed":
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(ci)
        if self.loss_type == "dpo":
            cond_emb = cond_emb.repeat(2, 1, 1)
        out = [z, cond_emb]
        # print(z.shape,cond_emb.shape);exit()

        ## optional output: self-reconst or caption
        # why notice: following options are false during training ; true during inference
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x_ori, xrec])
        if return_original_cond:
            out.append(cond)

        return out

    def forward(self, x, c, **kwargs):
        if "t" in kwargs:
            t = kwargs.pop("t")
        else:
            t = torch.randint(
                0, self.num_timesteps, (x.shape[0],), device=self.device
            ).long()
            # t = t//2 # [0,500]
            # t += self.num_timesteps//2 #[500,1000]
            # mean = 0      # 均值设为0
            # std = 1000    # 标准差设为1000
            # 从高斯分布中采样，并将负数取绝对值，再取整
            # t = torch.normal(mean, std, (x.shape[0],)).abs().long().to(self.device)
        self.log(
            "train/timestep",int(t.max().cpu().detach().clone()),
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        return self.p_losses(x, c, t, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        is_imgbatch = False
        if "loader_img" in batch.keys():
            ratio = 10.0 / self.temporal_length
            if random.uniform(0.0, 10.0) < ratio:
                is_imgbatch = True
                batch = batch["loader_img"]
            else:
                batch = batch["loader_video"]
        else:
            pass
        # dupfactor 
        self.dupfactor = batch['dupfactor']
        x, c = self.get_batch_input(
            batch, random_uncond=random_uncond, is_imgbatch=is_imgbatch
        )
        loss, loss_dict = self(x, c, is_imgbatch=is_imgbatch, **kwargs)
        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, **kwargs):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            )
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)
        if self.loss_type == "dpo":
            self.ref_pred = self.shared_ref_pred(x_noisy, t, cond, kwargs)
        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.frame_cond:
            if self.cond_mask.device is not self.device:
                self.cond_mask = self.cond_mask.to(self.device)
            ## condition on fist few frames
            x_noisy = x_start * self.cond_mask + (1.0 - self.cond_mask) * x_noisy
        model_output = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        if self.frame_cond:
            ## [b,c,t,h,w]: only care about the predicted part (avoid disturbance)
            model_output = model_output[:, :, self.frame_cond :, :, :]
            target = target[:, :, self.frame_cond :, :, :]
        # why notice: model output shape [1, 16, 2, 40, 64]
        if self.loss_type == "dpo":
            loss_simple = self.get_loss(model_output, target, mean=False)
        else:
            loss_simple = self.get_loss(model_output, target, mean=False).mean(
                [1, 2, 3, 4]
            )

        if torch.isnan(loss_simple).any():
            print(f"loss_simple exists nan: {loss_simple}")
            # import pdb; pdb.set_trace()
            for i in range(loss_simple.shape[0]):
                if torch.isnan(loss_simple[i]).any():
                    loss_simple[i] = torch.zeros_like(loss_simple[i])

        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        if self.logvar.device is not self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        # logvar_t = self.logvar[t.item()].to(self.device) # device conflict when ddp shared
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if self.original_elbo_weight > 0:
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(
                dim=(1, 2, 3, 4)
            )
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
            loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(
            batch, random_uncond=self.classifier_free_guidance
        )
        ## sync_dist | rank_zero_only
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        # self.log("epoch/global_step", self.global_step.float(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        """
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        """
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(
                f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}"
            )
        torch.cuda.empty_cache()
        return loss

    def _get_denoise_row_from_list(self, samples, desc=""):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_timesteps, b, C, H, W

        if denoise_row.dim() == 5:
            # img, num_imgs= n_log_timesteps * bs, grid_size=[bs,n_log_timesteps]
            # 先batch再n，grid时候一行是一个sample的不同steps，batch是列，行是n
            denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
            denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            # video, grid_size=[n_log_timesteps*bs, t]
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, "n b c t h w -> b n c t h w")
            denoise_grid = rearrange(denoise_grid, "b n c t h w -> (b n) c t h w")
            denoise_grid = rearrange(denoise_grid, "n c t h w -> (n t) c h w")
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid

    @torch.no_grad()
    def log_images(
        self,
        batch,
        sample=True,
        ddim_steps=200,
        ddim_eta=1.0,
        plot_denoise_rows=False,
        unconditional_guidance_scale=1.0,
        **kwargs,
    ):
        """log images for LatentDiffusion"""
        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, x, xrec, xc = self.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=True,
            return_original_cond=True,
        )
        N, _, T, H, W = x.shape
        log["inputs"] = x
        log["reconst"] = xrec
        log["condition"] = xc

        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_cat, c_emb = c["c_concat"][0], c["c_crossattn"][0]
                    # log["condition_cat"] = c_cat
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc = torch.zeros_like(c_emb)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                    mask=self.cond_mask,
                    x0=z,
                    **kwargs,
                )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            **kwargs,
        )
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (
                0.5 * model_log_variance
            ).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
        **kwargs,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        # sample an initial noise
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(
                img, cond, ts, clip_denoised=self.clip_denoised, **kwargs
            )
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length, *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            mask=mask,
            x0=x0,
            **kwargs,
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            kwargs.update({"clean_cond": True})
            samples, intermediates = ddim_sampler.sample(
                ddim_steps, batch_size, shape, cond, verbose=False, **kwargs
            )

        else:
            samples, intermediates = self.sample(
                cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs
            )

        return samples, intermediates

    def configure_optimizers(self):
        # """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)
    
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            print("Setting up LR scheduler...")
            lr_scheduler = self.configure_schedulers(opt)
            return [opt], [lr_scheduler]
        return opt

    # def configure_optimizers(self):
    #     """configure_optimizers for LatentDiffusion"""
    #     lr = self.learning_rate
    #     if self.empty_params_only and hasattr(self, "empty_paras"):
    #         params = [
    #             p for n, p in self.model.named_parameters() if n in self.empty_paras
    #         ]
    #         print("self.empty_paras", len(self.empty_paras))
    #         for n, p in self.model.named_parameters():
    #             if n not in self.empty_paras:
    #                 p.requires_grad = False
    #         mainlogger.info(f"@Training [{len(params)}] Empty Paramters ONLY.")
    #     else:
    #         params = list(self.model.parameters())
    #         mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

    #     if self.learn_logvar:
    #         mainlogger.info("Diffusion model optimizing logvar")
    #         if isinstance(params[0], dict):
    #             params.append({"params": [self.logvar]})
    #         else:
    #             params.append(self.logvar)

    #     ## optimizer
    #     optimizer = torch.optim.AdamW(params, lr=lr)
    #     ## lr scheduler
    #     if self.use_scheduler:
    #         mainlogger.info("Setting up LambdaLR scheduler...")
    #         lr_scheduler = self.configure_schedulers(optimizer)
    #         return [optimizer], [lr_scheduler]

    #     return optimizer

    def configure_schedulers(self, optimizer):
        assert "target" in self.scheduler_config
        scheduler_name = self.scheduler_config.target.split(".")[-1]
        interval = self.scheduler_config.interval
        frequency = self.scheduler_config.frequency
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                "scheduler": LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                "interval": interval,
                "frequency": frequency,
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                "scheduler": CosineAnnealingLR(
                    optimizer, T_max=decay_steps, last_epoch=last_step
                ),
                "interval": interval,
                "frequency": frequency,
            }
        else:
            raise NotImplementedError
        return lr_scheduler

    def on_save_checkpoint(self, checkpoint):
        # Remove reference model from checkpoint
        # since it won't be changed
        # and keep consistent to original model
        # checkpoint contain statedict
        print("==========saving checkpoint in ddpm3d=========")
        keys_to_remove = [
            key for key in checkpoint["state_dict"].keys() if "ref_model" in key
        ]
        for key in keys_to_remove:
            checkpoint["state_dict"].pop(key, None)
        return checkpoint
from lvdm.models.turbo_utils.lora import save_lora_weight,extract_lora_ups_down
from lvdm.models.turbo_utils.lora_handler import LoraHandler
from lvdm.models.turbo_utils.lora import collapse_lora, monkeypatch_remove_lora
from lvdm.models.turbo_utils.turbo_scheduler import T2VTurboScheduler
from lvdm.models.turbo_utils.ode_solver import DDIMSolver
class T2VTurboDPO(LatentDiffusion):
    def __init__(self, 
                 pretrained_unet_path=None,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1. Create the noise scheduler and the desired noise schedule.
        # print(kwargs)
        self.noise_scheduler = T2VTurboScheduler(
            linear_start=kwargs["linear_start"],
            linear_end=kwargs["linear_end"],
        )
    
        # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
        alpha_schedule = torch.sqrt(self.noise_scheduler.alphas_cumprod)
        sigma_schedule = torch.sqrt(1 - self.noise_scheduler.alphas_cumprod)
        use_scale = False 
        scale_b = kwargs["scale_b"]
        self.num_ddim_timesteps = 50 
        ddim_eta = 0.0 
        self.solver = DDIMSolver(
            self.noise_scheduler.alphas_cumprod.numpy(),
            ddim_timesteps=self.num_ddim_timesteps,
            use_scale=use_scale,
            scale_b=scale_b,
            ddim_eta=ddim_eta,
        )
        # self.device="cuda"
        # self.solver = self.solver.to(self.device)
        # print("self solver device",self.solver.ddim_timesteps.device,self.device)
        # self.loss_type = "dpo"
        # with open("before_lora.txt",'w') as f:
        #     for n,p in self.model.named_parameters():
        #         f.write(f"{n} {p.shape}\n")
        self._freeze_model()
        use_unet_lora = True
        # import pdb; pdb.set_trace()
        lora_manager = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=use_unet_lora,
            save_for_webui=True,
            unet_replace_modules=["UNetModel"],
        )
        # pretrained_unet_path="/root/autodl-tmp/unet_lora.pt"
        # import pdb; pdb.set_trace()
        unet_lora_params, unet_negation = lora_manager.add_lora_to_model(
            use_unet_lora,
            self.model,
            lora_manager.unet_replace_modules,
            lora_path=pretrained_unet_path,
            dropout=0.1,
            r=64,
        )
        # update ref_model 
        lora_manager = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=use_unet_lora,
            save_for_webui=True,
            unet_replace_modules=["UNetModel"],
        )
        # pretrained_unet_path="/root/autodl-tmp/unet_lora.pt"
        # import pdb; pdb.set_trace()
        unet_lora_params, unet_negation = lora_manager.add_lora_to_model(
            use_unet_lora,
            self.ref_model,
            lora_manager.unet_replace_modules,
            lora_path=pretrained_unet_path,
            dropout=0.1,
            r=64,
        )
        for name,param in self.ref_model.named_parameters():
            param.requires_grad = False
        # with open("after_lora_no_ckpt.txt",'w') as f:
        #     for n,p in self.model.named_parameters():
        #         f.write(f"{n} {p.shape}\n")
        # for name,param in self.model.named_parameters():
        #     if "lora" in name:
        #         print(param.shape)
        self.ref_model.eval()
        # collapse_lora(self.model, lora_manager.unet_replace_modules)
        # monkeypatch_remove_lora(self.model)
        # with open("after_collapse.txt",'w') as f:
        #     for n,p in self.model.named_parameters():
        #         f.write(f"{n} {p.shape}\n")
        import copy
        # import gc
        # self.ref_model = copy.deepcopy(self.model)
        # collapse_lora(self.ref_model, lora_manager.unet_replace_modules)
        # monkeypatch_remove_lora(self.ref_model)
        # # 禁用 ref_model 的梯度计算
        for param in self.ref_model.parameters():
            param.requires_grad = False
        # gc.collect()  # Python垃圾回收
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()  # 释放GPU显存
    def forward(self, x, c,*args ,**kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # t = torch.randint(
        #     0, self.num_timesteps, (x.shape[0],), device=self.device
        # ).long()
        index = torch.randint(
            0, self.num_ddim_timesteps, (x.shape[0],), device=self.device
        ).long()
        self.solver = self.solver.to(self.device)
        # print(index.device,self.solver.ddim_timesteps.device,self.solver.ddim_timesteps.dtype)
        start_timesteps = self.solver.ddim_timesteps[index]
        return self.p_losses(x,c,t=start_timesteps , *args, **kwargs)
    def q_sample(self, x_start, t, noise=None):
        # import pdb;pdb.set_trace()
        # noise = torch.randn_like(x_start)
        noisy_model_input = self.noise_scheduler.add_noise(
            x_start, noise, t
        )
        return noisy_model_input
        # noise = default(noise, lambda: torch.randn_like(x_start))
        # if self.use_scale:
        #     return (
        #         extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        #         * x_start
        #         * extract_into_tensor(self.scale_arr, t, x_start.shape)
        #         + extract_into_tensor(
        #             self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        #         )
        #         * noise
        #     )
        # else:
        #     return (
        #         extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        #         * x_start
        #         + extract_into_tensor(
        #             self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        #         )
        #         * noise
        #     )
    def on_save_checkpoint(self,checkpoint):
        weights = []
        for _up, _down in extract_lora_ups_down(
            self.model,target_replace_module={"UNetModel"}
        ):
            weights.append(_up.weight.to("cpu").to(torch.float32))
            weights.append(_down.weight.to("cpu").to(torch.float32))
            # print(_up.weight.shape,_down.weight.shape)\
        # import pdb;pdb.set_trace()
        print("len of weights",len(weights))
        checkpoint['state_dict']=weights
        return checkpoint
# # import rlhf utils 
# from lvdm.models.rlhf_utils.batch_ddim import batch_ddim_sampling
# from lvdm.models.rlhf_utils.reward_fn import aesthetic_loss_fn

# class RewardLVDMTrainer(LatentDiffusion):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Reward Gradient Training Using LoRA as a default
#         # TODO: use config and getattr to set default values
#         # sampling configs for DDIM
#         self.ddim_eta = 1.0 
#         self.ddim_steps = 20 # reduce some steps to speed up sampling process 
#         self.n_samples = 1
#         self.fps = 24 # default 24 following VADER 
#         # rlhf configs  
#         self.backprop_mode = "last" #m"backpropagation mode: 'last', 'rand', 'specific'"
#         self.decode_frame = -1  # it could also be any number str like '3', '10'. alt: alternate frames, fml: first, middle, last frames, all: all frames. '-1': random frame
#         self.reward_loss_type = "aesthetic" 
#         # self.configure_reward_loss()    
        
#     def configure_reward_loss(self,loss_type=None):
#         if loss_type is None:
#             loss_type = self.reward_loss_type
            
#         if loss_type == "aesthetic":
#             self.loss_fn = aesthetic_loss_fn(grad_scale=0.1,
#                                         aesthetic_target=10,
#                                         torch_dtype = self.model.dtype,
#                                         device = self.device)
#         else:
#             raise NotImplementedError(f"loss type {loss_type} not implemented")
        
#     def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
#         # the reason why configure here is to wait the model transfered to target device
#         # otherwise, the loss_fn will be on cpu if configure in __init__
#         self.configure_reward_loss()
#     def training_step(self, batch, batch_idx):
#         """training_step for Reward Model Feedback"""
#         # 构造输入,与denoise训练不一样的地方是这里不需要VAE encode ，只需要nosie shape 
#         # 默认的cond是text prompt
#         prompts = batch[self.cond_stage_key]
#         # print(prompts) #  Elon mask is talking
#         x, c = self.get_batch_input(
#             batch, random_uncond=self.classifier_free_guidance, is_imgbatch=False
#         ) # x is latent image ; c is text embedding(tensor)
#         kwargs ={}
#         batch_size = x.shape[0]
#         noise_shape = (batch_size, self.channels, self.temporal_length//4, *self.image_size)# (1, 4, 4, 40, 64)
#         # print("noise shape",noise_shape)
#         fps = torch.tensor([self.fps]*batch_size).to(self.device).long()
#         cond = {"c_crossattn": [c], "fps": fps}
#         # 注意这个batch_ddim_sampling是VADER自己改过的版本 
#         # 输入的cond 是 cond = {"c_crossattn": [text_emb], "fps": fps}
#         batch_samples = batch_ddim_sampling(self, cond, noise_shape, self.n_samples, \
#                                             self.ddim_steps, self.ddim_eta, self.classifier_free_guidance,\
#                                             None, backprop_mode=self.backprop_mode, decode_frame=self.decode_frame,\
#                                             **kwargs)

#         video_frames_ = batch_samples.permute(1,0,3,2,4,5)      # batch,samples,channels,frames,height,width >> s,b,f,c,h,w
#         # print("video_frames shape",batch_samples.shape,video_frames_.requires_grad)
#         s_, bs, nf, c_, h_, w_ = video_frames_.shape
#         assert s_ == 1                                  # samples should only be on single sample in training mode
#         video_frames_ = video_frames_.squeeze(0)        # s,b,f,c,h,w >> b,f,c,h,w
#         assert nf == 1                                  # reward should only be on single frame
#         video_frames_ = video_frames_.squeeze(1)        # b,f,c,h,w >> b,c,h,w
#         video_frames_ = video_frames_.to(x.dtype)

#         # some reward fn may require prompts as input. 
#         loss, rewards = self.loss_fn(video_frames_)  # rewards is for logging only. 
#         loss_dict = {"reward_train_loss": loss.detach().item(), "step_reward": rewards.detach().item()}
#         self.log_dict(
#             loss_dict,
#             prog_bar=True,
#             logger=True,
#             on_step=True,
#             on_epoch=True,
#             sync_dist=False,
#         )
#         self.log(
#             "global_step",
#             self.global_step,
#             prog_bar=True,
#             logger=True,
#             on_step=True,
#             on_epoch=False,
#         )
#         if (batch_idx + 1) % self.log_every_t == 0:
#             mainlogger.info(
#                 f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss} reward={rewards}"
#             )
#         return loss

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
#         # the training steps count is short no need for scheduler as default 
#         if self.use_scheduler:
#             lr_scheduler = self.configure_schedulers(opt)
#             return [opt], [lr_scheduler]
#         return opt


class LatentVisualDiffusion(LatentDiffusion):
    def __init__(
        self, cond_img_config, finegrained=False, random_cond=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.random_cond = random_cond
        self.instantiate_img_embedder(cond_img_config, freeze=True)
        num_tokens = 16 if finegrained else 4
        self.image_proj_model = self.init_projector(
            use_finegrained=finegrained,
            num_tokens=num_tokens,
            input_dim=1024,
            cross_attention_dim=1024,
            dim=1280,
        )

    def instantiate_img_embedder(self, config, freeze=True):
        embedder = instantiate_from_config(config)
        if freeze:
            self.embedder = embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False

    def init_projector(
        self, use_finegrained, num_tokens, input_dim, cross_attention_dim, dim
    ):
        if not use_finegrained:
            image_proj_model = ImageProjModel(
                clip_extra_context_tokens=num_tokens,
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=input_dim,
            )
        else:
            image_proj_model = Resampler(
                dim=input_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_tokens,
                embedding_dim=dim,
                output_dim=cross_attention_dim,
                ff_mult=4,
            )
        return image_proj_model

    ## Never delete this func: it is used in log_images() and inference stage
    def get_image_embeds(self, batch_imgs):
        ## img: b c h w
        img_token = self.embedder(batch_imgs)
        img_emb = self.image_proj_model(img_token)
        return img_emb


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(
        self,
        x,
        t,
        c_concat: list = None,
        c_crossattn: list = None,
        c_adm=None,
        s=None,
        mask=None,
        **kwargs,
    ):
        # temporal_context = fps is None
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == "hybrid":
            ## it is just right [b,c,t,h,w]: concatenate in channel dim
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, **kwargs)
        elif self.conditioning_key == "resblockcond":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == "hybrid-adm":
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm, **kwargs)
        elif self.conditioning_key == "hybrid-time":
            assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s)
        elif self.conditioning_key == "concat-time-mask":
            # assert s is not None
            # mainlogger.info('x & mask:',x.shape,c_concat[0].shape)
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask)
        elif self.conditioning_key == "concat-adm-mask":
            # assert s is not None
            # mainlogger.info('x & mask:',x.shape,c_concat[0].shape)
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=None, y=s, mask=mask)
        elif self.conditioning_key == "hybrid-adm-mask":
            cc = torch.cat(c_crossattn, 1)
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=cc, y=s, mask=mask)
        elif (
            self.conditioning_key == "hybrid-time-adm"
        ):  # adm means y, e.g., class index
            # assert s is not None
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s, y=c_adm)
        else:
            raise NotImplementedError()

        return out