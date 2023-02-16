import itertools
import torch

import numpy as np
import pytorch_lightning as pl

from einops import rearrange
from pytorch_lightning.utilities.distributed import rank_zero_only
from scipy import integrate
from tqdm import tqdm
from tqdm.auto import trange, tqdm
from functools import partial

from ldm.modules.ema import LitEma
from ldm.models.autoencoder import VQModelInterface
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import exists, default, instantiate_from_config, disabled_train


class DiffusionWrapperv2(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.sequential_cross_attn = diff_model_config.pop("sequential_crossattn", False)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm', 'crossattn-adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None):
        self.diffusion_model.dtype = x.dtype
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            if not self.sequential_cross_attn:
                cc = torch.cat(c_crossattn, 1)
            else:
                cc = c_crossattn
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, x, t, cc):
        out = self.diffusion_model(x, t, context=cc)
        return out


class DiffusionWrapperOut(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, h, emb, tp, hs, cc):
        return self.diffusion_model(h, emb, tp, hs, context=cc)


class DDPM(pl.LightningModule):
    # This one has all v1 and v2 stuff inside
    # Also it's quiet now
    def __init__(self,
                 unet_config=None,
                 timesteps=1000,
                 beta_schedule="linear",
                 ckpt_path=None,
                 ignore_keys=None,
                 load_only_unet=False,
                 monitor="val/loss",
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
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 make_it_fit=False,
                 ucg_training=None,
                 reset_ema=False,
                 reset_num_ema_updates=False,
                 loss_type="l2",
                 *v2_args,
                 **v2_kwargs
                 ):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = []
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps", "x0" and "v"'
        self.parameterization = parameterization
        # print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        if unet_config is not None:
            self.model = DiffusionWrapperv2(unet_config, conditioning_key)
        self.use_ema = use_ema

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor

        self.make_it_fit = make_it_fit
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = torch.nn.Parameter(self.logvar, requires_grad=True)

        self.ucg_training = ucg_training or dict()
        if self.ucg_training:
            self.ucg_prng = np.random.RandomState()

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


class FirstStage(DDPM):
    """main class"""

    def __init__(self,
                 first_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__()
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        self.instantiate_first_stage(first_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)


class CondStage(DDPM):
    """main class"""

    def __init__(self,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__()
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c


class UNet(DDPM):
    """main class"""

    def __init__(self,
                 unetConfigEncode=None,
                 unetConfigDecode=None,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 unet_bs=1,
                 scale_by_std=False,
                 force_null_conditioning=False,
                 *args, **kwargs):

        self.parameterization = "eps"
        self.v1 = True

        # the ugly comfy params
        self.x_spare_part = None
        self.current_step = 0
        self.total_steps = 0

        self.force_null_conditioning = force_null_conditioning
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.cdevice = "cuda"
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.turbo = False
        self.unet_bs = unet_bs
        self.restarted_from_ckpt = False
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        if unetConfigEncode is not None and unetConfigDecode is not None:
            self.unetConfigEncode = unetConfigEncode
            self.unetConfigDecode = unetConfigDecode
            self.cond_stage_forward = cond_stage_forward
            self.model1 = DiffusionWrapper(self.unetConfigEncode)
            self.model2 = DiffusionWrapperOut(self.unetConfigDecode)
            self.model1.eval()
            self.model2.eval()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.cdevice)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if not self.turbo:
            self.model1.to(self.cdevice)

        step = self.unet_bs
        h, emb, hs = self.model1(x_noisy[0:step], t[:step], cond[:step])
        bs = cond.shape[0]

        # assert bs%2 == 0
        lenhs = len(hs)

        for i in range(step, bs, step):
            h_temp, emb_temp, hs_temp = self.model1(x_noisy[i:i + step], t[i:i + step], cond[i:i + step])
            h = torch.cat((h, h_temp))
            emb = torch.cat((emb, emb_temp))
            for j in range(lenhs):
                hs[j] = torch.cat((hs[j], hs_temp[j]))

        if not self.turbo:
            self.model1.to("cpu")
            self.model2.to(self.cdevice)

        hs_temp = [hs[j][:step] for j in range(lenhs)]
        x_recon = self.model2(h[:step], emb[:step], x_noisy.dtype, hs_temp, cond[:step])

        for i in range(step, bs, step):
            hs_temp = [hs[j][i:i + step] for j in range(lenhs)]
            x_recon1 = self.model2(h[i:i + step], emb[i:i + step], x_noisy.dtype, hs_temp, cond[i:i + step])
            x_recon = torch.cat((x_recon, x_recon1))

        if not self.turbo:
            self.model2.to("cpu")

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def register_buffer1(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(self.cdevice):
                attr = attr.to(torch.device(self.cdevice))
        setattr(self, name, attr)

    def make_schedule_plms(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        pass

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps, verbose=verbose)

        assert self.alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = lambda x: x.to(self.cdevice)
        self.register_buffer1('betas', to_torch(self.betas))
        self.register_buffer1('alphas_cumprod', to_torch(self.alphas_cumprod))
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer1('ddim_sigmas', ddim_sigmas)
        self.register_buffer1('ddim_alphas', ddim_alphas)
        self.register_buffer1('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer1('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

    @torch.no_grad()
    def stochastic_encode(self, x0, t, seed, ddim_eta, ddim_steps, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        try:
            self.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        except:
            self.make_schedule(ddim_num_steps=ddim_steps + 1, ddim_eta=ddim_eta, verbose=False)

        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)

        if noise is None:
            b0, b1, b2, b3 = x0.shape
            img_shape = (1, b1, b2, b3)
            tens = []
            print("seeds used = ", [seed + s for s in range(b0)])
            for _ in range(b0):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=x0.device))
                seed += 1
            noise = torch.cat(tens)
            del tens
        if int(sqrt_alphas_cumprod.shape.numel()) != int(t):
            t = (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                 extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape) * noise)
            return t
        else:
            self.make_schedule(ddim_num_steps=ddim_steps + 1, ddim_eta=ddim_eta, verbose=False)
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            t = (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                 extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape) * noise)
            return t

    @torch.no_grad()
    def sample(self,
               S,
               conditioning,
               x0=None,
               shape=None,
               S_ddim_steps=None,
               seed=1234,
               callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               sampler="plms",
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               txt_scale=1.5,
               unconditional_conditioning=None,
               batch_size=None,
               mode="default"
               ):

        if self.turbo and self.v1:
            self.model1.to(self.cdevice)
            self.model2.to(self.cdevice)
        if mode == "default":
            if x0 is None:
                batch_size, b1, b2, b3 = shape
                img_shape = (1, b1, b2, b3)
                tens = []
                print("seeds used = ", [seed + s for s in range(batch_size)])
                for _ in range(batch_size):
                    torch.manual_seed(seed)
                    tens.append(torch.randn(img_shape, device=self.cdevice))
                    seed += 1
                noise = torch.cat(tens)
                del tens
                try:
                    self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
                except Exception:
                    self.make_schedule(ddim_num_steps=S + 1, ddim_eta=eta, verbose=False)
            x_latent = noise if x0 is None else x0
        else:
            batch_size, b1, b2, b3 = shape
            img_shape = (1, b1, b2, b3)
            tens = []
            print("seeds used = ", [seed + s for s in range(batch_size)])
            for _ in range(batch_size):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=self.cdevice))
                seed += 1
            noise = torch.cat(tens)
            del tens
            try:
                self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
            except Exception:
                self.make_schedule(ddim_num_steps=S + 1, ddim_eta=eta, verbose=False)

            if mode == "runway":
                x_latent = torch.cat((noise, x0, mask[:, :1, :, :]), dim=1)
            elif mode == "pix2pix":
                x_latent = torch.cat((noise, x0), dim=1)
            else:
                x_latent = noise if x0 is None else x0

        # sampling
        if sampler == "plms":
            self.make_schedule_plms(ddim_num_steps=S, ddim_eta=eta, verbose=False)
            print(f'Data shape for PLMS sampling is {shape}')
            samples = self.plms_sampling(conditioning, batch_size, x_latent,
                                         callback=callback, mode=mode,
                                         quantize_denoised=quantize_x0,
                                         mask=mask, x0=x0,
                                         ddim_use_original_steps=False,
                                         noise_dropout=noise_dropout,
                                         temperature=temperature,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs,
                                         log_every_t=log_every_t,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=unconditional_conditioning,
                                         )

        elif sampler == "ddim":
            samples = self.ddim_sampling(x_latent, conditioning, S, callback=callback, mode=mode, txt_scale=txt_scale,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=unconditional_conditioning,
                                         mask=mask, init_latent=x_T, use_original_steps=False)
        else:
            samples = self.k_sampling(x_latent, conditioning, S, sampler, S_ddim_steps=S_ddim_steps,
                                      unconditional_guidance_scale=unconditional_guidance_scale, mode=mode,
                                      unconditional_conditioning=unconditional_conditioning, callback=callback,
                                      mask=mask, init_latent=x0, use_original_steps=False)

        if self.turbo and self.v1:
            self.model1.to("cpu")
            self.model2.to("cpu")

        return samples

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)).to(x_start.device)
        return (extract_into_tensor(self.sqrt_alphas_cumprod.to(x_start.device), t.to(x_start.device),
                                    x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t.to(x_start.device),
                                    x_start.shape) * noise)

    @torch.no_grad()
    def plms_sampling(self, cond, b, img,
                      ddim_use_original_steps=False, mode="default",
                      callback=None, quantize_denoised=False,
                      mask=None, x0=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):

        device = self.betas.device
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            self.current_step = i
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts).to(img.device)
                img = img_orig * mask + (1. - mask) * img
                del img_orig

            outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      old_eps=old_eps, t_next=ts_next, mode=mode)
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(pred_x0)
        return img

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, mode="default",
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if self.parameterization == "v":
            e_t = self.predict_eps_from_z_and_v(x, t, e_t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t

    @torch.no_grad()
    def sample_dpmpp_2s_ancestral(self, model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.,
                                  s_noise=1.):
        """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver-2++(2S)
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
                denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
            # Noise addition
            x = x + torch.randn_like(x) * s_noise * sigma_up
        return x

    def get_model_output_k(self, x, t, unconditional_conditioning, c, cond_scale, model_wrap_sigmas,
                           text_cfg_scale=1.5, mode="default"):
        def get_scalings(sigma):
            c_out = -sigma
            c_in = 1 / (sigma ** 2 + 1.) ** 0.5
            return c_out, c_in

        def sigma_to_t(sigma):
            dists = torch.abs(sigma - model_wrap_sigmas[:, None])
            low_idx, high_idx = torch.sort(torch.topk(dists, dim=0, k=2, largest=False).indices, dim=0)[0]
            low, high = model_wrap_sigmas[low_idx], model_wrap_sigmas[high_idx]
            w = (low - sigma) / (low - high)
            w = w.clamp(0, 1)
            t = (1 - w) * low_idx + w * high_idx
            return t.view(sigma.shape)

        def apply_inner_model(input_x, sigma, **kwargs):
            c_out, c_in = [self.append_dims(x, input_x.ndim) for x in get_scalings(sigma)]
            eps = self.apply_model(input_x * c_in, sigma_to_t(sigma), **kwargs)
            return input_x + eps * c_out

        multiplier = 3 if mode == "pix2pix" else 2
        x_in = torch.cat([x] * multiplier)
        t_in = torch.cat([t] * multiplier)
        if mode == "pix2pix":
            self.x_spare_part = x[:, 4:, :, :]
            c_in = torch.cat([c, c, unconditional_conditioning])
            out_cond, out_img_cond, out_uncond = self.apply_model(x_in, t_in, c_in).chunk(3)
            e_t = out_uncond + text_cfg_scale * (
                    out_cond - out_img_cond) + cond_scale * (out_img_cond - out_uncond)
        else:
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = apply_inner_model(x_in, t_in, cond=c_in).chunk(2)
            e_t = e_t_uncond + cond_scale * (e_t - e_t_uncond)
        # if self.parameterization == "v":
        #     e_t = self.predict_eps_from_z_and_v(x, t.to(torch.int64), e_t)
        return e_t

    def linear_multistep_coeff(self, order, t, i, j):
        if order - 1 > i:
            raise ValueError(f'Order {order} too high for step {i}')

        def fn(tau):
            prod = 1.
            for k in range(order):
                if j == k:
                    continue
                prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
            return prod

        return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]

    def p_k_sample(self, x, c, sigmas, sampler, i, s_in, mode="default", unconditional_guidance_scale=1.,
                   unconditional_conditioning=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                   model_wrap_sigmas=None, ds=None, order=4):
        if ds is None:
            ds = []
        b, *_, device = *x.shape, x.device

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        self.x_spare_part = None
        match sampler:
            case "dpmpp_2m":
                denoised = self.get_model_output_k(x, sigmas[i] * s_in, unconditional_conditioning, c,
                                                   unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x = x[:, :4, :, :]
                t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                h = t_next - t
                if self.old_denoised is None or sigmas[i + 1] == 0:
                    x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
                else:
                    h_last = t - t_fn(sigmas[i - 1])
                    r = h_last / h
                    denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * self.old_denoised
                    x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
                self.old_denoised = denoised
            case "dpmpp_2s_ancestral":
                denoised = self.get_model_output_k(x, sigmas[i] * s_in, unconditional_conditioning, c,
                                                   unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x = x[:, :4, :, :]
                sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=1.)
                if sigma_down == 0:
                    # Euler method
                    d = self.to_d(x, sigmas[i], denoised).to(x.dtype).to(x.device)
                    dt = sigma_down - sigmas[i]
                    x = x + d * dt
                else:
                    # DPM-Solver-2++(2S)
                    t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                    r = 1 / 2
                    h = t_next - t
                    s = t + r * h
                    x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
                    if self.x_spare_part is not None:
                        x_2 = torch.cat((x_2, self.x_spare_part), dim=1)
                    denoised_2 = self.get_model_output_k(x_2, sigma_fn(s) * s_in, unconditional_conditioning, c,
                                                         unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                    x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
                # Noise addition
                x = x + torch.randn_like(x) * s_noise * sigma_up
            case "dpm_a":
                denoised = self.get_model_output_k(x, sigmas[i] * s_in, unconditional_conditioning, c,
                                                   unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x = x[:, :4, :, :]
                sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1])
                d = self.to_d(x, sigmas[i], denoised).to(x.dtype).to(x.device)
                # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
                sigma_mid = ((sigmas[i] ** (1 / 3) + sigma_down ** (1 / 3)) / 2) ** 3
                dt_1 = sigma_mid - sigmas[i]
                dt_2 = sigma_down - sigmas[i]
                x_2 = x + d * dt_1
                if self.x_spare_part is not None:
                    x_2 = torch.cat((x_2, self.x_spare_part), dim=1)
                denoised_2 = self.get_model_output_k(x_2, sigma_mid * s_in, unconditional_conditioning, c,
                                                     unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x_2 = x_2[:, :4, :, :]
                d_2 = self.to_d(x_2, sigma_mid, denoised_2).to(x.dtype).to(x.device)
                x = x + d_2 * dt_2
                x = x + torch.randn_like(x) * sigma_up
            case "dpm_2":
                gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
                eps = torch.randn_like(x) * s_noise
                sigma_hat = sigmas[i] * (gamma + 1)
                if gamma > 0:
                    x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
                denoised = self.get_model_output_k(x, sigma_hat * s_in, unconditional_conditioning, c,
                                                   unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x = x[:, :4, :, :]
                d = self.to_d(x, sigma_hat, denoised).to(x.dtype).to(x.device)
                # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
                sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
                dt_1 = sigma_mid - sigma_hat
                dt_2 = sigmas[i + 1] - sigma_hat
                x_2 = x + d * dt_1
                if self.x_spare_part is not None:
                    x_2 = torch.cat((x_2, self.x_spare_part), dim=1)
                denoised_2 = self.get_model_output_k(x_2, sigma_mid * s_in, unconditional_conditioning, c,
                                                     unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x_2 = x_2[:, :4, :, :]
                d_2 = self.to_d(x_2, sigma_mid, denoised_2).to(x.device)
                x = x + d_2 * dt_2
            case "euler_a":
                denoised = self.get_model_output_k(x, sigmas[i] * s_in, unconditional_conditioning, c,
                                                   unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x = x[:, :4, :, :]
                sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1])
                d = self.to_d(x, sigmas[i], denoised).to(x.dtype).to(x.device)
                # Euler method
                dt = sigma_down - sigmas[i]
                x = x + d * dt
                x = x + torch.randn_like(x) * sigma_up
            case "euler":
                gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
                eps = (torch.randn_like(x) * s_noise).to(x.dtype).to(x.device)
                sigma_hat = sigmas[i] * (gamma + 1)
                if gamma > 0:
                    x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
                denoised = self.get_model_output_k(x, sigma_hat * s_in, unconditional_conditioning, c,
                                                   unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x = x[:, :4, :, :]
                d = self.to_d(x, sigma_hat, denoised).to(x.dtype).to(x.device)
                dt = sigmas[i + 1] - sigma_hat
                # Euler method
                x = x + d * dt
            case "heun":
                gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
                eps = torch.randn_like(x) * s_noise
                sigma_hat = sigmas[i] * (gamma + 1)
                if gamma > 0:
                    x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
                denoised = self.get_model_output_k(x, sigma_hat * s_in, unconditional_conditioning, c,
                                                   unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x = x[:, :4, :, :]
                d = self.to_d(x, sigma_hat, denoised).to(x.dtype).to(x.device)
                dt = sigmas[i + 1] - sigma_hat
                if sigmas[i + 1] == 0:
                    # Euler method
                    x = x + d * dt
                else:
                    # Heun's method
                    x_2 = x + d * dt
                    if self.x_spare_part is not None:
                        x_2 = torch.cat((x_2, self.x_spare_part), dim=1)
                    denoised_2 = self.get_model_output_k(x_2, sigmas[i + 1] * s_in, unconditional_conditioning, c,
                                                         unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                    if mode == "pix2pix":
                        x_2 = x_2[:, :4, :, :]
                    d_2 = self.to_d(x_2, sigmas[i + 1], denoised_2).to(x.dtype).to(x.device)
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * dt
            case "lms":
                denoised = self.get_model_output_k(x, sigmas[i] * s_in, unconditional_conditioning, c,
                                                   unconditional_guidance_scale, model_wrap_sigmas, mode=mode)
                if mode == "pix2pix":
                    x = x[:, :4, :, :]
                d = self.to_d(x, sigmas[i], denoised).to(x.dtype).to(x.device)
                ds.append(d)
                if len(ds) > order:
                    ds.pop(0)
                cur_order = min(i + 1, order)
                coeffs = [self.linear_multistep_coeff(cur_order, sigmas.cpu(), i, j) for j in range(cur_order)]
                x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
        if self.x_spare_part is not None:
            x = torch.cat((x, self.x_spare_part), dim=1)
        return x

    @torch.no_grad()
    def ddim_sampling(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
                      mask=None, init_latent=None, use_original_steps=False, callback=None, mode="default",
                      txt_scale=1.5):

        timesteps = self.ddim_timesteps
        timesteps = timesteps[:t_start]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            self.current_step = i
            x0 = init_latent if init_latent is not None else torch.randn_like(x_dec)
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            if mask is not None and x0.shape[1] != 9:
                # x0_noisy = self.add_noise(mask, torch.tensor([index] * x0.shape[0]).to(self.cdevice))
                x0_noisy = x0
                x_dec = x0_noisy * mask + (1. - mask) * x_dec

            x_dec = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                       unconditional_guidance_scale=unconditional_guidance_scale, mode=mode,
                                       text_cfg_scale=txt_scale, unconditional_conditioning=unconditional_conditioning)
            if callback:
                callback(x_dec)

        if mask is not None:
            return x0 * mask + (1. - mask) * x_dec
        if mode == "pix2pix":
            x_dec = x_dec[:, :4, :, :]
        return x_dec

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, text_cfg_scale=1.5,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, mode="default"):
        b, *_, device = *x.shape, x.device
        multiplier = 3 if mode == "pix2pix" else 2
        x_spare_part = None
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * multiplier)
            t_in = torch.cat([t] * multiplier)
            if mode == "pix2pix":
                x_spare_part = x[:, 4:, :, :]
                c_in = torch.cat([c, c, unconditional_conditioning])
                out_cond, out_img_cond, out_uncond = self.apply_model(x_in, t_in, c_in).chunk(3)
                model_output = out_uncond + text_cfg_scale * (
                        out_cond - out_img_cond) + unconditional_guidance_scale * (out_img_cond - out_uncond)
                x = x[:, :4, :, :]
            else:
                c_in = torch.cat([unconditional_conditioning, c])
                model_uncond, model_t = self.apply_model(x_in, t_in, c_in).chunk(2)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.parameterization == "v":
            e_t = self.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        int(alphas[index] + 0)  # fixes a bug where generation gets stuck here for no reason at all
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        if x_spare_part is not None:
            x_prev = torch.cat((x_prev, x_spare_part), dim=1)
        return x_prev

    def t_to_sigma(self, t, k_sigmas):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        return (1 - w) * k_sigmas[low_idx] + w * k_sigmas[high_idx]

    def get_sigmas(self, n, k_sigmas):
        t_max = len(k_sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=k_sigmas.device)
        return self.append_zero(self.t_to_sigma(t, k_sigmas))

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7., device='cpu'):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return self.append_zero(sigmas).to(device)

    def k_sampling(self, x_latent, cond, S, sampler, unconditional_guidance_scale=1.0,
                   unconditional_conditioning=None, S_ddim_steps=None, callback=None,
                   mask=None, init_latent=None, use_original_steps=False, mode="default"):
        timesteps = self.ddim_timesteps
        timesteps = timesteps[:S]
        total_steps = timesteps.shape[0]
        self.old_denoised = None
        print(f"Running {sampler} Sampling with {total_steps} timesteps")
        model_wrap_sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).to(x_latent.device)

        if init_latent is not None and mode == "default":  # img2img
            sigmas = self.get_sigmas(S_ddim_steps, model_wrap_sigmas)
            noise = torch.randn_like(x_latent, device=x_latent.device) * sigmas[S_ddim_steps - S - 1]
            x_latent = x_latent + noise
            sigmas = sigmas[S_ddim_steps - S - 1:]
        elif init_latent is not None and mode != "default":
            sigmas = self.get_sigmas(S, model_wrap_sigmas)
            noise = torch.randn_like(x_latent[:, 4:, :, :], device=x_latent.device) * sigmas[0]
            x_latent[:, 4:, :, :] = x_latent[:, 4:, :, :] + noise
        else:
            sigmas = self.get_sigmas(S, model_wrap_sigmas)
            x_latent = x_latent * sigmas[0]

        s_in = x_latent.new_ones([x_latent.shape[0]]).to(x_latent.dtype).to(x_latent.device)
        ds = []
        for i in trange(total_steps, desc='Decoding image', total=total_steps):
            self.current_step = i
            if mask is not None and x_latent.shape[1] != 9:
                x_latent = init_latent * mask + (1. - mask) * x_latent

            x_latent = self.p_k_sample(x_latent, cond, sigmas, sampler, s_in=s_in, i=i, mode=mode,
                                       unconditional_guidance_scale=unconditional_guidance_scale,
                                       unconditional_conditioning=unconditional_conditioning,
                                       model_wrap_sigmas=model_wrap_sigmas, ds=ds)
            if callback:
                callback(x_latent)
        if mode == "pix2pix":
            x_latent = x_latent[:, :4, :, :]
        return x_latent

    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])

    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / self.append_dims(sigma, x.ndim)

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
        return x[(...,) + (None,) * dims_to_append]

    def get_ancestral_step(self, sigma_from, sigma_to, eta=1.):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        if not eta:
            return sigma_to, 0.
        sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up


class UNetV2(UNet):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 force_null_conditioning=False,
                 *args, **kwargs):
        if cond_stage_config == '__is_unconditional__' and not self.force_null_conditioning:
            conditioning_key = None
        super(UNetV2, self).__init__(
            num_timesteps_cond=num_timesteps_cond,
            cond_stage_key=cond_stage_key,
            cond_stage_trainable=cond_stage_trainable,
            concat_mode=concat_mode,
            cond_stage_forward=cond_stage_forward,
            conditioning_key=conditioning_key,
            scale_factor=scale_factor,
            scale_by_std=scale_by_std,
            force_null_conditioning=force_null_conditioning,
            unetConfigEncode=None,
            unetConfigDecode=None,
            *args,
            **kwargs
        )
        ckpt_path = kwargs.pop("ckpt_path", None)
        reset_ema = kwargs.pop("reset_ema", False)
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.v1 = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
            if reset_ema:
                assert self.use_ema
                print(
                    f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def make_schedule_plms(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps, verbose=verbose)
        alphas_cumprod = self.alphas_cumprod
        to_torch = lambda x: x.to(self.cdevice)

        self.register_buffer('betas', to_torch(self.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
