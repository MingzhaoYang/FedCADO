import argparse
import hashlib
import itertools
import logging
import pdb

import math
import os
import warnings
from pathlib import Path
from typing import Optional
import random

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision.utils import save_image

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import matplotlib.pyplot as plt
import torchvision
import copy
from utils import DeepInversionHook
from diffusers.utils import randn_tensor
from diffusers.models.embeddings import Timesteps
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
from datasets.openimage import get_openimage_classes
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
        
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        torch_dtype=torch.float16,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--category",
        type=int,
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of a sks urban scene",
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output',
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_text_encoder",
        default=True,
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=200,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default='fp16',
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help=(
            "The size of the crop to use for training. If the image is smaller than the crop, it will be padded."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    args.output_dir = os.path.join("logs/checkpoints", args.output_dir)
    return args

class MyLinear(torch.nn.Module):
    def __init__(self,init_embeddings,classes=345):
        super().__init__()
        
        #papa = init_embeddings.repeat(classes,1,1).half()
        papa = init_embeddings.repeat(1,1,1).half()
        self.papa =  torch.nn.Parameter(papa)
    def forward(self,classes):
        return self.papa[classes]


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
    
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False, torch_dtype=torch.float16)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
            torch_dtype=torch.float16,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=torch.float16)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=torch.float16)
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, torch_dtype=torch.float16
    )
    
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="safety_checker", revision=args.revision, torch_dtype=torch.float16
    )
    
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision, torch_dtype=torch.float16
    )
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    
    class ClientTune(nn.Module):
        def __init__(self, classes=345):
            super(ClientTune, self).__init__()
            self.encoder = torchvision.models.resnet18(pretrained=True)
            self.encoder.fc = torch.nn.Identity()
            self.final_proj = nn.Sequential(
                nn.Linear(512,classes)
            )

        def forward(self, x,timestep=None):   
            #oriout = self.encoder(x)
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)

            x = self.encoder.layer2(x)
                   
            x = self.encoder.layer3(x)
                   
            x = self.encoder.layer4(x)
    
            x = self.encoder.avgpool(x)
            x = self.encoder.fc(x)
            output = self.final_proj(x.squeeze())
            return output
        
    # Scheduler and math around the number of training steps.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True 
    generator = torch.Generator()
    # Prepare everything with our `accelerator`.
    unet, text_encoder,tokenizer,generator,noise_scheduler,vae,safety_checker,feature_extractor = accelerator.prepare(
        unet, text_encoder,tokenizer,generator,noise_scheduler,vae,safety_checker,feature_extractor)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16
    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    path = r"/home/share/DomainNet/clipart"
    f = os.listdir(path)
    for i in range(len(f)):
        f[i] = 'an image of '+f[i].lower()
    class_prompts = sorted(f)   
    print(class_prompts[args.category])
    #if not os.path.exists(f'/home/share/gen_data_nips/sybn_domainnet_90_noaddnoise_img30/'+class_prompts[args.category]):
     #   os.makedirs(f'/home/share/gen_data_nips/sybn_domainnet_90_noaddnoise_img30/'+class_prompts[args.category])
    
    
    
    #nicopp_path = "/home/share/NICOpp/NICO_DG/autumn"
    #f = os.listdir(nicopp_path)
    #for i in range(len(f)):
    #    f[i] = 'an image of ' + f[i].lower()
    #class_prompts = sorted(f) 
    #print(class_prompts)
    #print(aaa)
    #print(class_prompts[args.category])
    
    
    #open_image_class_prompts,open_image_rough_classes = get_openimage_classes()
    #class_prompts = open_image_rough_classes
    if not os.path.exists(f'/home/share/gen_data_nips/test_noedit/'+class_prompts[args.category]):
        os.makedirs(f'/home/share/gen_data_nips/test_noedit/'+class_prompts[args.category])
        
        
    
    uncond_inputs = tokenizer(
        '',
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",)
    uncond_input_ids = uncond_inputs.input_ids
    uncond_embeddings = text_encoder(uncond_input_ids.to(unet.device))[0]
    
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    #domains = ['clipart', 'infograph', 'painting', 'real', 'sketch']
    client_num= 6 
    
    seed = 0
    
    generator = generator.manual_seed(args.category+seed)
    train_epoch = [30,30,30,30,30,30]
    #train_epoch = [5,5,5,5,5,5]
    for client in range(client_num):
        model = ClientTune(classes=90)     
        client_name = int(args.category/10)
        path = '/home/yangmingzhao/2023_3/NIPS_train_prop/output/domainnet_90_noaddnoise_img30_'+domains[client]+'_epoch_90.tar'
        print(path)
        load_state_dict(model , torch.load(path,map_location='cpu'), prefix='module.')
        model = model.half()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model = accelerator.prepare(model)   
        #print(model)
        #for k, v in model.state_dict().items():
        #    print(k)
        
        #bn_model = ClientTune(classes=60)   
        #path = '/home/yangmingzhao/2023_3/NIPS_train_prop/output/nicou_noaddnoise_img30_None_'+str(client)+'_epoch_99_class_'+str(args.category)+'.tar'
        #load_state_dict(bn_model , torch.load(path,map_location='cpu'), prefix='')
        #bn_model = bn_model.half()
        #bn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bn_model) 
        #bn_model = accelerator.prepare(bn_model)  
        
        
        hooks = []
        #for m in bn_model.modules():
        for m in model.modules():
            #if isinstance(m, nn.BatchNorm2d):
            if isinstance(m, nn.SyncBatchNorm):
                hooks.append(DeepInversionHook(m))
                
        model.eval()
        #bn_model.eval()
        
        
        unet.eval()
        safety_checker.eval()
        text_encoder.eval()
        
        step = args.category
        text_inputs = tokenizer(
            class_prompts[step],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = text_encoder(text_input_ids.to(unet.device))[0]      
        
        for epoch in tqdm(range(1)):
            xxxx = 0
            img_num = 0
            while img_num < train_epoch[client] and xxxx < 500:
                for repeat in range(1):
                    torch.cuda.empty_cache()

                    with accelerator.accumulate(unet):
                        torch.cuda.empty_cache()
                        # Convert images to latent space
                        latents_shape = (1, unet.in_channels, 64, 64)
                        latents_dtype = weight_dtype
                        latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(unet.device)
                        latents = latents * noise_scheduler.init_noise_sigma

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        #timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=latents.device)
                        #timesteps = timesteps.long()
                        noise_scheduler.set_timesteps(50)
                        timesteps_tensor = noise_scheduler.timesteps.to(latents.device)
                        
                        for aaaa in range(10):
                            timesteps = timesteps_tensor[0].repeat(bsz)
                            # if timesteps[0].data<sss[-1]: continue
                            # print()
                            # Add noise to the latents according to the noise magnitude at each timestep
                            # (this is the forward diffusion process)
                            # Get the text embedding for conditioning
                            # print(encoder_hidden_states.shape)
                            text_embeddings = text_embeddings.detach().requires_grad_(True)
                            uncond_embeddings = uncond_embeddings.detach().requires_grad_(True)
                            latents = latents.detach().requires_grad_(True)
                            
                            input_embeddings = torch.cat([uncond_embeddings,text_embeddings],dim=0)  
                            
                            latent_model_input = torch.cat([latents] * 2)
                            #if timesteps > sss[0]:
                            #    with torch.no_grad():
                            #        model_preds = unet(latent_model_input, timesteps, input_embeddings).sample.half()
                            #else:
                            model_preds = unet(latent_model_input, timesteps, input_embeddings).sample.half()
                            uncond_pred, text_pred = model_preds.chunk(2)
                            model_pred = uncond_pred + 3*(text_pred-uncond_pred)
                            ori_latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).pred_original_sample.half()
                            #pre_latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).prev_sample.half()
                            #latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).prev_sample.half()
                            input_latents = 1/0.18215*ori_latents
                            image = vae.decode(input_latents).sample
                            image = (image / 2 + 0.5).clamp(0, 1)
                            #torchvision.utils.save_image(image, '/home/share/gen_data_nips/domainnet_allbn_imgnum_40_noise_clsfier/'+class_prompts[args.category]+'/train_prop'+str(client)+'_'+str(xxxx)+'_'+str(aaaa)+'before_change.jpg')
                            #image = torch.nn.AdaptiveAvgPool2d((224,224))(image)
                            trans = transforms.Compose([
                                transforms.Resize(
                                    (224, 224),
                                    interpolation=transforms.InterpolationMode.BICUBIC,
                                    antialias=False,
                                ),
                                transforms.Normalize(
                                    [0.48145466, 0.4578275, 0.40821073],
                                    [0.26862954, 0.26130258, 0.27577711]),
                            ])
                            output = model(trans(image),timesteps.repeat(bsz))
                            
                            #_ = bn_model(trans(image),timesteps.repeat(bsz))
                            
                            loss_bn = sum([h.r_feature for h in hooks]) # bn层loss
                            #loss_bn = torch.tensor(0)
                            loss_oh = F.cross_entropy(output.unsqueeze(0), torch.tensor(step).repeat(bsz).to(unet.device))  # ce_loss  
                            #loss = loss_oh
                            loss =  0.1 * loss_bn
                            cond_grad = torch.autograd.grad(loss, latents)[0]
                            #cond_grad = torch.autograd.grad(loss, model_pred)[0]
                            latents = latents -  cond_grad
                            
                            print("aaaa",aaaa,"loss", loss_oh.detach().item(),loss_bn.detach().item())
                            '''
                            #ori_latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).pred_original_sample.half()
                            #pre_latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).prev_sample.half()
                            latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).prev_sample.half()
                            input_latents = 1/0.18215*latents
                            image = vae.decode(input_latents).sample
                            image = (image / 2 + 0.5).clamp(0, 1)
                            torchvision.utils.save_image(image, '/home/share/gen_data_nips/test_noedit/'+class_prompts[args.category]+'/train_prop'+str(client)+'_'+str(xxxx)+'_'+str(aaaa)+'after_change.jpg')
                            '''
                        #sss = timesteps_tensor[-10:]
                        for timesteps in timesteps_tensor[:-1]:
                            timesteps = timesteps.repeat(bsz)
                            # if timesteps[0].data<sss[-1]: continue
                            # print()
                            # Add noise to the latents according to the noise magnitude at each timestep
                            # (this is the forward diffusion process)
                            # Get the text embedding for conditioning
                            # print(encoder_hidden_states.shape)
                            text_embeddings = text_embeddings.detach().requires_grad_(True)
                            uncond_embeddings = uncond_embeddings.detach().requires_grad_(True)
                            latents = latents.detach().requires_grad_(True)
                            
                            input_embeddings = torch.cat([uncond_embeddings,text_embeddings],dim=0)  
                            
                            
                            latent_model_input = torch.cat([latents] * 2)
                            #if timesteps > sss[0]:
                            #    with torch.no_grad():
                            #        model_preds = unet(latent_model_input, timesteps, input_embeddings).sample.half()
                            #else:
                            model_preds = unet(latent_model_input, timesteps, input_embeddings).sample.half()
                            uncond_pred, text_pred = model_preds.chunk(2)
                            model_pred = uncond_pred + 3*(text_pred-uncond_pred)
                            
                            ori_latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).pred_original_sample.half()
                            #pre_latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).prev_sample.half()
                            #latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).prev_sample.half()
                            input_latents = 1/0.18215*ori_latents
                            image = vae.decode(input_latents).sample
                            image = (image / 2 + 0.5).clamp(0, 1)
                            #torchvision.utils.save_image(image, '/home/share/gen_data_nips/domainnet_allbn_imgnum_40_noise_clsfier/'+class_prompts[args.category]+'/train_prop'+str(client)+'_'+str(xxxx)+'_'+str(timesteps.item())+'before_change.jpg')
                            #image = torch.nn.AdaptiveAvgPool2d((224,224))(image)
                            trans = transforms.Compose([
                                transforms.Resize(
                                    (224, 224),
                                    interpolation=transforms.InterpolationMode.BICUBIC,
                                    antialias=False,
                                ),
                                transforms.Normalize(
                                    [0.48145466, 0.4578275, 0.40821073],
                                    [0.26862954, 0.26130258, 0.27577711]),
                            ])
                            output = model(trans(image),timesteps.repeat(bsz))
                            
                            #_ = bn_model(trans(image),timesteps.repeat(bsz))
                            
                            loss_bn = sum([h.r_feature for h in hooks]) # bn层loss
                            #loss_bn = torch.tensor(0)
                            loss_oh = F.cross_entropy(output.unsqueeze(0), torch.tensor(step).repeat(bsz).to(unet.device))  # ce_loss  
                            #loss = loss_oh
                            loss = loss_oh + 0.1 * loss_bn
                            cond_grad = torch.autograd.grad(loss, latents)[0]
                            #cond_grad = torch.autograd.grad(loss, model_pred)[0]
                            #print((1 - noise_scheduler.alphas_cumprod[timesteps].cuda().half()).sqrt())
                            latents = latents - cond_grad
                            
                            print("timestep",int(timesteps),"loss", loss_oh.detach().item(),loss_bn.detach().item())
                            
                            '''
                            ori_latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).pred_original_sample.half()
                            #pre_latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).prev_sample.half()
                            #latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).prev_sample.half()
                            input_latents = 1/0.18215*ori_latents
                            image = vae.decode(input_latents).sample
                            image = (image / 2 + 0.5).clamp(0, 1)
                            torchvision.utils.save_image(image, '/home/share/gen_data_nips/domainnet_allbn_imgnum_40_noise_clsfier/'+class_prompts[args.category]+'/train_prop'+str(client)+'_'+str(xxxx)+'_'+str(timesteps.item())+'after_change.jpg')
                            '''
                            
                            with torch.no_grad():
                                latents = latents.detach().requires_grad_(False)
                                
                                latent_model_input = torch.cat([latents] * 2)
                                model_preds = unet(latent_model_input, timesteps, input_embeddings).sample.half()
                                uncond_pred, text_pred = model_preds.chunk(2)
                                model_pred = uncond_pred + 3*(text_pred-uncond_pred)
                                latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).prev_sample.half()
                            
                                #input_latents = 1/0.18215*latents
                                #image = vae.decode(input_latents).sample
                                #image = (image / 2 + 0.5).clamp(0, 1)
                                #torchvision.utils.save_image(image, '/home/share/gen_data_nips/test_noedit/'+class_prompts[args.category]+'/train_prop'+str(client)+'_'+str(xxxx)+'_'+str(timesteps.item())+'after_change.jpg')
                                                     
                            #loss = loss_oh
                            #accelerator.backward(loss)
                            unet.zero_grad()
                            vae.zero_grad()
                            
                            model.zero_grad()
                           # bn_model.zero_grad()
                            
                            torch.cuda.empty_cache()
                            
                    if xxxx%1==0:
                        with torch.no_grad():
                            ori_latents = noise_scheduler.step(model_pred, timesteps, latents,generator = generator).pred_original_sample.half()
                            input_latents = 1/0.18215*ori_latents.detach()
                            image = vae.decode(input_latents).sample
                            image = (image / 2 + 0.5).clamp(0, 1)
                            np_image = image.clone()
                            np_image = np_image.cpu().permute(0, 2, 3, 1).float().numpy()
                            safety_checker_input = feature_extractor(numpy_to_pil(np_image), return_tensors="pt").to(unet.device)
                            np_image, nsfw_content_detected = safety_checker(
                                images=np_image, clip_input=safety_checker_input.pixel_values.to(latents_dtype)
                            )
                            print("client",client,"category",step,"img_num",img_num,"xxxx",xxxx,"repeat",repeat,"timestep",int(timesteps),"pred_cls",torch.argmax(output.unsqueeze(0)).item(),"loss", loss_oh.detach().item(),loss_bn.detach().item())
                            #print("client",client,"category",step,"img_num",img_num,"xxxx",xxxx,"repeat",repeat,"timestep",int(timesteps))
                           # if nsfw_content_detected[0] == False and loss_oh <= 0.5 and torch.argmax(output.unsqueeze(0)) == step:
                            #if nsfw_content_detected[0] == False:
                            torchvision.utils.save_image(image, '/home/share/gen_data_nips/test_noedit/'+class_prompts[args.category]+'/train_prop'+str(client)+'_'+str(img_num)+'.jpg')
                            img_num = img_num + 1
                            xxxx = xxxx + 1
                            #else :
                                #xxxx = xxxx + 1
        
                
        
        
    # Create the pipeline using using the trained modules and save it.
    #torch.save(props,'prop.tar')
    accelerator.wait_for_everyone()
    accelerator.end_training()



if __name__ == "__main__":
    args = parse_args()
    main(args)