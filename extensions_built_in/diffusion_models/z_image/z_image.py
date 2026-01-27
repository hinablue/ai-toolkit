import os
from typing import List, Optional

import huggingface_hub
import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager
from safetensors.torch import load_file

from transformers import AutoTokenizer, Qwen3ForCausalLM, BitsAndBytesConfig
from diffusers import AutoencoderKL

try:
    from diffusers import ZImagePipeline
    from diffusers.models.transformers import ZImageTransformer2DModel
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )


scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}

# FP16 stability patches
# Ref: https://github.com/kohya-ss/musubi-tuner/pull/795

def clamp_fp16(x):
    """
    Clamps the tensor to the valid range of FP16 to avoid Inf/NaN.
    Typical max value for FP16 is 65504.
    """
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def apply_z_image_fp16_patches(model):
    """
    Applies Forward Hooks and Pre-Hooks to Z-Image modules to enforce FP16 clamping.
    This mimics the stability fixes in the Musubi Tuner implementation without rewriting the model definition.
    """
    print("Applying proper FP16 clamping patches to ZImage Transformer")

    def clamp_output_hook(module, args, output):
        # Clamps the module output
        return clamp_fp16(output)

    def clamp_input_pre_hook(module, args):
        # Clamps the first argument of the module input
        if len(args) > 0:
            x = args[0]
            # Verify it's a tensor before clamping (though usually it is for these layers)
            if isinstance(x, torch.Tensor):
                return (clamp_fp16(x),) + args[1:]
        return args

    # Helper to patch a standard ZImage block
    def patch_block(block):
        # 1. Patch Attention Output
        # Prevents NaN propagation into the residual connection
        if hasattr(block, "attention"):
            block.attention.register_forward_hook(clamp_output_hook)

        # 2. Patch FeedForward
        if hasattr(block, "feed_forward"):
            # Patch FFN Output
            block.feed_forward.register_forward_hook(clamp_output_hook)

            # Patch FFN Internal (before w2/down_proj)
            # In the reference implementation: return self.w2(clamp_fp16(F.silu(self.w1(x)) * x3))
            # We hook into w2 to clamp its input.
            if hasattr(block.feed_forward, "w2"):
                block.feed_forward.w2.register_forward_pre_hook(clamp_input_pre_hook)

    # Patch Main Layers
    if hasattr(model, "layers"):
        for layer in model.layers:
            patch_block(layer)

    # Patch Noise Refiner Layers (if present)
    if hasattr(model, "noise_refiner"):
        for layer in model.noise_refiner:
            patch_block(layer)

    # Patch Context Refiner Layers (if present)
    if hasattr(model, "context_refiner"):
        for layer in model.context_refiner:
            patch_block(layer)


class ZImageModel(BaseModel):
    arch = "zimage"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["ZImageTransformer2DModel"]

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2  # 16 for the VAE, 2 for patch size

    def load_training_adapter(self, transformer: ZImageTransformer2DModel):
        self.print_and_status_update("Loading assistant LoRA")
        lora_path = self.model_config.assistant_lora_path
        if not os.path.exists(lora_path):
            # assume it is a hub path
            lora_splits = lora_path.split("/")
            if len(lora_splits) != 3:
                raise ValueError(
                    f"Assistant LoRA path {lora_path} is not a valid local path or hub path."
                )
            repo_id = "/".join(lora_splits[:2])
            filename = lora_splits[2]
            try:
                lora_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                )
                # upgrade path to
                self.model_config.assistant_lora_path = lora_path
            except Exception as e:
                raise ValueError(
                    f"Failed to download assistant LoRA from {lora_path}: {e}"
                )
        # load the adapter and merge it in. We will inference with a -1.0 multiplier so the adapter effects only work during training.
        lora_state_dict = load_file(lora_path)
        dim = int(
            lora_state_dict[
                "diffusion_model.layers.0.attention.to_k.lora_A.weight"
            ].shape[0]
        )

        new_sd = {}
        for key, value in lora_state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        lora_state_dict = new_sd

        network_config = {
            "type": "lora",
            "linear": dim,
            "linear_alpha": dim,
            "transformer_only": True,
        }

        network_config = NetworkConfig(**network_config)
        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            is_assistant_adapter=True,
            is_ara=True,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        self.print_and_status_update("Merging in assistant LoRA")
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)

        network.merge_in(merge_weight=1.0)

        # mark it as not merged so inference ignores it.
        network.is_merged_in = False

        # add the assistant so sampler will activate it while sampling
        self.assistant_lora: LoRASpecialNetwork = network

        # deactivate lora during training
        self.assistant_lora.multiplier = -1.0
        self.assistant_lora.is_active = False

        # tell the model to invert assistant on inference since we want remove lora effects
        self.invert_assistant_lora = True

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading ZImage model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        self.print_and_status_update("Loading transformer")

        transformer_path = model_path
        transformer_subfolder = "transformer"
        if os.path.exists(transformer_path):
            transformer_subfolder = None
            transformer_path = os.path.join(transformer_path, "transformer")
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, "text_encoder")
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        transformer = ZImageTransformer2DModel.from_pretrained(
            transformer_path, subfolder=transformer_subfolder, torch_dtype=dtype
        )

        # Apply FP16 stability patches (hooks) only if using fp16
        if dtype == torch.float16:
            apply_z_image_fp16_patches(transformer)

        # load assistant lora if specified
        if self.model_config.assistant_lora_path is not None:
            self.load_training_adapter(transformer)
            # set qtype to be float8 if it is qfloat8
            if self.model_config.qtype == "qfloat8":
                self.model_config.qtype = "float8"

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[
                    transformer.x_pad_token,
                    transformer.cap_pad_token,
                ]
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype
        )

        te_kwargs = {}
        if self.model_config.quantize_te and self.model_config.qtype_te == "nf4":
            self.print_and_status_update("Loading Text Encoder in 4-bit (nf4)")
            te_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            te_kwargs["device_map"] = self.device_torch

        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype, **te_kwargs
        )

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        if self.model_config.quantize_te and self.model_config.qtype_te != "nf4":
            text_encoder.to(self.device_torch, dtype=dtype)
            flush()

            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))

        if self.model_config.quantize_te:
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype
        )

        self.noise_scheduler = ZImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        pipe: ZImagePipeline = ZImagePipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # just to make sure everything is on the right device and dtype
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = ZImageModel.get_train_scheduler()

        pipeline: ZImagePipeline = ZImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: ZImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        timestep_model_input = (1000 - timestep) / 1000

        text_embeds = text_embeddings.text_embeds
        if isinstance(text_embeds, torch.Tensor):
            if len(text_embeds.shape) == 3:
                # if it is a single batch tensor, unbind it into a list of tensors
                text_embeds = list(text_embeds.unbind(dim=0))
        elif isinstance(text_embeds, list):
            # check if items are rank 3 (batch, length, dim)
            if len(text_embeds[0].shape) == 3:
                # flatten the list of batches into a single list of tensors
                new_text_embeds = []
                for t in text_embeds:
                    if t is not None:
                        new_text_embeds += list(t.unbind(dim=0))
                text_embeds = new_text_embeds

        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_model_input,
            text_embeds,
        )[0]

        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=self.device_torch,
        )
        # encode_prompt returns list of rank-2 tensors [seq_len, dim]
        # Pad to same length and stack into rank-3 tensor [batch, seq_len, dim]
        # for compatibility with concat_prompt_embeds and predict_noise
        if isinstance(prompt_embeds, list):
            # Find max sequence length, TODO: Or just use 512?
            max_seq_len = max(t.shape[0] for t in prompt_embeds)
            print(f"max_seq_len: {max_seq_len}")
            # Pad each tensor to max length
            padded = []
            for t in prompt_embeds:
                if t.shape[0] < max_seq_len:
                    pad = torch.zeros(
                        (max_seq_len - t.shape[0], t.shape[1]),
                        dtype=t.dtype,
                        device=t.device,
                    )
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            prompt_embeds = torch.stack(padded, dim=0)
        pe = PromptEmbeds([prompt_embeds, None])
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: ZImageTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "zimage"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
