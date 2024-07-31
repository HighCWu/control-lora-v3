from typing import Any, Dict, List, Optional, Tuple, Union

import copy
import torch
from torch import nn, svd_lowrank

from peft.tuners.lora import LoraLayer, Conv2d as PeftConv2d
from diffusers.configuration_utils import register_to_config
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput, UNet2DConditionModel as UNet2DConditionModel


class UNet2DConditionModelEx(UNet2DConditionModel):
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
        extra_condition_names: List[str] = [],
    ):
        num_extra_conditions = len(extra_condition_names)
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels * (1 + num_extra_conditions),
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,)
        self._internal_dict = copy.deepcopy(self._internal_dict)
        self.config.in_channels = in_channels
        self.config.extra_condition_names = extra_condition_names
    
    @property
    def extra_condition_names(self) -> List[str]:
        return self.config.extra_condition_names

    def add_extra_conditions(self, extra_condition_names: Union[str, List[str]]):
        if isinstance(extra_condition_names, str):
            extra_condition_names = [extra_condition_names]
        conv_in_kernel = self.config.conv_in_kernel
        conv_in_weight = self.conv_in.weight
        self.config.extra_condition_names += extra_condition_names
        full_in_channels = self.config.in_channels * (1 + len(self.config.extra_condition_names))
        new_conv_in_weight = torch.zeros(
            conv_in_weight.shape[0], full_in_channels, conv_in_kernel, conv_in_kernel,
            dtype=conv_in_weight.dtype,
            device=conv_in_weight.device,)
        new_conv_in_weight[:,:conv_in_weight.shape[1]] = conv_in_weight
        self.conv_in.weight = nn.Parameter(
            new_conv_in_weight.data,
            requires_grad=conv_in_weight.requires_grad,)
        self.conv_in.in_channels = full_in_channels
        
        return self
    
    def activate_extra_condition_adapters(self):
        lora_layers = [layer for layer in self.modules() if isinstance(layer, LoraLayer)]
        if len(lora_layers) > 0:
            self._hf_peft_config_loaded = True
        for lora_layer in lora_layers:
            adapter_names = [k for k in lora_layer.scaling.keys() if k in self.config.extra_condition_names] 
            adapter_names += lora_layer.active_adapters
            adapter_names = list(set(adapter_names))
            lora_layer.set_adapter(adapter_names)
    
    def set_extra_condition_scale(self, scale: Union[float, List[float]] = 1.0):
        if isinstance(scale, float):
            scale = [scale] * len(self.config.extra_condition_names)

        lora_layers = [layer for layer in self.modules() if isinstance(layer, LoraLayer)]
        for s, n in zip(scale, self.config.extra_condition_names):
            for lora_layer in lora_layers:
                lora_layer.set_scale(n, s)
    
    @property
    def default_half_lora_target_modules(self) -> List[str]:
        module_names = []
        for name, module in self.named_modules():
            if "conv_out" in name or "up_blocks" in name:
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module_names.append(name)
        return list(set(module_names))
    
    @property
    def default_full_lora_target_modules(self) -> List[str]:
        module_names = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module_names.append(name)
        return list(set(module_names))
    
    @property
    def default_half_skip_attn_lora_target_modules(self) -> List[str]:
        return [
            module_name
            for module_name in self.default_half_lora_target_modules 
            if all(
                not module_name.endswith(attn_name) 
                for attn_name in 
                ["to_k", "to_q", "to_v", "to_out.0"]
            )
        ]
    
    @property
    def default_full_skip_attn_lora_target_modules(self) -> List[str]:
        return [
            module_name
            for module_name in self.default_full_lora_target_modules 
            if all(
                not module_name.endswith(attn_name) 
                for attn_name in 
                ["to_k", "to_q", "to_v", "to_out.0"]
            )
        ]

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        extra_conditions: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        if extra_conditions is not None:
            if isinstance(extra_conditions, list):
                extra_conditions = torch.cat(extra_conditions, dim=1)
            sample = torch.cat([sample, extra_conditions], dim=1)
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,)


class PeftConv2dEx(PeftConv2d):
    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        
        if isinstance(init_lora_weights, str) and "pissa" in init_lora_weights.lower():
            if self.conv2d_pissa_init(adapter_name, init_lora_weights):
                return
            # Failed
            init_lora_weights = "gaussian"

        super(PeftConv2d, self).reset_lora_parameters(adapter_name, init_lora_weights)

    def conv2d_pissa_init(self, adapter_name, init_lora_weights):
        weight = weight_ori = self.get_base_layer().weight
        weight = weight.flatten(start_dim=1)
        if self.r[adapter_name] > weight.shape[0]:
            return False
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)

        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A.view([-1] + list(weight_ori.shape[1:]))
        self.lora_B[adapter_name].weight.data = lora_B.view([-1, self.r[adapter_name]] + [1] * (weight_ori.ndim - 2))
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight.view_as(weight_ori)
        
        return True


# Patch peft conv2d
PeftConv2d.reset_lora_parameters = PeftConv2dEx.reset_lora_parameters
PeftConv2d.conv2d_pissa_init = PeftConv2dEx.conv2d_pissa_init
