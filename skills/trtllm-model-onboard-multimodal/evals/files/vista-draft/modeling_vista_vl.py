# Draft of tensorrt_llm/_torch/models/modeling_vista_vl.py (synthetic, for review only).
import copy

import torch
from PIL import Image
from torch import nn
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from ...inputs import BaseMultimodalInputProcessor
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import register_auto_model, register_input_processor


class VistaVLInputProcessor(BaseMultimodalInputProcessor):

    def __init__(self, model_path, config, tokenizer, trust_remote_code=True, **kwargs):
        super().__init__(**kwargs)
        self.processor = self._load_hf_processor(model_path, trust_remote_code)
        self.config = config

    def __call__(self, inputs, sampling_params):
        image = Image.open(inputs["multi_modal_data"]["image"][0]).convert("RGB")
        processed = self.processor(text=inputs["prompt"], images=[image], return_tensors="pt")
        multimodal_data = {
            "image": {
                "pixel_values": processed["pixel_values"].cuda(),
                "image_grid_thw": processed["image_grid_thw"].cuda(),
            }
        }
        return processed["input_ids"][0].tolist(), {"multimodal_data": multimodal_data}


class VistaVLProjector(nn.Module):

    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.fc1 = nn.Linear(vision_dim, text_dim)
        self.fc2 = nn.Linear(text_dim, text_dim)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x))))


@register_input_processor(VistaVLInputProcessor, model_type="vista_vl")
@register_auto_model("VistaVLForConditionalGeneration")
class VistaVLModel(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        config = model_config.pretrained_config
        self.image_token_id = config.image_token_id
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.projector = VistaVLProjector(config.vision_config.out_hidden_size,
                                          config.text_config.hidden_size)
        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = config.text_config
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

    def forward(self, attn_metadata, input_ids, position_ids, multimodal_params=None,
                pixel_values=None, **kwargs):
        mm_embeds = []
        if multimodal_params is not None and len(multimodal_params) > 0:
            for param in multimodal_params:
                pv = param.multimodal_data["image"]["pixel_values"]
                if pv is None:
                    continue
                grid = param.multimodal_data["image"]["image_grid_thw"]
                mm_embeds.append(self.projector(self.visual(pv, grid_thw=grid)))

        num_image_tokens = int((input_ids == self.image_token_id).sum().item())
        mm_positions = torch.nonzero(input_ids == self.image_token_id)
        if num_image_tokens == 0:
            mm_embeds = []
        lengths = torch.tensor([e.shape[0] for e in mm_embeds], device="cuda")
        assert lengths.sum().item() == mm_positions.shape[0]

        input_ids, inputs_embeds = fuse_input_embeds(self.llm.model.embed_tokens, input_ids,
                                                     mm_embeds)
        return self.llm.forward(attn_metadata, input_ids, position_ids,
                                inputs_embeds=inputs_embeds)
