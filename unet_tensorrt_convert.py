import torch
import os
from .trt_common import TRT_MODEL_CONVERSION_BASE

class UNET_TENSORRT_CONVERTER(TRT_MODEL_CONVERSION_BASE):
    def __init__(self):
        super(UNET_TENSORRT_CONVERTER, self).__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_SDXL"}),
                "batch_size_min": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "batch_size_opt": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "batch_size_max": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "height_min": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height_opt": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height_max": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_min": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_opt": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_max": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "context_min": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "context_opt": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "context_max": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
            },
        }

    def convert(self, model, filename_prefix, batch_size_min, batch_size_opt, batch_size_max, height_min, height_opt, height_max, width_min, width_opt, width_max, context_min, context_opt, context_max):
        return super()._convert(model, filename_prefix, batch_size_min, batch_size_opt, batch_size_max, height_min, height_opt, height_max, width_min, width_opt, width_max, context_min, context_opt, context_max)

NODE_CLASS_MAPPINGS = {
    "UNET_TENSORRT_CONVERTER": UNET_TENSORRT_CONVERTER,
}
