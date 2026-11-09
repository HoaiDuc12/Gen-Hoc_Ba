# -*- coding: utf-8 -*-
"""
Package Augment: Hệ thống sinh ảnh biến dị (Data Augmentation) cho học bạ THPT.
"""

from .pipeline import AugmentConfig, AugmentationPipeline, EffectOption, get_preset_config
from .camera import apply_perspective_transform, apply_lens_blur, apply_sensor_noise_and_jpeg, generate_desk_background
from .lighting import apply_flash_glare, apply_shadow_gradient, apply_vignette
from .fold import apply_folds
from .wrinkle import apply_wrinkles
from .blur import apply_local_blur, apply_ink_bleed
from .stain import apply_stains
from .scanner import apply_camscanner, apply_photocopy

__all__ = [
    "AugmentConfig",
    "AugmentationPipeline",
    "EffectOption",
    "get_preset_config",
    "apply_perspective_transform",
    "apply_lens_blur",
    "apply_sensor_noise_and_jpeg",
    "generate_desk_background",
    "apply_flash_glare",
    "apply_shadow_gradient",
    "apply_vignette",
    "apply_folds",
    "apply_wrinkles",
    "apply_local_blur",
    "apply_ink_bleed",
    "apply_stains",
    "apply_camscanner",
    "apply_photocopy",
]

