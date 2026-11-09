# -*- coding: utf-8 -*-
"""
Module Pipeline: Điều phối quy trình Augmentation cho dữ liệu học bạ THPT.
Hỗ trợ:
- Điều khiển bật/tắt từng hiệu ứng, cường độ (0.0..1.0), xác suất (0.0..1.0).
- Chế độ 'random' (ngẫu nhiên hoá theo xác suất) hoặc 'fixed' (áp dụng cố định).
- Bộ Presets: 'scan_clean', 'phone_photo', 'aged_document', 'photocopy'.
- Tự động biến đổi đồng bộ toạ độ Bounding Box 4 điểm (Polygon) khi có biến đổi hình học.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .camera import apply_perspective_transform, apply_lens_blur, apply_sensor_noise_and_jpeg
from .lighting import apply_flash_glare, apply_shadow_gradient, apply_vignette
from .fold import apply_folds
from .wrinkle import apply_wrinkles
from .blur import apply_local_blur, apply_ink_bleed
from .stain import apply_stains
from .scanner import apply_camscanner, apply_photocopy


@dataclass
class EffectOption:
    """Cấu hình cho 1 hiệu ứng đơn lẻ."""
    enabled: bool = True
    intensity: float = 0.5   # Cường độ: 0.0 .. 1.0
    probability: float = 1.0 # Xác suất xuất hiện: 0.0 .. 1.0


@dataclass
class AugmentConfig:
    """Bộ cấu hình toàn diện cho Pipeline Augmentation."""
    mode: str = "random"     # 'random' (ngẫu nhiên theo xác suất) hoặc 'fixed' (cố định)

    # 1. Biến đổi hình học & chụp ảnh (Camera & Geometry)
    perspective: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.45, probability=0.8))
    bg_style: str = "wood"   # 'wood' (bàn gỗ) hoặc 'office' (bàn văn phòng xám)
    lens_blur: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.35, probability=0.6))
    sensor_noise: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.30, probability=0.7))
    jpeg_quality: int = 78   # Chất lượng nén JPEG (30..95)

    # 2. Ánh sáng & bóng (Lighting)
    flash_glare: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.40, probability=0.5))
    shadow: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.45, probability=0.7))
    vignette: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.35, probability=0.6))

    # 3. Tình trạng giấy vật lý (Paper degradation)
    folds: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.50, probability=0.6))
    wrinkles: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.40, probability=0.5))
    stains: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.45, probability=0.4))

    # 4. Mực & độ nét chữ (Ink & Blur)
    local_blur: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.45, probability=0.5))
    ink_bleed: EffectOption = field(default_factory=lambda: EffectOption(enabled=True, intensity=0.35, probability=0.6))

    # 5. Chế độ quét & photocopy (Scanner mode)
    camscanner: EffectOption = field(default_factory=lambda: EffectOption(enabled=False, intensity=0.85, probability=0.0))
    camscanner_mode: str = "magic_color" # 'magic_color' hoặc 'bw'
    photocopy: EffectOption = field(default_factory=lambda: EffectOption(enabled=False, intensity=0.75, probability=0.0))


def get_preset_config(preset_name: str) -> AugmentConfig:
    """
    Trả về cấu hình mẫu theo preset chuẩn:
    - 'scan_clean': Scan CamScanner phẳng phiu, sạch sẽ, nền trắng bóc, không nghiêng.
    - 'phone_photo': Chụp điện thoại trên bàn gỗ nghiêng góc xiên, bóng đổ, flash nhẹ.
    - 'aged_document': Học bạ cũ nhiều nếp gấp, nhăn giấy, ố vàng, mờ vài ô điểm.
    - 'photocopy': Bản sao photocopy đen trắng độ tương phản cao, có vệt mép gáy sổ.
    """
    cfg = AugmentConfig()

    if preset_name == "scan_clean":
        cfg.mode = "fixed"
        cfg.perspective.enabled = False
        cfg.lens_blur.enabled = False
        cfg.flash_glare.enabled = False
        cfg.shadow.enabled = False
        cfg.vignette.enabled = False
        cfg.folds.enabled = False
        cfg.wrinkles.enabled = False
        cfg.stains.enabled = False
        cfg.local_blur.enabled = False
        cfg.ink_bleed.enabled = False
        cfg.sensor_noise.enabled = False
        cfg.jpeg_quality = 92
        cfg.camscanner.enabled = True
        cfg.camscanner.intensity = 0.90
        cfg.camscanner.probability = 1.0
        cfg.camscanner_mode = "magic_color"

    elif preset_name == "phone_photo":
        cfg.mode = "random"
        cfg.perspective.enabled = True
        cfg.perspective.intensity = 0.50
        cfg.perspective.probability = 1.0
        cfg.bg_style = "wood"
        cfg.shadow.enabled = True
        cfg.shadow.intensity = 0.50
        cfg.shadow.probability = 0.85
        cfg.flash_glare.enabled = True
        cfg.flash_glare.intensity = 0.40
        cfg.flash_glare.probability = 0.60
        cfg.vignette.enabled = True
        cfg.vignette.intensity = 0.40
        cfg.vignette.probability = 0.75
        cfg.folds.enabled = True
        cfg.folds.intensity = 0.40
        cfg.folds.probability = 0.65
        cfg.wrinkles.enabled = False
        cfg.stains.enabled = False
        cfg.lens_blur.enabled = True
        cfg.lens_blur.intensity = 0.35
        cfg.lens_blur.probability = 0.70
        cfg.sensor_noise.enabled = True
        cfg.sensor_noise.intensity = 0.35
        cfg.sensor_noise.probability = 0.85
        cfg.jpeg_quality = 75
        cfg.camscanner.enabled = False
        cfg.photocopy.enabled = False

    elif preset_name == "aged_document":
        cfg.mode = "random"
        cfg.perspective.enabled = True
        cfg.perspective.intensity = 0.25
        cfg.perspective.probability = 0.5
        cfg.bg_style = "office"
        cfg.folds.enabled = True
        cfg.folds.intensity = 0.70
        cfg.folds.probability = 1.0
        cfg.wrinkles.enabled = True
        cfg.wrinkles.intensity = 0.60
        cfg.wrinkles.probability = 0.90
        cfg.stains.enabled = True
        cfg.stains.intensity = 0.65
        cfg.stains.probability = 0.85
        cfg.ink_bleed.enabled = True
        cfg.ink_bleed.intensity = 0.50
        cfg.ink_bleed.probability = 0.80
        cfg.local_blur.enabled = True
        cfg.local_blur.intensity = 0.55
        cfg.local_blur.probability = 0.70
        cfg.shadow.enabled = True
        cfg.shadow.intensity = 0.35
        cfg.shadow.probability = 0.60
        cfg.flash_glare.enabled = False
        cfg.vignette.enabled = True
        cfg.vignette.intensity = 0.45
        cfg.vignette.probability = 0.80
        cfg.sensor_noise.enabled = True
        cfg.sensor_noise.intensity = 0.40
        cfg.sensor_noise.probability = 0.80
        cfg.jpeg_quality = 68
        cfg.camscanner.enabled = False
        cfg.photocopy.enabled = False

    elif preset_name == "photocopy":
        cfg.mode = "fixed"
        cfg.perspective.enabled = False
        cfg.lens_blur.enabled = False
        cfg.flash_glare.enabled = False
        cfg.shadow.enabled = False
        cfg.vignette.enabled = False
        cfg.folds.enabled = True
        cfg.folds.intensity = 0.35
        cfg.folds.probability = 0.6
        cfg.wrinkles.enabled = False
        cfg.stains.enabled = False
        cfg.local_blur.enabled = False
        cfg.ink_bleed.enabled = False
        cfg.sensor_noise.enabled = False
        cfg.camscanner.enabled = False
        cfg.photocopy.enabled = True
        cfg.photocopy.intensity = 0.85
        cfg.photocopy.probability = 1.0
        cfg.jpeg_quality = 80

    return cfg


class AugmentationPipeline:
    """
    Quy trình Augmentation toàn diện cho ảnh học bạ THPT.
    """

    def __init__(self, config: Optional[AugmentConfig] = None):
        self.config = config or AugmentConfig()

    def _should_apply(self, opt: EffectOption, rng: np.random.RandomState) -> Tuple[bool, float]:
        """Xác định hiệu ứng có được kích hoạt không và cường độ thực tế."""
        if not opt.enabled:
            return False, 0.0

        if self.config.mode == "fixed":
            return True, opt.intensity

        # Chế độ Random: tung xúc xắc theo xác suất
        if rng.uniform(0.0, 1.0) <= opt.probability:
            # Biến thiên nhẹ cường độ để đa dạng tập huấn luyện
            rand_intensity = rng.uniform(opt.intensity * 0.6, opt.intensity * 1.1)
            return True, float(np.clip(rand_intensity, 0.05, 1.0))
        return False, 0.0

    def process(
        self,
        image: np.ndarray,
        annotations: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Thực hiện chuỗi augmentation lên ảnh và đồng bộ nhãn annotations.

        Args:
            image: Ảnh học bạ (BGR numpy array).
            annotations: Danh sách ground truth nhãn (chứa 'bbox' polygon 4 điểm).
            seed: Random seed tái lập kết quả.

        Returns:
            (augmented_image, updated_annotations)
        """
        rng = np.random.RandomState(seed)
        current_img = image.copy()
        current_ann = [dict(ann) for ann in (annotations or [])]

        # -------------------------------------------------------------
        # BƯỚC 1: CÁC BIẾN ĐỔI HÌNH HỌC (Trước tiên - cập nhật Bounding Box)
        # -------------------------------------------------------------
        apply_p, p_int = self._should_apply(self.config.perspective, rng)
        if apply_p:
            p_seed = int(rng.randint(0, 1_000_000))
            current_img, current_ann, _ = apply_perspective_transform(
                current_img,
                annotations=current_ann,
                intensity=p_int,
                bg_style=self.config.bg_style,
                seed=p_seed
            )

        # -------------------------------------------------------------
        # BƯỚC 2: CÁC BIẾN ĐỔI BỀ MẶT GIẤY VẬT LÝ (Giấy nhăn, Nếp gấp)
        # -------------------------------------------------------------
        apply_wrk, wrk_int = self._should_apply(self.config.wrinkles, rng)
        if apply_wrk:
            wrk_seed = int(rng.randint(0, 1_000_000))
            current_img = apply_wrinkles(current_img, intensity=wrk_int, seed=wrk_seed)

        apply_fld, fld_int = self._should_apply(self.config.folds, rng)
        if apply_fld:
            fld_seed = int(rng.randint(0, 1_000_000))
            current_img = apply_folds(current_img, intensity=fld_int, seed=fld_seed)

        # -------------------------------------------------------------
        # BƯỚC 3: MỰC & MỜ CỤC BỘ (Loang mực bút bi, Mờ ô điểm)
        # -------------------------------------------------------------
        apply_bld, bld_int = self._should_apply(self.config.ink_bleed, rng)
        if apply_bld:
            bld_seed = int(rng.randint(0, 1_000_000))
            current_img = apply_ink_bleed(current_img, intensity=bld_int, seed=bld_seed)

        apply_lblur, lblur_int = self._should_apply(self.config.local_blur, rng)
        if apply_lblur:
            lblur_seed = int(rng.randint(0, 1_000_000))
            current_img = apply_local_blur(
                current_img,
                annotations=current_ann,
                intensity=lblur_int,
                seed=lblur_seed
            )

        # -------------------------------------------------------------
        # BƯỚC 4: VẾT Ố & BẨN (Ố vàng, Cà phê, Nước loang)
        # -------------------------------------------------------------
        apply_stn, stn_int = self._should_apply(self.config.stains, rng)
        if apply_stn:
            stn_seed = int(rng.randint(0, 1_000_000))
            current_img = apply_stains(current_img, intensity=stn_int, seed=stn_seed)

        # -------------------------------------------------------------
        # BƯỚC 5: HIỆU ỨNG QUÉT HOẶC PHOTOCOPY (Nếu bật)
        # -------------------------------------------------------------
        apply_pc, pc_int = self._should_apply(self.config.photocopy, rng)
        if apply_pc:
            pc_seed = int(rng.randint(0, 1_000_000))
            current_img = apply_photocopy(current_img, intensity=pc_int, seed=pc_seed)
        else:
            apply_cs, cs_int = self._should_apply(self.config.camscanner, rng)
            if apply_cs:
                cs_seed = int(rng.randint(0, 1_000_000))
                current_img = apply_camscanner(
                    current_img,
                    intensity=cs_int,
                    mode=self.config.camscanner_mode,
                    seed=cs_seed
                )

        # -------------------------------------------------------------
        # BƯỚC 6: ÁNH SÁNG & CHỤP ẢNH (Bóng đổ, Chói flash, Vignette)
        # -------------------------------------------------------------
        apply_shd, shd_int = self._should_apply(self.config.shadow, rng)
        if apply_shd:
            shd_seed = int(rng.randint(0, 1_000_000))
            current_img = apply_shadow_gradient(current_img, intensity=shd_int, seed=shd_seed)

        apply_glr, glr_int = self._should_apply(self.config.flash_glare, rng)
        if apply_glr:
            glr_seed = int(rng.randint(0, 1_000_000))
            current_img = apply_flash_glare(current_img, intensity=glr_int, seed=glr_seed)

        apply_vig, vig_int = self._should_apply(self.config.vignette, rng)
        if apply_vig:
            current_img = apply_vignette(current_img, intensity=vig_int)

        # -------------------------------------------------------------
        # BƯỚC 7: MỜ TIÊU CỰ, NHIỄU CẢM BIẾN & NÉN JPEG
        # -------------------------------------------------------------
        apply_lens, lens_int = self._should_apply(self.config.lens_blur, rng)
        if apply_lens:
            lens_seed = int(rng.randint(0, 1_000_000))
            current_img = apply_lens_blur(current_img, intensity=lens_int, gradient=True, seed=lens_seed)

        apply_noise, noise_int = self._should_apply(self.config.sensor_noise, rng)
        noise_level = noise_int if apply_noise else 0.0
        cam_seed = int(rng.randint(0, 1_000_000))
        current_img = apply_sensor_noise_and_jpeg(
            current_img,
            noise_intensity=noise_level,
            jpeg_quality=self.config.jpeg_quality,
            seed=cam_seed
        )

        return current_img, current_ann

