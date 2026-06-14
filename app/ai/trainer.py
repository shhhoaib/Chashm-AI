import cv2
import numpy as np
from pathlib import Path
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

from app.ai.engine import AIAnalysisEngine, PREPROCESS, _get_device
from app.config import TRAINED_DIR

IMG_SIZE = 224
NUM_CLASSES = 10
EPOCHS_DEFAULT = 5
BATCH_SIZE_DEFAULT = 8
LR_DEFAULT = 0.001
TYPE_NUM_CLASSES = 5
EXTERNAL_NUM_CLASSES = 15

TYPE_LABELS = ["fundus", "external_eye", "slit_lamp", "oct", "unknown"]

EXTERNAL_LABELS = [
    "normal", "conjunctivitis_bacterial", "conjunctivitis_viral",
    "conjunctivitis_allergic", "subconjunctival_hemorrhage", "pterygium",
    "pinguecula", "hordeolum", "chalazion", "scleritis",
    "scleral_icterus", "corneal_ulcer", "blepharitis",
    "keratitis", "dry_eye_external",
]

# ─────────────────────────────────────────────
#  TYPE CLASSIFIER SYNTHETIC DATA
# ─────────────────────────────────────────────

class _TypeImageGenerator:
    def __init__(self, size: int = IMG_SIZE):
        self.size = size

    def _random_bg(self):
        color = random.randint(10, 60)
        return np.full((self.size, self.size, 3), (color, color, color), dtype=np.uint8)

    def _random_noise(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.randint(0, 8, img.shape, dtype=np.uint8)
        return cv2.add(img, noise)

    def _fundus(self) -> np.ndarray:
        img = self._random_bg()
        cx, cy = self.size // 2, self.size // 2
        r = self.size // 2 - random.randint(5, 15)
        fundus_color = (
            random.randint(30, 80),
            random.randint(50, 100),
            random.randint(70, 130),
        )
        cv2.circle(img, (cx, cy), r, fundus_color, -1)
        inner = int(r * 0.85)
        lighter = (
            min(fundus_color[0] + 20, 120),
            min(fundus_color[1] + 20, 140),
            min(fundus_color[2] + 20, 170),
        )
        cv2.circle(img, (cx, cy), inner, lighter, -1)

        disc_x = cx + int(r * random.uniform(0.08, 0.22))
        disc_y = cy - int(r * random.uniform(0.05, 0.15))
        cv2.circle(img, (disc_x, disc_y), int(r * random.uniform(0.10, 0.18)),
                   (random.randint(140, 190), random.randint(160, 210), random.randint(180, 240)), -1)

        mac_x = cx - int(r * random.uniform(0.15, 0.25))
        mac_y = cy + int(r * random.uniform(0.02, 0.08))
        cv2.circle(img, (mac_x, mac_y), int(r * random.uniform(0.05, 0.10)),
                   (random.randint(50, 80), random.randint(60, 90), random.randint(70, 100)), -1)

        for _ in range(random.randint(20, 40)):
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(int(r * 0.3), int(r * 0.6))
            x1 = disc_x + random.randint(-8, 8)
            y1 = disc_y + random.randint(-8, 8)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            if 0 <= x2 < self.size and 0 <= y2 < self.size:
                cv2.line(img, (x1, y1), (x2, y2),
                         (random.randint(30, 60), random.randint(20, 40), random.randint(60, 90)),
                         random.randint(1, 2))

        if random.random() < 0.4:
            blur = random.uniform(0.5, 1.5)
            img = cv2.GaussianBlur(img, (5, 5), blur)

        return self._random_noise(img)

    def _external_eye(self) -> np.ndarray:
        img = _skin_texture(self.size)
        cx, cy = self.size // 2, self.size // 2 + random.randint(-3, 3)
        eye_w = random.randint(int(self.size * 0.35), int(self.size * 0.55))
        eye_h = random.randint(int(self.size * 0.15), int(self.size * 0.28))
        _draw_eye_shape(img, cx, cy, eye_w, eye_h)
        noise = np.random.randint(-4, 5, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.4:
            img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.8))
        return img

    def _slit_lamp(self) -> np.ndarray:
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img[:, :] = (random.randint(5, 25), random.randint(5, 25), random.randint(5, 25))

        beam_w = random.randint(int(self.size * 0.06), int(self.size * 0.15))
        beam_h = random.randint(int(self.size * 0.35), int(self.size * 0.60))
        beam_x = self.size // 2 + random.randint(-15, 15)
        beam_y = self.size // 2 - beam_h // 2 + random.randint(-10, 10)

        beam_bright = random.randint(200, 245)
        cv2.rectangle(img, (beam_x - beam_w // 2, beam_y),
                      (beam_x + beam_w // 2, beam_y + beam_h),
                      (beam_bright, beam_bright, beam_bright - random.randint(0, 20)), -1)

        for i in range(beam_h):
            y = beam_y + i
            if 0 <= y < self.size:
                grad = random.randint(-8, 8)
                v = max(0, min(255, beam_bright + grad))
                for dx in range(-beam_w // 2, beam_w // 2):
                    px = beam_x + dx
                    if 0 <= px < self.size:
                        img[y, px] = (v, v, max(0, v - random.randint(0, 15)))

        for _ in range(random.randint(5, 15)):
            sx = beam_x + random.randint(-2, 2)
            sy = beam_y + random.randint(0, beam_h)
            cv2.circle(img, (sx, sy), random.randint(1, 3),
                       (250, 250, 240), -1)

        if random.random() < 0.3:
            cx, cy = self.size // 2, self.size // 2
            for i in range(5, 15):
                alpha = 0.03
                cv2.circle(img, (cx, cy), int(self.size * 0.4) + i,
                           (random.randint(10, 30),) * 3, 1)

        return self._random_noise(img)

    def _oct(self) -> np.ndarray:
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        bg = random.randint(20, 40)
        img[:, :] = (bg, bg, bg)

        oct_w = random.randint(int(self.size * 0.55), int(self.size * 0.85))
        oct_h = random.randint(int(self.size * 0.30), int(self.size * 0.50))
        ox = self.size // 2 - oct_w // 2
        oy = self.size // 2 - oct_h // 2 + random.randint(-10, 10)

        for layer_i in range(oct_h):
            y = oy + layer_i
            if y >= self.size:
                break
            layer_pct = layer_i / oct_h
            if layer_pct < 0.15:
                c = random.randint(180, 230)
                color = (c, c - random.randint(0, 10), c - random.randint(0, 15))
            elif layer_pct < 0.25:
                c = random.randint(80, 140)
                color = (c, c, c)
            elif layer_pct < 0.40:
                c = random.randint(40, 80)
                color = (c - random.randint(0, 10), c, c + random.randint(0, 10))
            elif layer_pct < 0.70:
                c = random.randint(60, 100)
                color = (c, c + random.randint(0, 5), c + random.randint(0, 10))
            else:
                c = random.randint(30, 50)
                color = (c, c, c + random.randint(0, 5))

            waviness = 3 * np.sin(2 * np.pi * layer_i / oct_h * random.uniform(0.5, 2.0))
            for px in range(oct_w):
                px2 = ox + px
                if px2 >= self.size:
                    break
                offset = int(waviness * np.sin(px / oct_w * np.pi * 2))
                yy = y + offset
                if 0 <= yy < self.size:
                    noise_v = random.randint(-3, 3)
                    img[yy, px2] = (
                        max(0, min(255, color[0] + noise_v)),
                        max(0, min(255, color[1] + noise_v)),
                        max(0, min(255, color[2] + noise_v)),
                    )

        if random.random() < 0.4:
            cx, cy = self.size // 2, self.size // 2
            cv2.rectangle(img, (cx - 30, 0), (cx + 30, self.size - 1), (bg - 5, bg - 5, bg - 5), -1)

        return self._random_noise(img)

    def _unknown(self) -> np.ndarray:
        choice = random.randint(0, 3)
        if choice == 0:
            img = np.random.randint(0, 255, (self.size, self.size, 3), dtype=np.uint8)
            return cv2.GaussianBlur(img, (15, 15), 5)
        elif choice == 1:
            color = (random.randint(50, 200),) * 3
            img = np.full((self.size, self.size, 3), color, dtype=np.uint8)
            for _ in range(random.randint(5, 20)):
                x1, y1 = random.randint(0, self.size), random.randint(0, self.size)
                x2, y2 = random.randint(0, self.size), random.randint(0, self.size)
                cv2.line(img, (x1, y1), (x2, y2),
                         (random.randint(0, 255),) * 3, random.randint(1, 3))
            return img
        elif choice == 2:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            for _ in range(random.randint(3, 8)):
                cx = random.randint(0, self.size)
                cy = random.randint(0, self.size)
                r = random.randint(10, self.size // 3)
                cv2.circle(img, (cx, cy), r,
                           (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)), -1)
            return img
        else:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            freq_x = random.uniform(0.01, 0.1)
            freq_y = random.uniform(0.01, 0.1)
            for y in range(self.size):
                for x in range(self.size):
                    v = int(128 + 64 * np.sin(freq_x * x + freq_y * y))
                    img[y, x] = (v, v, v)
            return img

    def generate(self, type_idx: int) -> np.ndarray:
        generators = [self._fundus, self._external_eye, self._slit_lamp, self._oct, self._unknown]
        return generators[type_idx]()


# ─────────────────────────────────────────────
#  EXTERNAL EYE DISEASE GENERATOR
# ─────────────────────────────────────────────

def _perlin_noise(size: int, scale: float = 10.0) -> np.ndarray:
    x = np.linspace(0, scale, size, endpoint=False)
    y = np.linspace(0, scale, size, endpoint=False)
    X, Y = np.meshgrid(x, y)
    noise = np.sin(X) * np.cos(Y) + np.sin(X * 1.7 + Y * 0.5) * 0.5 + np.cos(X * 0.3 - Y * 2.1) * 0.3
    noise = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)
    return noise


def _skin_texture(size: int) -> np.ndarray:
    base_r = random.randint(140, 220)
    base_g = random.randint(110, 180)
    base_b = random.randint(80, 140)
    skin = np.full((size, size, 3), (base_b, base_g, base_r), dtype=np.uint8)

    noise = _perlin_noise(size, random.uniform(6, 14))
    for c in range(3):
        skin[:, :, c] = np.clip(skin[:, :, c].astype(np.int16) + (noise.astype(np.int16) - 128) * random.randint(2, 5), 0, 255).astype(np.uint8)

    tiny = np.random.randint(-3, 4, (size, size, 3), dtype=np.int16)
    skin = np.clip(skin.astype(np.int16) + tiny, 0, 255).astype(np.uint8)
    return skin


def _draw_eye_shape(img: np.ndarray, cx: int, cy: int, eye_w: int, eye_h: int,
                    sclera_color=None, iris_color=None) -> dict:
    h, w = img.shape[:2]
    if sclera_color is None:
        sclera_color = (random.randint(220, 245), random.randint(220, 245), random.randint(220, 245))

    lid_top_y = cy - eye_h - random.randint(2, 6)
    lid_bottom_y = cy + eye_h + random.randint(2, 6)

    # Sclera
    sclera_pts = np.array([
        [cx - eye_w, cy],
        [cx - int(eye_w * 0.7), cy - eye_h],
        [cx, cy - eye_h - 2],
        [cx + int(eye_w * 0.7), cy - eye_h],
        [cx + eye_w, cy],
        [cx + int(eye_w * 0.7), cy + eye_h],
        [cx, cy + eye_h + 2],
        [cx - int(eye_w * 0.7), cy + eye_h],
    ], np.int32)
    cv2.fillPoly(img, [sclera_pts], sclera_color)

    # Sclera shading (slight pink at corners)
    for corner_x, corner_y in [(cx - eye_w + 5, cy), (cx + eye_w - 5, cy)]:
        for r in range(15, 5, -3):
            alpha = random.uniform(0.02, 0.06)
            overlay = img.copy()
            cv2.circle(overlay, (corner_x, corner_y), r,
                       (random.randint(180, 200), random.randint(150, 180), random.randint(190, 210)), -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Tiny blood vessels in sclera
    for _ in range(random.randint(3, 8)):
        vx = cx + random.randint(-eye_w + 5, eye_w - 5)
        vy = cy + random.randint(-eye_h + 5, eye_h - 5)
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(5, 20)
        vx2 = int(vx + length * np.cos(angle))
        vy2 = int(vy + length * np.sin(angle))
        if 0 <= vx2 < w and 0 <= vy2 < h:
            cv2.line(img, (vx, vy), (vx2, vy2),
                     (random.randint(150, 180), random.randint(120, 150), random.randint(180, 210)), 1)

    # Iris
    iris_r = random.randint(int(eye_h * 0.55), int(eye_h * 0.75))
    if iris_color is None:
        iris_color = (
            random.randint(30, 100),
            random.randint(40, 80),
            random.randint(40, 70),
        )
    cv2.circle(img, (cx, cy), iris_r, iris_color, -1)

    # Iris texture - radial lines
    for angle_deg in range(0, 360, random.randint(3, 8)):
        a = np.radians(angle_deg)
        r1 = iris_r * random.uniform(0.3, 0.5)
        r2 = iris_r * random.uniform(0.7, 0.95)
        x1 = int(cx + r1 * np.cos(a))
        y1 = int(cy + r1 * np.sin(a))
        x2 = int(cx + r2 * np.cos(a))
        y2 = int(cy + r2 * np.sin(a))
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            cv2.line(img, (x1, y1), (x2, y2), (
                max(0, iris_color[0] - random.randint(10, 30)),
                max(0, iris_color[1] - random.randint(10, 30)),
                max(0, iris_color[2] - random.randint(10, 30)),
            ), 1)

    # Iris limbal ring
    cv2.circle(img, (cx, cy), iris_r, (
        max(0, iris_color[0] - 20),
        max(0, iris_color[1] - 20),
        max(0, iris_color[2] - 15),
    ), 2)

    # Pupil
    pupil_r = random.randint(int(iris_r * 0.35), int(iris_r * 0.55))
    cv2.circle(img, (cx, cy), pupil_r, (random.randint(5, 15),) * 3, -1)

    # Corneal light reflex
    reflex_x = cx + random.randint(-int(pupil_r * 0.3), int(pupil_r * 0.3))
    reflex_y = cy - random.randint(int(pupil_r * 0.1), int(pupil_r * 0.3))
    reflex_r = max(1, int(pupil_r * random.uniform(0.15, 0.3)))
    cv2.circle(img, (reflex_x, reflex_y), reflex_r, (240, 240, 245), -1)
    if reflex_r > 2:
        cv2.circle(img, (reflex_x - 1, reflex_y - 1), max(1, reflex_r - 1), (250, 250, 255), -1)

    # Eyebrows / upper lid shadow
    lid_shade = np.zeros_like(img)
    cv2.ellipse(lid_shade, (cx, cy - eye_h + 2), (eye_w + 5, eye_h // 3), 0, 0, 180,
                (random.randint(50, 80), random.randint(40, 60), random.randint(30, 50)), -1)
    lid_shade = cv2.GaussianBlur(lid_shade, (9, 9), 4)
    mask = lid_shade > 0
    img[mask] = cv2.addWeighted(img, 0.6, lid_shade, 0.4, 0)[mask]

    # Lower lid
    cv2.ellipse(img, (cx, cy + eye_h), (eye_w - 2, eye_h // 4), 0, 0, 360,
                (random.randint(60, 90), random.randint(50, 70), random.randint(40, 60)), 1)

    # Eyelashes (upper lid)
    num_lashes = random.randint(8, 18)
    for _ in range(num_lashes):
        lx = cx + random.randint(-eye_w + 5, eye_w - 5)
        ly = lid_top_y + random.randint(-2, 2)
        l_angle = -np.pi / 2 + random.uniform(-0.3, 0.3)
        l_len = random.randint(4, 12)
        lx2 = int(lx + l_len * np.cos(l_angle))
        ly2 = int(ly + l_len * np.sin(l_angle))
        if 0 <= lx2 < w and 0 <= ly2 < h:
            cv2.line(img, (lx, ly), (lx2, ly2),
                     (random.randint(15, 35), random.randint(10, 25), random.randint(10, 20)), 1)

    return {"cx": cx, "cy": cy, "eye_w": eye_w, "eye_h": eye_h,
            "iris_r": iris_r, "pupil_r": pupil_r,
            "lid_top_y": lid_top_y, "lid_bottom_y": lid_bottom_y}


class _ExternalEyeGenerator:
    def __init__(self, size: int = IMG_SIZE):
        self.size = size

    def _create_base_eye(self, disease_modifier=None) -> np.ndarray:
        size = self.size
        img = _skin_texture(size)

        cx, cy = size // 2, size // 2 + random.randint(-3, 3)
        eye_w = random.randint(int(size * 0.35), int(size * 0.55))
        eye_h = random.randint(int(size * 0.18), int(size * 0.28))

        sclera_color = (random.randint(220, 245), random.randint(220, 245), random.randint(220, 245))
        iris_color = (
            random.randint(30, 120),
            random.randint(40, 100),
            random.randint(50, 80),
        )

        if disease_modifier:
            sclera_color, iris_color = disease_modifier(sclera_color, iris_color)

        eye_info = _draw_eye_shape(img, cx, cy, eye_w, eye_h, sclera_color, iris_color)

        if random.random() < 0.3:
            blur = random.uniform(0.3, 1.2)
            img = cv2.GaussianBlur(img, (3, 3), blur)

        noise = np.random.randint(-4, 5, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def _modify_conjunctivitis(self, sclera, iris):
        r = random.randint(35, 75)
        g = random.randint(20, 50)
        b = random.randint(5, 25)
        sclera = (max(0, sclera[0] - r), max(0, sclera[1] - g), max(0, sclera[2] - b))
        return sclera, iris

    def _add_vessels(self, img, cx, cy, eye_w, eye_h, count=15):
        h, w = img.shape[:2]
        for _ in range(count):
            theta = random.uniform(0, 2 * np.pi)
            rad = random.uniform(0, eye_w * random.uniform(0.3, 0.8))
            x = int(cx + rad * np.cos(theta))
            y = int(cy + rad * np.sin(theta))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), random.randint(1, 2),
                           (random.randint(100, 150), random.randint(80, 120), random.randint(160, 200)), -1)

    def _normal(self) -> np.ndarray:
        return self._create_base_eye()

    def _conjunctivitis_bacterial(self) -> np.ndarray:
        img = self._create_base_eye(self._modify_conjunctivitis)
        cx, cy = self.size // 2, self.size // 2
        self._add_vessels(img, cx, cy, self.size // 3, self.size // 5, count=random.randint(12, 25))
        for _ in range(random.randint(3, 7)):
            dx = random.randint(-35, 35)
            dy = random.randint(-20, 20)
            cv2.circle(img, (cx + dx, cy + dy), random.randint(4, 12),
                       (random.randint(170, 210), random.randint(140, 180), random.randint(100, 140)), -1)
        return img

    def _conjunctivitis_viral(self) -> np.ndarray:
        img = self._create_base_eye(self._modify_conjunctivitis)
        cx, cy = self.size // 2, self.size // 2
        eye_w = self.size // 3
        self._add_vessels(img, cx, cy, eye_w, self.size // 5, count=random.randint(8, 18))
        for _ in range(random.randint(2, 5)):
            dx = random.randint(-25, 25)
            dy = random.randint(-15, 15)
            cv2.circle(img, (cx + dx, cy + dy), random.randint(1, 3),
                       (random.randint(180, 210), random.randint(180, 210), random.randint(180, 210)), -1)
        return img

    def _conjunctivitis_allergic(self) -> np.ndarray:
        def mod(sclera, iris):
            r = random.randint(20, 45)
            g = random.randint(15, 35)
            b = random.randint(5, 15)
            sclera = (max(0, sclera[0] - r), max(0, sclera[1] - g), max(0, sclera[2] - b))
            return sclera, iris
        img = self._create_base_eye(mod)
        cx, cy = self.size // 2, self.size // 2
        self._add_vessels(img, cx, cy, self.size // 3, self.size // 5, count=random.randint(5, 12))
        # Chemosis / swelling
        if random.random() < 0.5:
            for _ in range(random.randint(2, 5)):
                dx = random.randint(-15, 15)
                dy = random.randint(-10, 10)
                cv2.circle(img, (cx + dx, cy + dy), random.randint(4, 8),
                           (random.randint(190, 210), random.randint(180, 200), random.randint(170, 190)), -1)
        return img

    def _subconjunctival_hemorrhage(self) -> np.ndarray:
        img = self._create_base_eye()
        cx, cy = self.size // 2, self.size // 2
        rx = random.randint(15, 50)
        ry = random.randint(8, 30)
        hem_color = (random.randint(20, 60), random.randint(20, 45), random.randint(180, 240))
        x_off = random.randint(-15, 15)
        y_off = random.randint(-8, 8)
        angle = random.uniform(-20, 20)
        cv2.ellipse(img, (cx + x_off, cy + y_off), (rx, ry), angle, 0, 360, hem_color, -1)
        # Irregular borders
        for _ in range(random.randint(2, 5)):
            dx = random.randint(-rx + 3, rx - 3)
            dy = random.randint(-ry + 3, ry - 3)
            if abs(dx) > rx * 0.3 or abs(dy) > ry * 0.3:
                cv2.circle(img, (cx + x_off + dx, cy + y_off + dy), random.randint(2, 6),
                           (max(0, hem_color[0] - 10), max(0, hem_color[1] - 10), min(255, hem_color[2] + 10)), -1)
        return img

    def _pterygium(self) -> np.ndarray:
        img = self._create_base_eye()
        cx, cy = self.size // 2, self.size // 2
        side = random.choice([-1, 1])
        tip_x = cx + side * random.randint(30, 50)
        tip_y = cy + random.randint(-8, 8)
        base_y = random.randint(12, 22)
        base_x = cx + side * random.randint(5, 15)
        pts = np.array([
            [tip_x, tip_y],
            [base_x, cy - base_y - random.randint(0, 5)],
            [base_x, cy + base_y + random.randint(0, 5)],
        ], np.int32)
        flesh_color = (random.randint(160, 200), random.randint(150, 190), random.randint(140, 180))
        cv2.fillPoly(img, [pts], flesh_color)
        # Vessels on pterygium
        for _ in range(random.randint(2, 5)):
            px = tip_x + int((base_x - tip_x) * random.uniform(0.2, 0.8))
            py = tip_y + random.randint(-5, 5)
            cv2.line(img, (px, py), (px - side * random.randint(3, 8), py + random.randint(-2, 2)),
                     (random.randint(120, 150), random.randint(100, 130), random.randint(100, 130)), 1)
        return img

    def _pinguecula(self) -> np.ndarray:
        img = self._create_base_eye()
        cx, cy = self.size // 2, self.size // 2
        for side in [-1, 1]:
            if random.random() < 0.7:
                nx = cx + side * random.randint(25, 45)
                ny = cy + random.randint(-8, 8)
                cv2.circle(img, (nx, ny), random.randint(4, 10),
                           (random.randint(180, 210), random.randint(170, 200), random.randint(160, 190)), -1)
                cv2.circle(img, (nx, ny), random.randint(3, 8),
                           (random.randint(190, 220), random.randint(180, 210), random.randint(170, 200)), -1)
        return img

    def _hordeolum(self) -> np.ndarray:
        img = self._create_base_eye()
        cx = self.size // 2 + random.randint(-15, 15)
        top_y = random.randint(5, 25)
        r = random.randint(8, 16)
        cv2.circle(img, (cx, top_y), r,
                   (random.randint(30, 60), random.randint(25, 50), random.randint(180, 230)), -1)
        cv2.circle(img, (cx, top_y), r + 2,
                   (random.randint(20, 40), random.randint(15, 35), random.randint(150, 200)), 1)
        # Pus point
        if random.random() < 0.6:
            cv2.circle(img, (cx + random.randint(-2, 2), top_y + random.randint(-2, 2)),
                       random.randint(2, 4), (random.randint(200, 235), random.randint(200, 235), random.randint(200, 235)), -1)
        return img

    def _chalazion(self) -> np.ndarray:
        img = self._create_base_eye()
        cx = self.size // 2 + random.randint(-12, 12)
        top_y = random.randint(10, 30)
        r = random.randint(6, 14)
        cv2.circle(img, (cx, top_y), r,
                   (random.randint(120, 160), random.randint(110, 150), random.randint(100, 140)), -1)
        cv2.circle(img, (cx, top_y), r + 1,
                   (random.randint(100, 140), random.randint(90, 130), random.randint(80, 120)), 1)
        return img

    def _scleritis(self) -> np.ndarray:
        def mod(sclera, iris):
            r = random.randint(20, 50)
            g = random.randint(15, 35)
            b = random.randint(5, 15)
            sclera = (max(0, sclera[0] - r), max(0, sclera[1] - g), max(0, sclera[2] - b))
            return sclera, iris
        img = self._create_base_eye(mod)
        cx, cy = self.size // 2, self.size // 2
        self._add_vessels(img, cx, cy, self.size // 3, self.size // 5, count=random.randint(15, 30))
        for _ in range(random.randint(2, 5)):
            dx = random.randint(-20, 20)
            dy = random.randint(-15, 15)
            cv2.circle(img, (cx + dx, cy + dy), random.randint(4, 8),
                       (random.randint(40, 70), random.randint(30, 55), random.randint(180, 220)), -1)
        return img

    def _scleral_icterus(self) -> np.ndarray:
        def mod(sclera, iris):
            r = max(0, sclera[0] - random.randint(10, 30))
            g = max(0, sclera[1] - random.randint(30, 50))
            b = max(0, sclera[2] - random.randint(50, 80))
            sclera = (r, g, b)
            return sclera, iris
        img = self._create_base_eye(mod)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0].astype(np.int16) + random.randint(15, 30), 0, 179).astype(np.uint8)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) + random.randint(20, 50), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def _corneal_ulcer(self) -> np.ndarray:
        img = self._create_base_eye()
        cx, cy = self.size // 2, self.size // 2
        for _ in range(random.randint(1, 2)):
            dx = random.randint(-12, 12)
            dy = random.randint(-10, 10)
            cv2.circle(img, (cx + dx, cy + dy), random.randint(6, 14),
                       (random.randint(190, 230), random.randint(190, 230), random.randint(190, 230)), -1)
            cv2.circle(img, (cx + dx, cy + dy), random.randint(7, 16),
                       (random.randint(170, 210), random.randint(170, 210), random.randint(170, 210)), 1)
        return img

    def _blepharitis(self) -> np.ndarray:
        img = self._create_base_eye()
        h, w = img.shape[:2]
        cx = w // 2
        # Upper lid margin inflammation
        for x in range(cx - 60, cx + 60, random.randint(8, 15)):
            for _ in range(random.randint(1, 3)):
                dx = random.randint(-3, 3)
                dy = random.randint(0, 12)
                px = min(max(x + dx, 0), w - 1)
                py = min(dy, h - 1)
                cv2.circle(img, (px, py), random.randint(2, 5),
                           (random.randint(180, 220), random.randint(140, 180), random.randint(120, 160)), -1)
        # Lower lid margin inflammation
        for x in range(cx - 55, cx + 55, random.randint(10, 18)):
            for _ in range(random.randint(1, 2)):
                dx = random.randint(-3, 3)
                dy = random.randint(-12, 0)
                px = min(max(x + dx, 0), w - 1)
                py = min(h - 1 + dy, h - 1)
                if py >= 0:
                    cv2.circle(img, (px, py), random.randint(2, 4),
                               (random.randint(180, 220), random.randint(140, 180), random.randint(120, 160)), -1)
        return img

    def _keratitis(self) -> np.ndarray:
        img = self._create_base_eye()
        cx, cy = self.size // 2, self.size // 2
        for _ in range(random.randint(2, 4)):
            dx = random.randint(-12, 12)
            dy = random.randint(-12, 12)
            cv2.circle(img, (cx + dx, cy + dy), random.randint(3, 6),
                       (random.randint(200, 235), random.randint(200, 235), random.randint(200, 235)), -1)
            cv2.circle(img, (cx + dx, cy + dy), random.randint(2, 5),
                       (random.randint(180, 210), random.randint(180, 210), random.randint(180, 210)), -1)
        return img

    def _dry_eye_external(self) -> np.ndarray:
        def mod(sclera, iris):
            r = max(0, sclera[0] - random.randint(5, 15))
            g = max(0, sclera[1] - random.randint(5, 15))
            b = max(0, sclera[2] - random.randint(5, 15))
            return (r, g, b), iris
        img = self._create_base_eye(mod)
        h, w = img.shape[:2]
        # Dry spots / tear film breakup
        for _ in range(random.randint(8, 20)):
            x = random.randint(int(w * 0.2), int(w * 0.8))
            y = random.randint(int(h * 0.3), int(h * 0.7))
            cv2.circle(img, (x, y), random.randint(1, 3),
                       (random.randint(130, 160), random.randint(130, 160), random.randint(130, 160)), -1)
        # Overall dullness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) - random.randint(5, 15), 0, 255).astype(np.uint8)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int16) - random.randint(5, 20), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def generate(self, disease_idx: int) -> np.ndarray:
        generators = [
            self._normal, self._conjunctivitis_bacterial, self._conjunctivitis_viral,
            self._conjunctivitis_allergic, self._subconjunctival_hemorrhage, self._pterygium,
            self._pinguecula, self._hordeolum, self._chalazion, self._scleritis,
            self._scleral_icterus, self._corneal_ulcer, self._blepharitis,
            self._keratitis, self._dry_eye_external,
        ]
        img = generators[disease_idx]()

        # Augmentation pipeline
        if random.random() < 0.5:
            brightness = random.uniform(0.7, 1.3)
            img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            contrast = random.uniform(0.7, 1.3)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            img = np.clip(mean + (img.astype(np.float32) - mean) * contrast, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = np.mod(hsv[:, :, 0] + random.uniform(-10, 10), 180)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), random.uniform(0.3, 1.0))
        if random.random() < 0.2:
            angle = random.uniform(-5, 5)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return img


# ─────────────────────────────────────────────
#  IMPROVED FUNDUS DISEASE GENERATOR
# ─────────────────────────────────────────────

class _FundusDiseaseGenerator:
    def __init__(self, size: int = IMG_SIZE):
        self.size = size
        self.diseases = [
            "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD",
            "Retinal Detachment", "Hypertensive Retinopathy", "Macular Edema",
            "Retinitis Pigmentosa", "Optic Disc Edema", "Dry Eye Disease",
        ]

    def _create_fundus(self) -> np.ndarray:
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        bg = random.randint(5, 30)
        img[:, :] = (bg, bg, bg)
        cx, cy = self.size // 2, self.size // 2
        r = self.size // 2 - random.randint(8, 18)

        fundus_color = (random.randint(25, 55), random.randint(50, 90), random.randint(70, 120))
        cv2.circle(img, (cx, cy), r, fundus_color, -1)
        inner = int(r * random.uniform(0.82, 0.92))
        cv2.circle(img, (cx, cy), inner,
                   (min(fundus_color[0] + random.randint(15, 35), 130),
                    min(fundus_color[1] + random.randint(15, 35), 150),
                    min(fundus_color[2] + random.randint(15, 35), 180)), -1)

        disc_x = cx + int(r * random.uniform(0.10, 0.20))
        disc_y = cy - int(r * random.uniform(0.05, 0.12))
        cv2.circle(img, (disc_x, disc_y), int(r * random.uniform(0.12, 0.18)),
                   (random.randint(140, 180), random.randint(160, 200), random.randint(190, 230)), -1)

        mac_x = cx - int(r * random.uniform(0.15, 0.25))
        mac_y = cy + int(r * random.uniform(0.02, 0.08))
        cv2.circle(img, (mac_x, mac_y), int(r * random.uniform(0.05, 0.10)),
                   (random.randint(40, 70), random.randint(50, 80), random.randint(60, 90)), -1)

        for _ in range(random.randint(15, 30)):
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(int(r * 0.25), int(r * 0.65))
            x1 = disc_x + random.randint(-6, 6)
            y1 = disc_y + random.randint(-6, 6)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            if 0 <= x2 < self.size and 0 <= y2 < self.size:
                cv2.line(img, (x1, y1), (x2, y2),
                         (random.randint(25, 50), random.randint(15, 35), random.randint(55, 85)),
                         random.randint(1, 2))

        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.8))

        return self._add_noise(img)

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.randint(-6, 7, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def _add_dr_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 10
        num_hem = int(8 + 25 * severity)
        for _ in range(num_hem):
            theta = random.uniform(0, 2 * np.pi)
            rad = random.uniform(0, r * random.uniform(0.5, 0.85))
            x = int(cx + rad * np.cos(theta))
            y = int(cy + rad * np.sin(theta))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), random.randint(2, 6),
                           (random.randint(20, 40), random.randint(20, 40), random.randint(150, 220)), -1)
                cv2.circle(img, (x, y), random.randint(3, 7),
                           (random.randint(10, 30), random.randint(10, 30), random.randint(120, 180)), 1)
        num_ex = int(3 + 12 * severity)
        for _ in range(num_ex):
            theta = random.uniform(0, 2 * np.pi)
            rad = random.uniform(0, r * 0.55)
            x = int(cx + rad * np.cos(theta))
            y = int(cy + rad * np.sin(theta))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), random.randint(3, 9),
                           (random.randint(170, 200), random.randint(200, 220), random.randint(210, 235)), -1)

    def _add_glaucoma_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 10
        disc_x = cx + int(r * random.uniform(0.10, 0.20))
        disc_y = cy - int(r * random.uniform(0.05, 0.12))
        disc_r = int(r * random.uniform(0.14, 0.20))
        cup_r = int(disc_r * (0.45 + severity * 0.45))
        cv2.circle(img, (disc_x, disc_y), cup_r,
                   (random.randint(190, 220), random.randint(200, 230), random.randint(210, 245)), -1)
        cv2.circle(img, (disc_x, disc_y), disc_r,
                   (random.randint(130, 160), random.randint(150, 180), random.randint(180, 210)), 2)
        rim_thinness = int(severity * 3)
        for i in range(rim_thinness):
            angle = i * 2 * np.pi / rim_thinness
            nx = disc_x + int(disc_r * 0.7 * np.cos(angle))
            ny = disc_y + int(disc_r * 0.7 * np.sin(angle))
            cv2.circle(img, (nx, ny), max(1, int(disc_r * 0.3 * (1 - severity * 0.5))),
                       (random.randint(120, 150), random.randint(140, 170), random.randint(170, 200)), -1)

    def _add_cataract_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        overlay = img.copy()
        haze = int(80 + 100 * severity)
        cv2.circle(overlay, (w // 2, h // 2), min(w, h) // 2, (haze, haze, haze), -1)
        alpha = 0.15 + 0.35 * severity
        beta = random.uniform(0.3, 0.5) * severity
        cv2.addWeighted(overlay, alpha, img, 1 - alpha + beta, random.randint(5, 15), img)

    def _add_amd_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 10
        mac_x = cx - int(r * random.uniform(0.15, 0.25))
        mac_y = cy + int(r * random.uniform(0.02, 0.08))
        num_drusen = int(5 + 20 * severity)
        for _ in range(num_drusen):
            x = mac_x + random.randint(-35, 35)
            y = mac_y + random.randint(-30, 30)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), random.randint(2, 7),
                           (random.randint(150, 190), random.randint(190, 220), random.randint(210, 240)), -1)
                cv2.circle(img, (x, y), random.randint(1, 5),
                           (random.randint(170, 210), random.randint(210, 240), random.randint(230, 250)), -1)
        if severity > 0.6:
            ga_r = int(10 + 20 * (severity - 0.6))
            cv2.circle(img, (mac_x, mac_y), ga_r,
                       (random.randint(140, 170), random.randint(160, 190), random.randint(180, 210)), -1)

    def _add_retinal_detachment_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 10
        angle = random.uniform(0, 2 * np.pi)
        arc_len = int(60 + 200 * severity)
        det_color = (random.randint(25, 45), random.randint(35, 55), random.randint(50, 70))
        cv2.ellipse(img, (cx, cy), (r - 2, r - 2), np.degrees(angle), 0, arc_len, det_color, -1)
        cv2.ellipse(img, (cx, cy), (r - 5, r - 5), np.degrees(angle), 5, max(10, arc_len - 5),
                    (det_color[0] - 10, det_color[1] - 10, det_color[2] - 10), 2)
        waves = int(2 + severity * 4)
        for i in range(waves):
            a = angle + (i / waves) * np.radians(arc_len)
            nx = cx + int((r - 5 - i * 3) * np.cos(a))
            ny = cy + int((r - 5 - i * 3) * np.sin(a))
            cv2.circle(img, (nx, ny), random.randint(3, 8),
                       (random.randint(30, 50), random.randint(40, 60), random.randint(60, 80)), -1)

    def _add_hypertensive_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 10
        disc_x = cx + int(r * random.uniform(0.10, 0.20))
        disc_y = cy - int(r * random.uniform(0.05, 0.12))
        num_av = int(4 + 10 * severity)
        for _ in range(num_av):
            angle = random.uniform(0, 2 * np.pi)
            length = int(r * (0.3 + 0.4 * severity))
            x1 = disc_x + random.randint(-3, 3)
            y1 = disc_y + random.randint(-3, 3)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            if 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(img, (x1, y1), (x2, y2),
                         (random.randint(80, 110), random.randint(60, 90), random.randint(40, 70)), 1)
                ox = int(x1 + r * 0.12 * np.cos(angle))
                oy = int(y1 + r * 0.12 * np.sin(angle))
                cv2.line(img, (ox, oy), (x2, y2),
                         (random.randint(100, 130), random.randint(80, 110), random.randint(60, 90)), 1)
        # Silver-wiring
        for _ in range(int(3 * severity)):
            angle = random.uniform(0, 2 * np.pi)
            sx = disc_x + int(r * 0.15 * np.cos(angle))
            sy = disc_y + int(r * 0.15 * np.sin(angle))
            cv2.line(img, (sx, sy), (sx + int(20 * np.cos(angle)), sy + int(20 * np.sin(angle))),
                     (random.randint(160, 200), random.randint(150, 190), random.randint(140, 180)), 1)

    def _add_macular_edema_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 10
        mac_x = cx - int(r * random.uniform(0.15, 0.25))
        mac_y = cy + int(r * random.uniform(0.02, 0.08))
        swell_r = int(10 + 28 * severity)
        cv2.circle(img, (mac_x, mac_y), swell_r,
                   (random.randint(60, 80), random.randint(70, 90), random.randint(90, 110)), -1)
        cv2.circle(img, (mac_x, mac_y), swell_r + 4,
                   (random.randint(50, 70), random.randint(60, 80), random.randint(75, 95)), 1)
        for _ in range(int(5 * severity)):
            theta = random.uniform(0, 2 * np.pi)
            rad = random.uniform(0, swell_r * 0.8)
            cx2 = mac_x + int(rad * np.cos(theta))
            cy2 = mac_y + int(rad * np.sin(theta))
            cv2.circle(img, (cx2, cy2), random.randint(2, 5),
                       (random.randint(50, 70), random.randint(60, 80), random.randint(80, 100)), -1)

    def _add_rp_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 10
        num_spicules = int(10 + 35 * severity)
        for _ in range(num_spicules):
            theta = random.uniform(0, 2 * np.pi)
            rad = random.uniform(r * 0.25, r * 0.85)
            x = int(cx + rad * np.cos(theta))
            y = int(cy + rad * np.sin(theta))
            if 0 <= x < w and 0 <= y < h:
                pts = np.array([
                    [x, y],
                    [x + random.randint(-6, 6), y + random.randint(4, 12)],
                    [x + random.randint(-4, 4), y + random.randint(8, 18)],
                ], np.int32)
                cv2.fillPoly(img, [pts], (random.randint(20, 40), random.randint(20, 35), random.randint(40, 60)))
                cv2.polylines(img, [pts], True, (random.randint(40, 60), random.randint(40, 55), random.randint(60, 80)), 1)
        # Attenuated vessels
        for _ in range(int(5 * severity)):
            angle = random.uniform(0, 2 * np.pi)
            ax = cx + int(r * 0.1 * np.cos(angle))
            ay = cy + int(r * 0.1 * np.sin(angle))
            cv2.line(img, (ax, ay), (ax + int(30 * np.cos(angle)), ay + int(30 * np.sin(angle))),
                     (random.randint(15, 30), random.randint(10, 25), random.randint(35, 55)), 1)

    def _add_optic_disc_edema_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 10
        disc_x = cx + int(r * random.uniform(0.10, 0.20))
        disc_y = cy - int(r * random.uniform(0.05, 0.12))
        base_r = int(r * random.uniform(0.12, 0.18))
        swell = int(5 + 15 * severity)
        cv2.circle(img, (disc_x, disc_y), base_r + swell,
                   (random.randint(140, 170), random.randint(160, 190), random.randint(190, 220)), -1)
        cv2.circle(img, (disc_x, disc_y), base_r + swell + 2,
                   (random.randint(120, 150), random.randint(140, 170), random.randint(170, 200)), 1)
        for _ in range(int(3 * severity)):
            theta = random.uniform(0, 2 * np.pi)
            nx = disc_x + int((base_r + swell) * 0.7 * np.cos(theta))
            ny = disc_y + int((base_r + swell) * 0.7 * np.sin(theta))
            cv2.circle(img, (nx, ny), random.randint(2, 5),
                       (random.randint(150, 180), random.randint(170, 200), random.randint(200, 230)), -1)

    def _add_dry_eye_features(self, img: np.ndarray, severity: float):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 10
        num_stains = int(10 + 40 * severity)
        for _ in range(num_stains):
            theta = random.uniform(0, 2 * np.pi)
            rad = random.uniform(0, r * 0.9)
            x = int(cx + rad * np.cos(theta))
            y = int(cy + rad * np.sin(theta))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), random.randint(1, 3),
                           (random.randint(120, 150), random.randint(130, 160), random.randint(140, 170)), -1)
        # Surface irregularity
        overlay = np.zeros_like(img)
        cv2.circle(overlay, (cx, cy), r, (random.randint(15, 30),) * 3, -1)
        alpha = 0.05 * severity
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def generate(self, disease_idx: int, severity: float = 0.7) -> np.ndarray:
        img = self._create_fundus()
        funcs = [
            self._add_dr_features, self._add_glaucoma_features, self._add_cataract_features,
            self._add_amd_features, self._add_retinal_detachment_features,
            self._add_hypertensive_features, self._add_macular_edema_features,
            self._add_rp_features, self._add_optic_disc_edema_features, self._add_dry_eye_features,
        ]
        if 0 <= disease_idx < len(funcs):
            funcs[disease_idx](img, severity)
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.6))
        return img


# ─────────────────────────────────────────────
#  DATASETS
# ─────────────────────────────────────────────

class _TypeClassifierDataset(Dataset):
    def __init__(self, num_per_class: int = 500, img_size: int = IMG_SIZE):
        self.generator = _TypeImageGenerator(img_size)
        self.num_per_class = num_per_class
        self.num_classes = TYPE_NUM_CLASSES
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data = self._generate()

    def _generate(self):
        data = []
        for class_idx in range(self.num_classes):
            for _ in range(self.num_per_class):
                img_np = self.generator.generate(class_idx)
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                tensor = self.transform(pil_img)
                data.append((tensor, class_idx))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class _ExternalEyeDataset(Dataset):
    def __init__(self, num_per_class: int = 400, img_size: int = IMG_SIZE):
        self.generator = _ExternalEyeGenerator(img_size)
        self.num_per_class = num_per_class
        self.num_classes = EXTERNAL_NUM_CLASSES
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data = self._generate()

    def _generate(self):
        data = []
        for class_idx in range(self.num_classes):
            for _ in range(self.num_per_class):
                img_np = self.generator.generate(class_idx)
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                tensor = self.transform(pil_img)
                data.append((tensor, class_idx))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class _FundusDataset(Dataset):
    def __init__(self, num_samples: int = 3000, img_size: int = IMG_SIZE):
        self.generator = _FundusDiseaseGenerator(img_size)
        self.num_samples = max(num_samples, 100)
        self.img_size = img_size
        self.num_classes = NUM_CLASSES
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data = self._generate()

    def _generate(self):
        data = []
        samples_per_class = max(1, self.num_samples // self.num_classes)
        for class_idx in range(self.num_classes):
            for _ in range(samples_per_class):
                severity = random.uniform(0.3, 1.0)
                img_np = self.generator.generate(class_idx, severity)
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                tensor = self.transform(pil_img)
                data.append((tensor, class_idx))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────
#  MODEL UTILITIES
# ─────────────────────────────────────────────

def _replace_classifier_head(model, key: str, num_classes: int):
    if key == "efficientnet":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif key == "convnext":
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif key == "resnet":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif key == "mobilenet":
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def _freeze_backbone(model, key: str):
    if key == "efficientnet":
        for param in model.features.parameters():
            param.requires_grad = False
        classifier_params = model.classifier[1].parameters()
    elif key == "convnext":
        for param in model.features.parameters():
            param.requires_grad = False
        classifier_params = model.classifier[2].parameters()
    elif key == "resnet":
        for param in model.parameters():
            param.requires_grad = False
        classifier_params = model.fc.parameters()
    elif key == "mobilenet":
        for param in model.features.parameters():
            param.requires_grad = False
        classifier_params = model.classifier[3].parameters()
    else:
        classifier_params = model.parameters()
    for p in classifier_params:
        p.requires_grad = True
    return model


def _train_model(model, loader: DataLoader, epochs: int, lr: float, device, cb, label: str = ""):
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_batches = len(loader) * epochs
    batch_count = 0
    running_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            epoch_loss += loss.item()
            batch_count += 1
            pct = int((batch_count / total_batches) * 100)
            if cb:
                acc = correct / max(total, 1)
                cb(pct, f"[{label or 'model'}] E{epoch+1}/{epochs} loss={loss.item():.3f} acc={acc:.2f}")
        running_loss = epoch_loss / len(loader)
    acc = correct / max(total, 1)
    return {"final_loss": running_loss, "accuracy": acc, "epochs": epochs}


# ─────────────────────────────────────────────
#  TRAINING ORCHESTRATION
# ─────────────────────────────────────────────

def _train_type_classifier(device, progress_callback) -> dict:
    def cb(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    cb(1, "Generating synthetic image type dataset...")
    dataset = _TypeClassifierDataset(num_per_class=500)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    cb(5, f"Type dataset ready: {len(dataset)} samples, {TYPE_NUM_CLASSES} classes")

    cb(6, "Loading MobileNetV3-Small for type classifier...")
    model = models.mobilenet_v3_small(weights="DEFAULT").to(device)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, TYPE_NUM_CLASSES)
    _freeze_backbone(model, "mobilenet")

    def type_cb(pct, msg):
        cb(10 + int(pct * 0.85), msg)

    result = _train_model(model, loader, epochs=8, lr=0.002, device=device, cb=type_cb, label="type_cls")
    save_path = TRAINED_DIR / "type_classifier.pt"
    torch.save(model.state_dict(), save_path)
    cb(95, f"Type classifier saved (acc={result['accuracy']:.2f})")
    return result


def _train_external_eye_classifier(device, progress_callback) -> dict:
    def cb(pct, msg):
        if progress_callback:
            progress_callback(pct + 100, msg)

    cb(101, "Generating synthetic external eye dataset...")
    dataset = _ExternalEyeDataset(num_per_class=400)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    cb(105, f"External eye dataset ready: {len(dataset)} samples, {EXTERNAL_NUM_CLASSES} classes")

    cb(106, "Loading ResNet18 for external eye classifier...")
    model = models.resnet18(weights="DEFAULT").to(device)
    model.fc = nn.Linear(model.fc.in_features, EXTERNAL_NUM_CLASSES)
    _freeze_backbone(model, "resnet")

    def ext_cb(pct, msg):
        cb(110 + int(pct * 0.85), msg)

    result = _train_model(model, loader, epochs=8, lr=0.002, device=device, cb=ext_cb, label="ext_eye")
    save_path = TRAINED_DIR / "external_eye_model.pt"
    torch.save(model.state_dict(), save_path)
    cb(195, f"External eye model saved (acc={result['accuracy']:.2f})")
    return result


def _train_fundus_models(device, epochs, batch_size, lr, progress_callback) -> dict:
    results = {}
    errors = []

    def cb(pct, msg):
        if progress_callback:
            progress_callback(pct + 200, msg)

    cb(1, "Generating improved synthetic fundus dataset...")
    dataset = _FundusDataset(num_samples=max(600, batch_size * NUM_CLASSES))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cb(5, f"Fundus dataset ready: {len(dataset)} samples, {NUM_CLASSES} classes")

    model_configs = [
        ("efficientnet", "EfficientNetV2-S"),
        ("convnext", "ConvNeXt Tiny"),
        ("resnet", "ResNet18"),
        ("mobilenet", "MobileNetV3-Small"),
    ]
    total_models = len(model_configs)
    for m_idx, (key, name) in enumerate(model_configs):
        try:
            cb(5 + int((m_idx / total_models) * 90),
               f"Loading base {name} for fine-tuning...")
            base_loaders = {
                "efficientnet": lambda: models.efficientnet_v2_s(weights="DEFAULT"),
                "convnext": lambda: models.convnext_tiny(weights="DEFAULT"),
                "resnet": lambda: models.resnet18(weights="DEFAULT"),
                "mobilenet": lambda: models.mobilenet_v3_small(weights="DEFAULT"),
            }
            model = base_loaders[key]().to(device)
            model = _replace_classifier_head(model, key, NUM_CLASSES)
            model = _freeze_backbone(model, key)

            start_pct = 5 + int((m_idx / total_models) * 90)
            end_pct = 5 + int(((m_idx + 1) / total_models) * 90)

            def _make_model_cb(s, e):
                def inner(pct_in_epoch, msg):
                    scaled = s + int((pct_in_epoch / 100) * (e - s))
                    cb(min(scaled, 99), msg)
                return inner

            train_result = _train_model(model, loader, epochs, lr, device,
                                         _make_model_cb(start_pct, end_pct), label=name)
            results[key] = train_result
            save_path = TRAINED_DIR / f"{key}_finetuned.pt"
            torch.save(model.state_dict(), save_path)
            cb(end_pct, f"{name} saved (loss={train_result['final_loss']:.3f}, acc={train_result['accuracy']:.2f})")
        except Exception as e:
            errors.append(f"{name}: {e}")
            cb(0, f"ERROR training {name}: {e}")

    return {"results": results, "errors": errors}


# ─────────────────────────────────────────────
#  MAIN TRAINING ENTRY POINTS
# ─────────────────────────────────────────────

def run_training(epochs: int = EPOCHS_DEFAULT, batch_size: int = BATCH_SIZE_DEFAULT,
                 lr: float = LR_DEFAULT, progress_callback=None) -> dict:
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    device = _get_device()
    all_errors = []
    all_results = {}

    def cb(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    try:
        # Phase 1: Type classifier
        cb(0, "=== Phase 1: Training Image Type Classifier ===")
        type_result = _train_type_classifier(device, progress_callback)
        all_results["type_classifier"] = type_result
        if type_result.get("error"):
            all_errors.append(f"type_classifier: {type_result['error']}")

        # Phase 2: External eye classifier
        cb(100, "=== Phase 2: Training External Eye Classifier ===")
        ext_result = _train_external_eye_classifier(device, progress_callback)
        all_results["external_eye"] = ext_result
        if ext_result.get("error"):
            all_errors.append(f"external_eye: {ext_result['error']}")

        # Phase 3: Fundus disease models
        cb(200, "=== Phase 3: Fine-tuning Fundus Disease Models ===")
        fundus_result = _train_fundus_models(device, epochs, batch_size, lr,
                                              lambda p, m: cb(p + 200, m) if progress_callback else None)
        all_results["fundus"] = fundus_result

        # Filter out model-level errors
        for e in fundus_result.get("errors", []):
            all_errors.append(f"fundus: {e}")

        cb(300, "All training complete!")
        return {
            "success": len(all_errors) == 0,
            "results": all_results,
            "errors": all_errors,
        }

    except Exception as e:
        import traceback
        return {"success": False, "results": all_results, "errors": [str(e), traceback.format_exc()]}
