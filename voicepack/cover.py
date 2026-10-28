"""封面图生成：把用户提供的角色 CG 图做成 1920x1080 的视频封面。

用户自己提供 CG 图（--image），本模块只负责：
* ``cover_mode="crop"``：等比放大到铺满 16:9，居中裁剪（适合本来就是 16:9 的 CG）；
* ``cover_mode="blur"``：整图等比缩进画面居中，四周用放大 + 高斯模糊的同一张图填充
  （竖版 / 特殊比例的 CG 用这个，不会把人裁掉）。
* 两种模式都压一层底部渐变、叠标题（角色名）和副标题（句数 / 时长），
  标题为空则不出字；超长标题自动缩字号，必要时换两行。

不再自动“创作”封面：``render_placeholder_cover`` 只是调试用的渐变占位图，
正常代码路径（CLI / 流水线）**不会**调用它。

只依赖 Pillow；numpy 缺失时自动退回纯 PIL 实现，不会报错。
可以单独运行：

    python -m voicepack.cover --image CG.png --out cover.png \\
        --title "四季 ナツ" --subtitle "语音合集 · 4092 句 · 4 小时 30 分" --mode crop
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

try:  # numpy 只是加速，环境里没有也要能跑
    import numpy as np
except Exception:  # pragma: no cover - 没装 numpy 时的纯 PIL 退路
    np = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
FONT_DIR = Path("C:/Windows/Fonts")
# 标题优先用粗体微软雅黑，副标题用常规体
_FONT_BOLD_CANDIDATES = ("msyhbd.ttc", "msyh.ttc", "simhei.ttf", "simsun.ttc")
_FONT_REGULAR_CANDIDATES = ("msyh.ttc", "msyhbd.ttc", "simsun.ttc", "simhei.ttf")
_FONT_LATIN_CANDIDATES = ("segoeuib.ttf", "arialbd.ttf", "msyhbd.ttc", "msyh.ttc")

# 调试占位图用的主题色（粉紫黄昏感），正常封面不会用到
_GRAD_STOPS = (
    (0.00, (178, 194, 238)),
    (0.30, (218, 192, 232)),
    (0.58, (242, 202, 214)),
    (0.82, (247, 220, 200)),
    (1.00, (221, 174, 188)),
)

KICKER_DEFAULT = "VOICE COLLECTION"
COVER_MODES = ("crop", "blur")


# --------------------------------------------------------------------------- #
# 字体
# --------------------------------------------------------------------------- #
def _load_font(candidates: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """按候选列表找 Windows 中文字体；全部失败时退回 PIL 默认字体（不报错）。"""
    size = max(1, int(size))
    for name in candidates:
        path = FONT_DIR / name
        if not path.is_file():
            continue
        for index in (0, 1):  # msyh.ttc 有多个 face，index 0 是常规/粗体主 face
            try:
                return ImageFont.truetype(str(path), size, index=index)
            except Exception:
                continue
    try:
        return ImageFont.truetype("msyh.ttc", size)
    except Exception:
        pass
    return ImageFont.load_default(size)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return float(box[2] - box[0])
    except Exception:
        return float(len(text) * getattr(font, "size", 12))


def _text_height(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return float(box[3] - box[1])
    except Exception:
        return float(getattr(font, "size", 12))


def _fit_font(draw, text: str, candidates: tuple[str, ...], size: int, max_width: float, min_size: int) -> ImageFont:
    """从 size 开始往下缩，直到文字宽度不超过 max_width。"""
    size = max(int(min_size), int(size))
    while size > min_size:
        font = _load_font(candidates, size)
        if _text_width(draw, text, font) <= max_width:
            return font
        size = int(size * 0.94)
    return _load_font(candidates, min_size)


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: float, limit: int = 2) -> list[str]:
    """把一行字折成最多 ``limit`` 行；放得下就原样一行。

    折行点：先找空格（英文名用），没有空格就按字符宽度寻找“最平衡”的切点。
    """
    text = text.strip()
    if not text:
        return []
    if _text_width(draw, text, font) <= max_width:
        return [text]
    if limit <= 1:
        return [text]

    best: tuple[float, str, str] | None = None

    def consider(left: str, right: str) -> None:
        nonlocal best
        left, right = left.strip(), right.strip()
        if not left or not right:
            return
        wl, wr = _text_width(draw, left, font), _text_width(draw, right, font)
        if wl <= max_width and wr <= max_width:
            score = max(wl, wr)
            if best is None or score < best[0]:
                best = (score, left, right)

    # 1) 空格处折行（英文名 / 「角色名 - 作品名」这种）
    spaces = [i for i, ch in enumerate(text) if ch in (" ", "-", "·", "/")]
    for i in spaces:
        consider(text[:i], text[i + 1 :])
    # 2) 中日文没有空格：按字符找最平衡的切分
    if best is None:
        for i in range(1, len(text)):
            consider(text[:i], text[i:])
    if best is None:
        return [text]
    return [best[1], best[2]]


# --------------------------------------------------------------------------- #
# 调试占位图（正常流程不会调用）
# --------------------------------------------------------------------------- #
def _lerp_color(stops, t: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1 or i == len(stops) - 2:
            k = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            k = min(1.0, max(0.0, k))
            return tuple(int(round(c0[j] + (c1[j] - c0[j]) * k)) for j in range(3))
    return stops[-1][1]


def _gradient_background(width: int, height: int) -> Image.Image:
    """斜向多段渐变 + 两个大范围柔光（仅调试占位图使用）。"""
    sw, sh = max(64, width // 12), max(36, height // 12)

    if np is not None:
        ys = np.linspace(0.0, 1.0, sh, dtype=np.float32)[:, None]
        xs = np.linspace(0.0, 1.0, sw, dtype=np.float32)[None, :]
        t = np.clip(0.70 * ys + 0.30 * xs, 0.0, 1.0).astype(np.float32)

        arr = np.zeros((sh, sw, 3), dtype=np.float32)
        for c in range(3):
            pos = [s[0] for s in _GRAD_STOPS]
            vals = [s[1][c] for s in _GRAD_STOPS]
            arr[..., c] = np.interp(t, pos, vals).astype(np.float32)

        gx, gy, gr = 0.74, 0.16, 0.60
        d1 = np.sqrt(((xs - gx) * (width / height)) ** 2 + (ys - gy) ** 2)
        glow1 = np.exp(-(d1 / gr) ** 2).astype(np.float32)
        arr += glow1[..., None] * np.array([255 - 188, 250 - 201, 235 - 238], dtype=np.float32) * 0.62

        gx2, gy2, gr2 = 1.02, 0.86, 0.45
        d2 = np.sqrt(((xs - gx2) * (width / height)) ** 2 + (ys - gy2) ** 2)
        glow2 = np.exp(-(d2 / gr2) ** 2).astype(np.float32)
        arr += glow2[..., None] * np.array([8.0, -12.0, 4.0], dtype=np.float32)

        vx = np.sqrt(((xs - 0.5) * 1.15) ** 2 + ((ys - 0.5) * 1.0) ** 2) / 0.707
        vig = (1.0 - 0.20 * np.clip(vx, 0.0, 1.0) ** 2).astype(np.float32)
        arr *= vig[..., None]

        small = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        return small.resize((width, height), Image.Resampling.BICUBIC)

    small = Image.new("RGB", (sw, sh))
    px = small.load()
    for y in range(sh):
        for x in range(sw):
            t = 0.70 * (y / max(sh - 1, 1)) + 0.30 * (x / max(sw - 1, 1))
            px[x, y] = _lerp_color(_GRAD_STOPS, t)
    big = small.resize((width, height), Image.Resampling.BICUBIC)
    glow = _radial_mask(width, height, 0.74, 0.16, 0.62)
    white = Image.new("RGB", (width, height), (72, 49, 0))
    big = ImageChops.add(big, Image.composite(white, Image.new("RGB", (width, height), (0, 0, 0)), glow))
    return big


def _radial_mask(width: int, height: int, cx: float, cy: float, radius: float) -> Image.Image:
    sw, sh = max(64, width // 12), max(36, height // 12)
    small = Image.new("L", (sw, sh))
    px = small.load()
    for y in range(sh):
        yn = y / max(sh - 1, 1)
        for x in range(sw):
            xn = x / max(sw - 1, 1)
            d = math.sqrt(((xn - cx) * width / height) ** 2 + (yn - cy) ** 2)
            px[x, y] = int(255 * math.exp(-((d / radius) ** 2)))
    return small.resize((width, height), Image.Resampling.BICUBIC)


def _disk_sprite(radius: int = 128) -> Image.Image:
    """一颗“焦外光斑”的 RGBA 贴图（调试占位图用）。"""
    size = radius * 2
    if np is not None:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.sqrt((xx - radius + 0.5) ** 2 + (yy - radius + 0.5) ** 2) / radius
        inside = np.clip((1.0 - d) / 0.10, 0.0, 1.0)
        inside = inside * inside * (3 - 2 * inside)
        ring = np.exp(-(((d - 0.86) / 0.14) ** 2))
        alpha = np.clip(inside * (0.82 + 0.18 * ring), 0.0, 1.0)
        a = (alpha * 255).astype(np.uint8)
        img = Image.fromarray(a, "L")
        return Image.merge("RGBA", (img, img, img, img))

    img = Image.new("L", (size, size), 0)
    dpx = img.load()
    for y in range(size):
        for x in range(size):
            d = math.hypot(x - radius + 0.5, y - radius + 0.5) / radius
            inside = min(1.0, max(0.0, (1.0 - d) / 0.10))
            inside = inside * inside * (3 - 2 * inside)
            ring = math.exp(-(((d - 0.86) / 0.14) ** 2))
            dpx[x, y] = int(255 * min(1.0, inside * (0.82 + 0.18 * ring)))
    return Image.merge("RGBA", (img, img, img, img))


def _add_bokeh(base: Image.Image, rng: random.Random) -> Image.Image:
    w, h = base.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sprite = _disk_sprite(128)

    tints = (
        (255, 251, 243),
        (255, 244, 220),
        (255, 225, 233),
        (232, 240, 255),
        (252, 232, 250),
    )
    items = []
    for _ in range(28):
        bucket = rng.random()
        if bucket < 0.58:
            r = rng.uniform(10, 42)
            strength = rng.uniform(0.14, 0.32)
        elif bucket < 0.90:
            r = rng.uniform(42, 88)
            strength = rng.uniform(0.10, 0.22)
        else:
            r = rng.uniform(88, 138)
            strength = rng.uniform(0.06, 0.13)
        x = rng.uniform(-0.04, 1.04) * w
        y = rng.uniform(-0.06, 0.88) * h
        items.append((r, x, y, strength, rng.choice(tints)))

    items.sort(key=lambda it: -it[0])
    for r, x, y, strength, tint in items:
        d = int(r * 2)
        if d < 4:
            continue
        spr = sprite.resize((d, d), Image.Resampling.BILINEAR)
        a = spr.getchannel("A").point(lambda v: int(v * strength))
        colored = Image.new("RGBA", (d, d), tint + (255,))
        colored.putalpha(a)
        layer.alpha_composite(colored, (int(x - r), int(y - r)))

    layer = layer.convert("RGBa").filter(ImageFilter.GaussianBlur(3.0)).convert("RGBA")
    flattened = Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 255)), layer).convert("RGB")
    rgb = base.convert("RGB")
    if np is not None:
        a = np.asarray(rgb, dtype=np.float32) / 255.0
        b = np.asarray(flattened, dtype=np.float32) / 255.0
        mixed = 1.0 - (1.0 - a) * (1.0 - b)
        return Image.fromarray(np.clip(mixed * 255.0, 0, 255).astype(np.uint8), "RGB")
    return ImageChops.screen(rgb, flattened)


def _add_sparkles(base: Image.Image, rng: random.Random) -> Image.Image:
    w, h = base.size
    mask = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(mask)
    for _ in range(8):
        cx = rng.uniform(0.06, 0.94) * w
        cy = rng.uniform(0.06, 0.55) * h
        s = rng.uniform(4.5, 9.0)
        alpha = int(rng.uniform(130, 210))
        long = s * rng.uniform(1.5, 2.6)
        dr.polygon([(cx, cy - long), (cx + s * 0.20, cy), (cx, cy + long), (cx - s * 0.20, cy)], fill=alpha)
        dr.polygon([(cx - long, cy), (cx, cy + s * 0.20), (cx + long, cy), (cx, cy - s * 0.20)], fill=alpha)
        dr.ellipse([cx - s * 0.30, cy - s * 0.30, cx + s * 0.30, cy + s * 0.30], fill=245)
    mask = mask.filter(ImageFilter.GaussianBlur(0.9))
    white = Image.new("RGB", (w, h), (255, 255, 255))
    return Image.composite(white, base.convert("RGB"), mask)


def _add_grain(img: Image.Image, amount: float = 2.4, seed: int = 7) -> Image.Image:
    """细颗粒噪点：消掉渐变的“塑料感”，也更抗视频压缩。"""
    w, h = img.size
    if np is not None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, amount, (h, w, 1)).astype(np.float32)
        arr = np.asarray(img, dtype=np.float32) + noise
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    noise = Image.effect_noise((w, h), amount * 1.7).convert("L")
    return ImageChops.add(img, Image.merge("RGB", (noise, noise, noise)), scale=1, offset=-128)


# --------------------------------------------------------------------------- #
# 底部压暗渐变 + 文字
# --------------------------------------------------------------------------- #
def _bottom_scrim(width: int, height: int, strength: float = 1.0) -> Image.Image:
    """从画面中下部到底部的黑色渐变遮罩，保证标题可读（RGBA）。"""
    scrim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    top = int(height * 0.56)
    max_alpha = int(205 * max(0.2, min(1.5, strength)))
    col = scrim.load()
    for y in range(top, height):
        t = (y - top) / max(1, height - top)
        a = int(max_alpha * (t ** 1.85) * (3 - 2 * t))
        a = min(max_alpha, a)
        for x in range(width):
            col[x, y] = (8, 10, 22, a)
    return scrim


def _bbox(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[float, float, float, float]:
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except Exception:
        w = float(len(text) * getattr(font, "size", 12))
        return 0.0, 0.0, w, float(getattr(font, "size", 12))


def _draw_at_top(draw, text: str, font, fill, top: float, width: int) -> tuple[float, float, float]:
    """水平居中、以“字形顶部”为基准画一行字，返回 (左边界, 宽, 高)。"""
    bx0, by0, bx1, by1 = _bbox(draw, text, font)
    tw = bx1 - bx0
    th = by1 - by0
    left = (width - tw) / 2
    draw.text((left - bx0, top - by0), text, font=font, fill=fill)
    return left, tw, th


def _draw_tracked(draw, center_x: float, top_y: float, text: str, font, fill, tracking: float) -> float:
    """手写 letterspacing 的水平居中文字，返回总宽度。"""
    widths = []
    for ch in text:
        try:
            widths.append(float(draw.textlength(ch, font=font)))
        except Exception:
            widths.append(float(getattr(font, "size", 12)))
    total = sum(widths) + tracking * max(0, len(text) - 1)
    x = center_x - total / 2
    for ch, cw in zip(text, widths):
        box = _bbox(draw, ch, font)
        draw.text((x - box[0], top_y - box[1]), ch, font=font, fill=fill)
        x += cw + tracking
    return total


def _fit_title(draw, title: str, width: int, height: int):
    """给标题找字号 / 换行方案：先缩字号，一行放不下就折两行，仍放不下就缩到 14px。

    返回 ``(lines, font, heights, line_gap)``。
    """
    margin = int(width * 0.09)
    max_w = width - margin * 2
    max_size = max(18, int(height * 0.108))
    min_size = max(14, int(height * 0.040))
    line_gap_ratio = 0.016

    size = max_size
    while size >= min_size:
        font = _load_font(_FONT_BOLD_CANDIDATES, size)
        lines = _wrap_lines(draw, title, font, max_w, limit=2)
        if lines and all(_text_width(draw, line, font) <= max_w for line in lines):
            heights = [_text_height(draw, line, font) for line in lines]
            return lines, font, heights, int(height * line_gap_ratio)
        size = int(size * 0.94)

    # 极端长标题：继续缩到能放下为止
    for size in range(min_size, 13, -2):
        font = _load_font(_FONT_BOLD_CANDIDATES, size)
        lines = _wrap_lines(draw, title, font, max_w, limit=2)
        if lines and all(_text_width(draw, line, font) <= max_w for line in lines):
            heights = [_text_height(draw, line, font) for line in lines]
            return lines, font, heights, int(height * line_gap_ratio)

    # 真的放不下：截断并加省略号，保证不溢出
    font = _load_font(_FONT_BOLD_CANDIDATES, 14)
    lines = []
    for raw in _wrap_lines(draw, title, font, max_w, limit=2)[:2]:
        cut = raw
        while cut and _text_width(draw, cut + "…", font) > max_w:
            cut = cut[:-1]
        lines.append((cut + "…") if cut != raw else raw)
    heights = [_text_height(draw, line, font) for line in lines]
    return lines, font, heights, int(height * line_gap_ratio)


def _composite_text_block(canvas: Image.Image, title: str, subtitle: str, width: int, height: int) -> Image.Image:
    """在压暗后的画面上排版：小标签 / 大标题（可两行）/ 金色短线 / 副标题。

    排版自下而上量算真实字形高度；标题和副标题都为空时整块跳过（不出任何字）。
    """
    title = (title or "").strip()
    subtitle = (subtitle or "").strip()
    if not title and not subtitle:
        return canvas.convert("RGB")

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    meas = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    margin = int(width * 0.09)
    max_w = width - margin * 2

    title_lines, title_font, title_heights, line_gap = ([], None, [], 0)
    title_h = 0.0
    if title:
        title_lines, title_font, title_heights, line_gap = _fit_title(meas, title, width, height)
        title_h = sum(title_heights) + line_gap * max(0, len(title_lines) - 1)

    subtitle_font = None
    sub_h = 0.0
    if subtitle:
        subtitle_font = _fit_font(
            meas, subtitle, _FONT_REGULAR_CANDIDATES, int(height * 0.041), max_w * 0.94, int(height * 0.021)
        )
        sub_h = _text_height(meas, subtitle, subtitle_font)

    kicker_font = _load_font(_FONT_LATIN_CANDIDATES, max(14, int(height * 0.024)))
    kicker_tracking = kicker_font.size * 0.44
    kick_h = _text_height(meas, KICKER_DEFAULT, kicker_font) if title else 0.0

    gap_sub_title = int(height * 0.030)
    gap_title_line = int(height * 0.026)
    gap_line_kicker = int(height * 0.018)
    line_h = max(2, height // 300)
    line_w = int(width * 0.050)

    bottom = height - int(height * 0.070)
    # ---- 自下而上：副标题 -> 短线 -> 标题块 -> 小标签 ----
    sub_top = bottom - sub_h
    if title:
        accent_y = sub_top - gap_sub_title - line_h
        title_bottom = accent_y - gap_title_line
        title_top = title_bottom - title_h
        kicker_top = title_top - gap_line_kicker - kick_h
    else:
        accent_y = sub_top - gap_sub_title - line_h
        title_top = title_bottom = kicker_top = accent_y

    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sdr = ImageDraw.Draw(shadow)

    # 标题（居中，逐行，带柔和投影）
    y = title_top
    for line, lh in zip(title_lines, title_heights):
        _draw_at_top(sdr, line, title_font, (0, 0, 0, 170), y + max(3, height // 260), width)
        _draw_at_top(dr, line, title_font, (255, 253, 250, 255), y, width)
        y += lh + line_gap

    # 副标题（略降透明度）
    if subtitle:
        _draw_at_top(sdr, subtitle, subtitle_font, (0, 0, 0, 120), sub_top + 2, width)
        _draw_at_top(dr, subtitle, subtitle_font, (255, 251, 247, 226), sub_top, width)

    # 装饰短线（暖金）
    dr.rounded_rectangle(
        [(width - line_w) / 2, accent_y, (width + line_w) / 2, accent_y + line_h],
        radius=line_h / 2,
        fill=(255, 226, 186, 238),
    )

    # 顶部小标签（字距拉开）
    if title:
        _draw_tracked(dr, width / 2, kicker_top, KICKER_DEFAULT, kicker_font, (255, 255, 255, 195), kicker_tracking)

    shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, height // 220)))
    canvas = canvas.convert("RGBA")
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(layer)
    return canvas.convert("RGB")


# --------------------------------------------------------------------------- #
# 图片处理：crop / blur 两种铺满方式
# --------------------------------------------------------------------------- #
def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            return im.convert("RGBA")
        return im.convert("RGB")


def _crop_fill(src: Image.Image, width: int, height: int, focus: float = 0.5) -> Image.Image:
    """等比放大到铺满目标尺寸后居中裁剪（focus=0 顶部对齐，1 底部对齐）。"""
    sw, sh = src.size
    if sw <= 0 or sh <= 0:
        raise ValueError("封面图尺寸无效")
    scale = max(width / sw, height / sh)
    nw, nh = max(width, int(round(sw * scale))), max(height, int(round(sh * scale)))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    ox = (nw - width) // 2
    oy = int((nh - height) * min(1.0, max(0.0, focus)))
    return resized.crop((ox, oy, ox + width, oy + height))


def _blur_fill(src: Image.Image, width: int, height: int, blur_strength: float) -> Image.Image:
    """整图等比缩进画面居中，背景用同一张图放大 + 高斯模糊（竖版 CG 不裁人）。"""
    # 背景
    bg = _crop_fill(src, width, height).convert("RGB")
    radius = max(0.0, float(blur_strength)) * (width / 1920.0)
    if radius >= 0.3:
        bg = bg.filter(ImageFilter.GaussianBlur(radius))
    bg = ImageEnhance.Brightness(bg).enhance(0.55)
    bg = ImageEnhance.Color(bg).enhance(0.92)

    # 前景：等比缩放到画面内（留 2% 边，避免贴边太死）；
    # 源图比例本来就接近 16:9 时直接铺满，免得四周出现一圈莫名其妙的模糊边框
    sw, sh = src.size
    cover_scale = max(width / sw, height / sh)
    contain_scale = min(width / sw, height / sh)
    if cover_scale / contain_scale < 1.05:
        fg = _crop_fill(src, width, height).convert("RGBA")
        pos = (0, 0)
    else:
        scale = contain_scale * 0.96
        fw, fh = max(1, int(sw * scale)), max(1, int(sh * scale))
        fg = src.resize((fw, fh), Image.Resampling.LANCZOS)
        if fg.mode != "RGBA":
            fg = fg.convert("RGBA")
        pos = ((width - fw) // 2, (height - fh) // 2)

    # 前景投影，让竖图和大背景有层次
    shadow_alpha = fg.getchannel("A").filter(ImageFilter.GaussianBlur(max(6, int(width * 0.02))))
    shadow_alpha = shadow_alpha.point(lambda v: int(v * 0.55))
    shadow = Image.new("RGBA", fg.size, (0, 0, 0, 255))
    shadow.putalpha(shadow_alpha)
    canvas = bg.convert("RGBA")
    canvas.alpha_composite(shadow, pos)
    canvas.alpha_composite(fg, pos)
    return canvas.convert("RGB")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def render_cover(
    src_image: str | Path | None = None,
    out_png: str | Path | None = None,
    title: str = "",
    subtitle: str = "",
    width: int = 1920,
    height: int = 1080,
    cover_mode: str = "crop",
    blur_strength: float = 24,
) -> Path:
    """用用户提供的 CG 图生成视频封面 PNG（RGB），返回输出路径。

    Parameters
    ----------
    src_image:   角色 CG 图路径（**必需**，不再自动生成封面）
    out_png:     输出 PNG 路径，默认当前目录 cover.png
    title:       主标题（通常是角色名），为空则不出标题
    subtitle:    副标题（如「语音合集 · 4092 句 · 4 小时 30 分」），为空则不出副标题
    cover_mode:  "crop" 居中裁剪填满 / "blur" 整图缩进 + 模糊背景填充
    blur_strength: blur 模式下背景高斯模糊半径（1920 宽时的像素值）
    """
    width, height = int(width), int(height)
    if width < 320 or height < 180:
        raise ValueError("封面尺寸太小，至少 320x180")
    width += width % 2
    height += height % 2
    if src_image is None:
        raise ValueError("封面必须提供 CG 图（--image），不再自动生成封面；没有图时只输出 MP3")
    src_path = Path(src_image)
    if not src_path.is_file():
        raise FileNotFoundError(f"找不到封面图: {src_path}")
    cover_mode = (cover_mode or "crop").strip().lower()
    if cover_mode not in COVER_MODES:
        raise ValueError(f"未知 cover_mode: {cover_mode!r}（可选 {'/'.join(COVER_MODES)}）")

    out = Path(out_png) if out_png else Path("cover.png")
    out = out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as raw:
        src = ImageOps.exif_transpose(raw)
        src = src.convert("RGBA") if (src.mode in ("RGBA", "LA") or "transparency" in src.info) else src.convert("RGB")
        if cover_mode == "blur":
            canvas = _blur_fill(src, width, height, blur_strength)
        elif src.mode == "RGBA":
            # 竖版裁切时取景偏上一点，人物脸不容易被裁掉；透明 PNG 垫渐变底，避免黑底
            backdrop = _gradient_background(width, height).convert("RGBA")
            backdrop.alpha_composite(_crop_fill(src, width, height, focus=0.42))
            canvas = backdrop.convert("RGB")
        else:
            canvas = _crop_fill(src, width, height, focus=0.42).convert("RGB")

    # 图片本身已有颗粒感/色彩，这里只做极轻的影调统一
    canvas = _add_grain(canvas, amount=1.0, seed=11)
    canvas = ImageEnhance.Color(canvas).enhance(1.03)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.02)
    # 只有要出文字时才压底部渐变；没有标题的纯图封面保持干净
    if (title or "").strip() or (subtitle or "").strip():
        canvas = canvas.convert("RGBA")
        canvas.alpha_composite(_bottom_scrim(width, height))
        canvas = canvas.convert("RGB")
    canvas = _composite_text_block(canvas, title, subtitle, width, height)

    pnginfo = None
    try:
        from PIL import PngImagePlugin

        pnginfo = PngImagePlugin.PngInfo()
        if title:
            pnginfo.add_text("Title", title)
        if subtitle:
            pnginfo.add_text("Description", subtitle)
    except Exception:
        pnginfo = None
    canvas.save(out, format="PNG", optimize=True, pnginfo=pnginfo)
    return out


def render_placeholder_cover(
    out_png: str | Path | None = None,
    title: str = "",
    subtitle: str = "",
    width: int = 1920,
    height: int = 1080,
    seed: int = 20260428,
) -> Path:
    """**仅供调试/测试**的程序化渐变封面。

    正常流程（CLI、一键脚本、流水线）不会调用它——封面一律要求用户提供 CG 图。
    """
    width, height = int(width), int(height)
    width += width % 2
    height += height % 2
    out = Path(out_png) if out_png else Path("cover_placeholder.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    canvas = _gradient_background(width, height)
    canvas = _add_bokeh(canvas, rng)
    canvas = _add_sparkles(canvas, rng)
    canvas = _add_grain(canvas, amount=2.6, seed=seed % 1000)
    canvas = canvas.convert("RGBA")
    canvas.alpha_composite(_bottom_scrim(width, height))
    canvas = _composite_text_block(canvas.convert("RGB"), title, subtitle, width, height)
    canvas.save(out, format="PNG", optimize=True)
    return out


# --------------------------------------------------------------------------- #
# 命令行：单独生成封面
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="用 CG 图生成 1920x1080 视频封面（必须提供图片）")
    ap.add_argument("image", nargs="?", help="角色 CG 图路径")
    ap.add_argument("--image", "-i", dest="image_opt", default=None, help="角色 CG 图路径（同位置参数）")
    ap.add_argument("--out", "-o", default="cover.png", help="输出 PNG 路径")
    ap.add_argument("--title", "-t", default="", help="主标题（角色名），留空则不出标题")
    ap.add_argument("--subtitle", "-s", default="", help="副标题，留空则不出副标题")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--mode", choices=COVER_MODES, default="crop", help="铺满方式（默认 crop）")
    ap.add_argument("--blur-strength", type=float, default=24, help="blur 模式背景模糊半径（默认 24）")
    ap.add_argument("--placeholder", action="store_true", help="调试：不提供图片，生成渐变占位封面")
    args = ap.parse_args(argv)

    image = args.image_opt or args.image
    if image is None and not args.placeholder:
        print("[错误] 需要提供 CG 图（--image 路径）；封面不再自动生成。")
        return 2

    if args.placeholder:
        path = render_placeholder_cover(
            args.out, title=args.title, subtitle=args.subtitle, width=args.width, height=args.height
        )
    else:
        path = render_cover(
            image,
            args.out,
            title=args.title,
            subtitle=args.subtitle,
            width=args.width,
            height=args.height,
            cover_mode=args.mode,
            blur_strength=args.blur_strength,
        )
    with Image.open(path) as im:
        print(f"封面已生成: {path}  ({im.width}x{im.height} {im.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
