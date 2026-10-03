# -*- coding: utf-8 -*-
"""
Telegram Passport Photo Printing Bot
====================================
Single-file production bot: aiogram 3.x + Pillow + numpy + optional rembg
background removal + embedded Telegram Mini App served via aiohttp.

Required environment variables:
    BOT_TOKEN   - Telegram bot token from @BotFather

Optional environment variables:
    WEBAPP_URL  - public HTTPS URL of this service (Railway domain). If set,
                  the "Open Photo Maker" Mini App button is shown.
    OWNER_ID    - numeric Telegram id of the owner (reserved, optional)
    PORT        - HTTP port for the aiohttp server (Railway provides it)
"""

import asyncio
import binascii
import io
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from aiohttp import web

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    WebAppInfo,
)

# Optional advanced background removal. It is loaded lazily so Render can bind
# its HTTP port immediately instead of waiting for the ONNX model at boot.
_rembg_remove = None
_REMBG_CHECKED = False
REMBG_AVAILABLE = False


def _load_rembg() -> bool:
    global _rembg_remove, _REMBG_CHECKED, REMBG_AVAILABLE
    if _REMBG_CHECKED:
        return REMBG_AVAILABLE
    _REMBG_CHECKED = True
    try:
        from rembg import remove
        _rembg_remove = remove
        REMBG_AVAILABLE = True
    except Exception:
        _rembg_remove = None
        REMBG_AVAILABLE = False
    return REMBG_AVAILABLE

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
WEBAPP_URL = os.getenv("WEBAPP_URL", "").strip().rstrip("/")
OWNER_ID = os.getenv("OWNER_ID", "").strip()
def _read_port() -> int:
    raw = os.getenv("PORT", "8080").strip()
    try:
        port = int(raw)
    except ValueError:
        logging.getLogger("passport-bot").warning("Invalid PORT=%r; using 8080", raw)
        return 8080
    return port if 1 <= port <= 65535 else 8080


PORT = _read_port()

DPI = 300                      # print quality
MARGIN_MM = 10.0               # page margin on every side
SPACING_MM = 5.0               # gap between photos
BORDER_RGB = (45, 45, 45)      # thin clean border around each photo
JPEG_QUALITY = 95

MAX_QTY = 500
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_SIDE = 8000

MM_MIN_PAGE, MM_MAX_PAGE = 50.0, 1200.0
MM_MIN_PHOTO, MM_MAX_PHOTO = 10.0, 400.0

PAGE_SIZES = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A5": (148.0, 210.0),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
}

PHOTO_SIZES = {
    "25 × 35 mm": (25.0, 35.0),
    "30 × 40 mm": (30.0, 40.0),
    "35 × 45 mm": (35.0, 45.0),
    "2 × 2 inch": (50.8, 50.8),
}

BG_PRESETS = {
    "blue": ("🔵 Blue", (67, 142, 219)),
    "red": ("🔴 Red", (211, 47, 47)),
    "white": ("⚪ White", (255, 255, 255)),
    "green": ("🟢 Green", (46, 125, 50)),
}

HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}

TMP_ROOT = Path(tempfile.gettempdir()) / "passport_photo_bot"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# Protect the public Mini App endpoint from accidental overload on small Render instances.
try:
    GENERATION_CONCURRENCY = max(1, int(os.getenv("GENERATION_CONCURRENCY", "1") or "1"))
except ValueError:
    GENERATION_CONCURRENCY = 1
_generation_semaphore = asyncio.Semaphore(GENERATION_CONCURRENCY)
_started_at = time.time()
_generation_count = 0
log = logging.getLogger("passport-bot")

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return max(1, int(round(mm / 25.4 * dpi)))


def unit_to_mm(value: float, unit: str) -> float:
    if unit == "cm":
        return value * 10.0
    if unit == "inch":
        return value * 25.4
    return value  # mm


def parse_float(text: str):
    try:
        v = float(text.strip().replace(",", "."))
        if math.isfinite(v):
            return v
    except (ValueError, AttributeError):
        pass
    return None


def parse_hex_color(text: str):
    """Return (r, g, b) for '#RRGGBB' / 'RRGGBB', else None."""
    if not text:
        return None
    t = text.strip()
    if not HEX_RE.match(t):
        return None
    t = t.lstrip("#")
    return (int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16))


def rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def user_dir(user_id: int) -> Path:
    d = TMP_ROOT / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_user(user_id: int) -> None:
    try:
        shutil.rmtree(TMP_ROOT / str(user_id), ignore_errors=True)
    except Exception:
        log.warning("cleanup failed for user %s", user_id)


def compute_grid(page_w_mm, page_h_mm, pw_mm, ph_mm):
    """Dynamic layout: how many columns/rows of pw x ph photos fit inside the
    printable area. Returns (cols, rows) or None if even one photo can't fit."""
    printable_w = page_w_mm - 2 * MARGIN_MM
    printable_h = page_h_mm - 2 * MARGIN_MM
    if pw_mm > printable_w or ph_mm > printable_h:
        return None
    cols = int((printable_w + SPACING_MM) // (pw_mm + SPACING_MM))
    rows = int((printable_h + SPACING_MM) // (ph_mm + SPACING_MM))
    if cols < 1 or rows < 1:
        return None
    return cols, rows

# --------------------------------------------------------------------------- #
# Image processing (shared by Telegram bot AND Mini App API)
# --------------------------------------------------------------------------- #

def load_image(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image file is too large (max 25 MB).")
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img.load()
    if max(img.size) > MAX_IMAGE_SIDE:
        raise ValueError("Image dimensions are too large.")
    if img.mode in ("RGBA", "LA", "P", "CMYK", "I;16", "I", "F", "L"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _fallback_bg_remove(img: Image.Image, rgb) -> Image.Image:
    """Safe fallback when advanced background removal is unavailable/fails:
    distance-to-border-color segmentation with a feathered mask."""
    arr = np.asarray(img).astype(np.int16)
    h, w, _ = arr.shape
    b = max(4, min(h, w) // 40)
    border = np.concatenate([
        arr[:b, :, :].reshape(-1, 3),
        arr[-b:, :, :].reshape(-1, 3),
        arr[:, :b, :].reshape(-1, 3),
        arr[:, -b:, :].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    fg = (dist > 45).astype(np.uint8) * 255
    mask = Image.fromarray(fg, "L").filter(ImageFilter.GaussianBlur(2))
    bg_img = Image.new("RGB", img.size, rgb)
    return Image.composite(img, bg_img, mask)


def replace_background(img: Image.Image, rgb) -> Image.Image:
    """Intelligently remove the existing background and apply rgb.
    Uses rembg when available; falls back gracefully, never crashes."""
    if _load_rembg():
        try:
            out = _rembg_remove(img)  # RGBA with alpha matte
            if isinstance(out, bytes):
                out = Image.open(io.BytesIO(out)).convert("RGBA")
            else:
                out = out.convert("RGBA")
            if out.size != img.size:
                out = out.resize(img.size, Image.LANCZOS)
            alpha = out.getchannel("A").filter(ImageFilter.GaussianBlur(0.6))
            bg_img = Image.new("RGB", img.size, rgb)
            fg = out.convert("RGB")
            return Image.composite(fg, bg_img, alpha)
        except Exception:
            log.warning("rembg background removal failed; using fallback", exc_info=True)
    try:
        return _fallback_bg_remove(img, rgb)
    except Exception:
        log.warning("fallback background removal failed; keeping original", exc_info=True)
        return img


def crop_to_ratio(img: Image.Image, ratio: float) -> Image.Image:
    """Aspect-ratio-preserving intelligent crop (never stretches).
    Keeps the upper part of the frame so the head/face is not cut off."""
    w, h = img.size
    cur = w / h
    if abs(cur - ratio) < 1e-3:
        return img
    if cur > ratio:  # too wide -> crop sides, centered
        new_w = int(round(h * ratio))
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    # too tall -> crop from bottom with an upper bias to protect the head
    new_h = int(round(w / ratio))
    y0 = int(round((h - new_h) * 0.30))
    y0 = max(0, min(h - new_h, y0))
    return img.crop((0, y0, w, y0 + new_h))


def build_passport_photo(img: Image.Image, bg_rgb, pw_mm, ph_mm) -> Image.Image:
    if bg_rgb is not None:
        img = replace_background(img, bg_rgb)
    img = crop_to_ratio(img, pw_mm / ph_mm)
    target = (mm_to_px(pw_mm), mm_to_px(ph_mm))
    if img.size != target:
        img = img.resize(target, Image.LANCZOS)
    return img


def build_sheet(photo: Image.Image, page_w_mm, page_h_mm,
                pw_mm, ph_mm, count) -> Image.Image:
    """Render one print sheet: clean white page, dynamically computed grid,
    centered, evenly spaced, thin consistent borders. Never overflows."""
    grid = compute_grid(page_w_mm, page_h_mm, pw_mm, ph_mm)
    if grid is None:
        raise ValueError("Selected photo size does not fit on the selected page.")
    cols, _rows = grid
    count = max(1, min(count, cols * _rows))

    W, H = mm_to_px(page_w_mm), mm_to_px(page_h_mm)
    pw, ph = photo.size
    sp = mm_to_px(SPACING_MM)
    border = max(1, DPI // 150)

    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    rows_used = math.ceil(count / cols)
    grid_h = rows_used * ph + (rows_used - 1) * sp
    top = (H - grid_h) // 2

    for i in range(count):
        row, col = divmod(i, cols)
        items_in_row = min(cols, count - row * cols)
        row_w = items_in_row * pw + (items_in_row - 1) * sp
        left = (W - row_w) // 2
        x = left + col * (pw + sp)
        y = top + row * (ph + sp)
        sheet.paste(photo, (x, y))
        draw.rectangle([x - border, y - border,
                        x + pw - 1 + border, y + ph - 1 + border],
                       outline=BORDER_RGB, width=border)
    return sheet


def generate_files(image_bytes: bytes, page_w_mm: float, page_h_mm: float,
                   bg_rgb, pw_mm: float, ph_mm: float,
                   qty: int, fmt: str):
    """Full pipeline shared by bot & Mini App.
    Returns (list_of_(filename, bytes), num_pages). Raises ValueError on
    invalid input that can't be produced."""
    img = load_image(image_bytes)
    grid = compute_grid(page_w_mm, page_h_mm, pw_mm, ph_mm)
    if grid is None:
        raise ValueError(
            "Selected photo size does not fit on the selected page. "
            "Please choose a smaller photo size or a bigger page.")
    cols, rows = grid
    capacity = cols * rows
    pages = math.ceil(qty / capacity)

    passport = build_passport_photo(img, bg_rgb, pw_mm, ph_mm)

    sheets = []
    remaining = qty
    for _ in range(pages):
        take = min(capacity, remaining)
        sheets.append(build_sheet(passport, page_w_mm, page_h_mm,
                                  pw_mm, ph_mm, take))
        remaining -= take

    stamp = time.strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]
    base = f"passport_photos_{stamp}_{uid}"
    files = []

    def to_jpeg(sheet, name):
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=0,
                   dpi=(DPI, DPI), optimize=True)
        files.append((name, buf.getvalue()))

    def to_png(sheet, name):
        buf = io.BytesIO()
        sheet.save(buf, "PNG", dpi=(DPI, DPI))
        files.append((name, buf.getvalue()))

    def to_pdf():
        buf = io.BytesIO()
        first, rest = sheets[0], sheets[1:]
        first.save(buf, "PDF", resolution=float(DPI), save_all=True,
                   append_images=rest)
        files.append((f"{base}.pdf", buf.getvalue()))

    if fmt in ("pdf", "pdf_jpg", "pdf_png"):
        to_pdf()
    if fmt in ("jpg", "pdf_jpg"):
        for i, s in enumerate(sheets, 1):
            to_jpeg(s, f"{base}_p{i}.jpg")
    if fmt in ("png", "pdf_png"):
        for i, s in enumerate(sheets, 1):
            to_png(s, f"{base}_p{i}.png")
    return files, pages

# --------------------------------------------------------------------------- #
# FSM states
# --------------------------------------------------------------------------- #

class Flow(StatesGroup):
    page_custom_w = State()
    page_custom_h = State()
    page_custom_u = State()
    bg_custom_hex = State()
    ps_custom_w = State()
    ps_custom_h = State()
    ps_custom_u = State()
    qty_custom = State()

# --------------------------------------------------------------------------- #
# Keyboards
# --------------------------------------------------------------------------- #

def _rows(*btns_per_row):
    return [[InlineKeyboardButton(text=t, callback_data=c) for t, c in row]
            for row in btns_per_row if row]


def kb_page():
    rows = _rows([("A4", "pg:A4"), ("A3", "pg:A3"), ("A5", "pg:A5")],
                 [("Letter", "pg:Letter"), ("Legal", "pg:Legal")],
                 [("📐 Custom Size", "pg:custom")])
    rows.append([InlineKeyboardButton(text="❌ Cancel", callback_data="flow:cancel")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_unit(prefix, back_cb):
    rows = _rows([("mm", f"{prefix}:mm"), ("cm", f"{prefix}:cm"),
                  ("inch", f"{prefix}:inch")])
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data=back_cb)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_bg():
    rows = _rows([("🔵 Blue", "bg:blue"), ("🔴 Red", "bg:red")],
                 [("⚪ White", "bg:white"), ("🟢 Green", "bg:green")],
                 [("🟣 Custom (HEX)", "bg:custom"), ("⏭ Skip", "bg:skip")])
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="nav:page")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_photo_size():
    rows = _rows([("25 × 35 mm", "ps:25x35"), ("30 × 40 mm", "ps:30x40")],
                 [("35 × 45 mm", "ps:35x45"), ("2 × 2 inch", "ps:2x2")],
                 [("📐 Custom", "ps:custom")])
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="nav:bg")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_qty():
    rows = _rows([("1", "qt:1"), ("2", "qt:2"), ("3", "qt:3"), ("4", "qt:4")],
                 [("6", "qt:6"), ("8", "qt:8"), ("10", "qt:10"), ("12", "qt:12")],
                 [("🔢 Custom", "qt:custom")])
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="nav:ps")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_fmt():
    rows = _rows([("📄 PDF", "fm:pdf")],
                 [("🖼 JPEG", "fm:jpg"), ("🖼 PNG", "fm:png")],
                 [("📄 PDF + JPEG", "fm:pdf_jpg"),
                  ("📄 PDF + PNG", "fm:pdf_png")])
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="nav:qty")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_summary():
    return InlineKeyboardMarkup(inline_keyboard=_rows(
        [("✅ Generate", "sum:generate")],
        [("✏️ Change Settings", "sum:change"), ("❌ Cancel", "flow:cancel")],
    ))


def kb_change():
    rows = _rows(
        [("📄 Page Size", "chg:page"), ("🎨 Background", "chg:bg")],
        [("📐 Photo Size", "chg:ps"), ("🖼 Quantity", "chg:qty")],
        [("📦 Output Format", "chg:fmt")])
    rows.append([InlineKeyboardButton(text="⬅️ Back to Summary",
                                      callback_data="nav:summary")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_fit_error():
    rows = _rows(
        [("📐 Change Photo Size", "chg:ps"), ("📄 Change Page", "chg:page")],
        [("🖼 Change Quantity", "chg:qty")],
        [("❌ Cancel", "flow:cancel")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_main_menu():
    rows = [[InlineKeyboardButton(text="📸 Create Passport Photos",
                                  callback_data="menu:create")]]
    if WEBAPP_URL:
        rows.append([InlineKeyboardButton(
            text="📱 Open Photo Maker",
            web_app=WebAppInfo(url=WEBAPP_URL))])
    return InlineKeyboardMarkup(inline_keyboard=rows)

# --------------------------------------------------------------------------- #
# Bot handlers
# --------------------------------------------------------------------------- #

router = Router()

WELCOME = (
    "👋 <b>Passport Photo Maker</b>\n\n"
    "Send me any photo and I'll turn it into a professional, print-ready "
    "passport photo sheet (PDF / JPEG / PNG, 300 DPI).\n\n"
    "Use the menu below to begin."
)

HELP_TEXT = (
    "ℹ️ <b>How to use</b>\n\n"
    "1️⃣ Tap <b>📸 Create Passport Photos</b> or simply send a photo.\n"
    "2️⃣ Choose the <b>page size</b> (A4/A3/A5/Letter/Legal/Custom).\n"
    "3️⃣ Optionally <b>change the background</b> (Blue/Red/White/Green/HEX/Skip).\n"
    "4️⃣ Select the <b>passport photo size</b> (e.g. 35 × 45 mm).\n"
    "5️⃣ Choose <b>how many photos</b> you need.\n"
    "6️⃣ Pick the <b>output format</b> (PDF/JPEG/PNG) and press <b>Generate</b>.\n\n"
    "🖨 Output is print-ready at <b>300 DPI</b> with exact physical sizes.\n"
    "Commands: /start • /help • /cancel"
)

PAGE_PROMPT = "📄 <b>SELECT PAGE SIZE</b>"
BG_PROMPT = "🎨 <b>CHANGE BACKGROUND?</b>"
PS_PROMPT = "📐 <b>SELECT PASSPORT PHOTO SIZE</b>"
QTY_PROMPT = "🖼 <b>HOW MANY PHOTOS?</b>"
FMT_PROMPT = "📦 <b>SELECT OUTPUT FORMAT</b>"


async def safe_edit(cq: CallbackQuery, text: str, kb=None):
    try:
        await cq.message.edit_text(text, reply_markup=kb)
    except TelegramAPIError:
        try:
            await cq.message.answer(text, reply_markup=kb)
        except TelegramAPIError:
            pass


async def get_data_photo_path(state: FSMContext):
    data = await state.get_data()
    p = data.get("img_path")
    if not p:
        return None, data
    path = Path(p)
    # Path-traversal guard: only files inside our tmp root are acceptable.
    try:
        path.resolve().relative_to(TMP_ROOT.resolve())
    except ValueError:
        return None, data
    if not path.is_file():
        return None, data
    return path, data


async def summary_text(state: FSMContext) -> str:
    data = await state.get_data()
    page_name = data.get("page_name", "Custom")
    pw, ph = data.get("page_w", 0), data.get("page_h", 0)
    psw, psh = data.get("ps_w", 0), data.get("ps_h", 0)
    qty = data.get("qty", 0)
    fmt = data.get("fmt", "pdf")
    bg = data.get("bg")
    bg_name = data.get("bg_name", "Skip (original)")

    fmt_names = {"pdf": "PDF", "jpg": "JPEG", "png": "PNG",
                 "pdf_jpg": "PDF + JPEG", "pdf_png": "PDF + PNG"}

    lines = [
        "✅ <b>PHOTO READY</b>\n",
        f"📄 Page: {page_name} ({pw:g} × {ph:g} mm)",
        f"📐 Photo Size: {psw:g} × {psh:g} mm",
        f"🖼 Quantity: {qty}",
        f"🎨 Background: {bg_name}",
        f"📦 Format: {fmt_names.get(fmt, fmt)}",
        f"🖨 Quality: {DPI} DPI",
    ]

    grid = compute_grid(pw, ph, psw, psh)
    if grid is None:
        lines.append("\n⚠️ <b>This photo size does not fit on the selected "
                     "page.</b> Use ✏️ Change Settings to pick a smaller "
                     "photo size or a bigger page.")
    else:
        cols, rows = grid
        capacity = cols * rows
        if qty > capacity:
            pages = math.ceil(qty / capacity)
            lines.append(f"\n🗒 Layout: {cols} × {rows} per page → "
                         f"<b>{pages} pages</b> will be generated.")
        else:
            lines.append(f"\n🗒 Layout: {cols} × {rows} grid on 1 page.")
    return "\n".join(lines)


async def show_summary(cq_or_msg, state: FSMContext):
    text = await summary_text(state)
    kb = kb_summary()
    if isinstance(cq_or_msg, CallbackQuery):
        await safe_edit(cq_or_msg, text, kb)
    else:
        await cq_or_msg.answer(text, reply_markup=kb)


async def after_setting_selected(cq_or_msg, state: FSMContext,
                                 next_step: str):
    """If the user came from ✏️ Change Settings, jump straight back to the
    summary; otherwise continue the normal forward flow."""
    data = await state.get_data()
    editing = data.get("editing", False)
    if editing:
        await state.update_data(editing=False)
        await show_summary(cq_or_msg, state)
        return
    prompts = {"bg": (BG_PROMPT, kb_bg()), "ps": (PS_PROMPT, kb_photo_size()),
               "qty": (QTY_PROMPT, kb_qty()), "fmt": (FMT_PROMPT, kb_fmt())}
    text, kb = prompts[next_step]
    if isinstance(cq_or_msg, CallbackQuery):
        await safe_edit(cq_or_msg, text, kb)
    else:
        await cq_or_msg.answer(text, reply_markup=kb)


# ----- /start, /help, /cancel -------------------------------------------- #

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(WELCOME, reply_markup=kb_main_menu())


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(HELP_TEXT, reply_markup=kb_main_menu())


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    cleanup_user(message.from_user.id)
    await message.answer("❌ Cancelled. Temporary files removed.",
                         reply_markup=kb_main_menu())


@router.callback_query(F.data == "menu:create")
async def cb_menu_create(cq: CallbackQuery, state: FSMContext):
    await state.clear()
    await safe_edit(cq, "📸 Send me your photo now — as a <b>photo</b> or as an "
                        "<b>image file/document</b>. I'll use the highest "
                        "available quality.")
    await cq.answer()


@router.callback_query(F.data == "flow:cancel")
async def cb_flow_cancel(cq: CallbackQuery, state: FSMContext):
    await state.clear()
    cleanup_user(cq.from_user.id)
    await safe_edit(cq, "❌ Cancelled. Temporary files removed.")
    await cq.message.answer(WELCOME, reply_markup=kb_main_menu())
    await cq.answer()


# ----- photo intake ------------------------------------------------------- #

async def _intake(message: Message, state: FSMContext, file_id: str,
                  fname: str):
    status = await message.answer("⏳ Downloading your photo…")
    try:
        tg_file = await message.bot.get_file(file_id)
        if tg_file.file_size and tg_file.file_size > MAX_IMAGE_BYTES:
            await status.edit_text("⚠️ Image is too large (max 25 MB). "
                                   "Please send a smaller file.")
            return
        buf = await message.bot.download_file(tg_file.file_path)
        data = buf.read()
        try:
            img = load_image(data)  # validate early, keep original bytes
            img.close()
        except Exception:
            await status.edit_text(
                "⚠️ This doesn't look like a valid image. Please send a "
                "clear JPEG/PNG photo.")
            return

        d = user_dir(message.from_user.id)
        path = d / f"orig_{uuid.uuid4().hex[:8]}.bin"
        path.write_bytes(data)
        await state.clear()
        await state.update_data(img_path=str(path), img_name=fname)
        await status.edit_text("✅ Photo received!")
        await message.answer(PAGE_PROMPT, reply_markup=kb_page())
    except TelegramAPIError:
        log.warning("telegram download failed", exc_info=True)
        await status.edit_text("⚠️ Couldn't download the file from Telegram. "
                               "Please try sending it again.")
    except Exception:
        log.exception("photo intake failed")
        await status.edit_text("⚠️ Something went wrong while reading your "
                               "photo. Please try again.")


@router.message(F.photo)
async def on_photo(message: Message, state: FSMContext):
    biggest = message.photo[-1]  # highest available quality
    await _intake(message, state, biggest.file_id, "photo.jpg")


@router.message(F.document)
async def on_document(message: Message, state: FSMContext):
    doc = message.document
    mime = (doc.mime_type or "").lower()
    name = doc.file_name or "image"
    if not (mime in IMAGE_MIMES or mime.startswith("image/")):
        await message.answer("⚠️ Please send an <b>image</b> file "
                             "(JPEG/PNG/etc.).")
        return
    if doc.file_size and doc.file_size > MAX_IMAGE_BYTES:
        await message.answer("⚠️ Image is too large (max 25 MB).")
        return
    await _intake(message, state, doc.file_id, name)


# ----- navigation (Back buttons) ------------------------------------------ #

@router.callback_query(F.data.startswith("nav:"))
async def cb_nav(cq: CallbackQuery, state: FSMContext):
    target = cq.data.split(":", 1)[1]
    prompts = {"page": (PAGE_PROMPT, kb_page()), "bg": (BG_PROMPT, kb_bg()),
               "ps": (PS_PROMPT, kb_photo_size()), "qty": (QTY_PROMPT, kb_qty()),
               "fmt": (FMT_PROMPT, kb_fmt())}
    if target == "summary":
        await show_summary(cq, state)
    elif target in prompts:
        text, kb = prompts[target]
        await safe_edit(cq, text, kb)
    await cq.answer()


@router.callback_query(F.data.startswith("chg:"))
async def cb_change(cq: CallbackQuery, state: FSMContext):
    await state.update_data(editing=True)
    await cb_nav(cq, state)  # reuse the same prompts


@router.callback_query(F.data == "sum:change")
async def cb_sum_change(cq: CallbackQuery, state: FSMContext):
    await safe_edit(cq, "✏️ <b>Which setting do you want to change?</b>",
                    kb_change())
    await cq.answer()


# ----- step 1: page size --------------------------------------------------- #

async def _need_photo(cq: CallbackQuery, state: FSMContext) -> bool:
    path, _ = await get_data_photo_path(state)
    if path is None:
        await safe_edit(cq, "⌛ Your session expired. Please send the photo "
                            "again to start over.", kb_main_menu())
        await cq.answer()
        return True
    return False


@router.callback_query(F.data.startswith("pg:"))
async def cb_page(cq: CallbackQuery, state: FSMContext):
    if await _need_photo(cq, state):
        return
    val = cq.data.split(":", 1)[1]
    if val == "custom":
        await state.set_state(Flow.page_custom_w)
        await safe_edit(cq, "📐 Enter the page <b>width</b> (a number, e.g. "
                            "<code>210</code>). You'll choose the unit next.")
    else:
        w, h = PAGE_SIZES[val]
        await state.update_data(page_w=w, page_h=h, page_name=val)
        await after_setting_selected(cq, state, "bg")
    await cq.answer()


@router.message(Flow.page_custom_w)
async def msg_page_w(message: Message, state: FSMContext):
    v = parse_float(message.text or "")
    if v is None or v <= 0:
        await message.answer("⚠️ Please enter a valid positive number for the "
                             "page width (e.g. <code>210</code>).")
        return
    await state.update_data(tmp_page_w=v)
    await state.set_state(Flow.page_custom_h)
    await message.answer("📐 Now enter the page <b>height</b> (e.g. "
                         "<code>297</code>).")


@router.message(Flow.page_custom_h)
async def msg_page_h(message: Message, state: FSMContext):
    v = parse_float(message.text or "")
    if v is None or v <= 0:
        await message.answer("⚠️ Please enter a valid positive number for the "
                             "page height (e.g. <code>297</code>).")
        return
    await state.update_data(tmp_page_h=v)
    await state.set_state(Flow.page_custom_u)
    await message.answer("📏 Select the <b>unit</b>:",
                         reply_markup=kb_unit("pgu", "nav:page"))


@router.callback_query(F.data.startswith("pgu:"))
async def cb_page_unit(cq: CallbackQuery, state: FSMContext):
    unit = cq.data.split(":", 1)[1]
    data = await state.get_data()
    w_mm = unit_to_mm(data.get("tmp_page_w", 0), unit)
    h_mm = unit_to_mm(data.get("tmp_page_h", 0), unit)
    if not (MM_MIN_PAGE <= w_mm <= MM_MAX_PAGE and
            MM_MIN_PAGE <= h_mm <= MM_MAX_PAGE):
        await safe_edit(cq, "⚠️ Those dimensions look unreasonable "
                            f"({w_mm:g} × {h_mm:g} mm). Page sides must be "
                            f"between {MM_MIN_PAGE:g} and {MM_MAX_PAGE:g} mm. "
                            "Please try again.", kb_page())
        await state.set_state(None)
        await cq.answer()
        return
    await state.update_data(page_w=w_mm, page_h=h_mm,
                            page_name=f"Custom {w_mm:g}×{h_mm:g} mm")
    await state.set_state(None)
    await after_setting_selected(cq, state, "bg")
    await cq.answer()


# ----- step 2: background -------------------------------------------------- #

@router.callback_query(F.data.startswith("bg:"))
async def cb_bg(cq: CallbackQuery, state: FSMContext):
    if await _need_photo(cq, state):
        return
    val = cq.data.split(":", 1)[1]
    if val == "custom":
        await state.set_state(Flow.bg_custom_hex)
        await safe_edit(cq, "🟣 Send the background color as a <b>HEX code</b>, "
                            "e.g. <code>#FFFFFF</code>.")
    elif val == "skip":
        await state.update_data(bg=None, bg_name="Skip (original)")
        await after_setting_selected(cq, state, "ps")
    else:
        name, rgb = BG_PRESETS[val]
        await state.update_data(bg=rgb_to_hex(rgb), bg_name=name)
        await after_setting_selected(cq, state, "ps")
    await cq.answer()


@router.message(Flow.bg_custom_hex)
async def msg_bg_hex(message: Message, state: FSMContext):
    rgb = parse_hex_color(message.text or "")
    if rgb is None:
        await message.answer("⚠️ Invalid HEX color. Send it like "
                             "<code>#FFFFFF</code> (6 hex digits).")
        return
    await state.update_data(bg=rgb_to_hex(rgb), bg_name=f"Custom {rgb_to_hex(rgb)}")
    await state.set_state(None)
    await after_setting_selected(message, state, "ps")


# ----- step 3: passport photo size ------------------------------------------ #

_PS_CB = {"25x35": "25 × 35 mm", "30x40": "30 × 40 mm",
          "35x45": "35 × 45 mm", "2x2": "2 × 2 inch"}


@router.callback_query(F.data.startswith("ps:"))
async def cb_ps(cq: CallbackQuery, state: FSMContext):
    if await _need_photo(cq, state):
        return
    val = cq.data.split(":", 1)[1]
    if val == "custom":
        await state.set_state(Flow.ps_custom_w)
        await safe_edit(cq, "📐 Enter the photo <b>width</b> (a number, e.g. "
                            "<code>35</code>).")
    else:
        name = _PS_CB[val]
        w, h = PHOTO_SIZES[name]
        await state.update_data(ps_w=w, ps_h=h, ps_name=name)
        await after_setting_selected(cq, state, "qty")
    await cq.answer()


@router.message(Flow.ps_custom_w)
async def msg_ps_w(message: Message, state: FSMContext):
    v = parse_float(message.text or "")
    if v is None or v <= 0:
        await message.answer("⚠️ Please enter a valid positive number for the "
                             "photo width (e.g. <code>35</code>).")
        return
    await state.update_data(tmp_ps_w=v)
    await state.set_state(Flow.ps_custom_h)
    await message.answer("📐 Now enter the photo <b>height</b> (e.g. "
                         "<code>45</code>).")


@router.message(Flow.ps_custom_h)
async def msg_ps_h(message: Message, state: FSMContext):
    v = parse_float(message.text or "")
    if v is None or v <= 0:
        await message.answer("⚠️ Please enter a valid positive number for the "
                             "photo height (e.g. <code>45</code>).")
        return
    await state.update_data(tmp_ps_h=v)
    await state.set_state(Flow.ps_custom_u)
    await message.answer("📏 Select the <b>unit</b>:",
                         reply_markup=kb_unit("psu", "nav:ps"))


@router.callback_query(F.data.startswith("psu:"))
async def cb_ps_unit(cq: CallbackQuery, state: FSMContext):
    unit = cq.data.split(":", 1)[1]
    data = await state.get_data()
    w_mm = unit_to_mm(data.get("tmp_ps_w", 0), unit)
    h_mm = unit_to_mm(data.get("tmp_ps_h", 0), unit)
    if not (MM_MIN_PHOTO <= w_mm <= MM_MAX_PHOTO and
            MM_MIN_PHOTO <= h_mm <= MM_MAX_PHOTO):
        await safe_edit(cq, "⚠️ Those photo dimensions look unreasonable. "
                            "Please try again.", kb_photo_size())
        await state.set_state(None)
        await cq.answer()
        return
    await state.update_data(ps_w=w_mm, ps_h=h_mm,
                            ps_name=f"{w_mm:g} × {h_mm:g} mm")
    await state.set_state(None)
    await after_setting_selected(cq, state, "qty")
    await cq.answer()

# ----- step 4: quantity ----------------------------------------------------- #

@router.callback_query(F.data.startswith("qt:"))
async def cb_qty(cq: CallbackQuery, state: FSMContext):
    if await _need_photo(cq, state):
        return
    val = cq.data.split(":", 1)[1]
    if val == "custom":
        await state.set_state(Flow.qty_custom)
        await safe_edit(cq, f"🔢 Enter how many photos you need (1–{MAX_QTY}).")
    else:
        try:
            qty = int(val)
        except ValueError:
            await cq.answer("Invalid quantity.")
            return
        if not (1 <= qty <= MAX_QTY):
            await cq.answer("Invalid quantity.")
            return
        await state.update_data(qty=qty)
        await after_setting_selected(cq, state, "fmt")
    await cq.answer()


@router.message(Flow.qty_custom)
async def msg_qty(message: Message, state: FSMContext):
    text = (message.text or "").strip()
    if not text.isdigit():
        await message.answer(f"⚠️ Please enter a whole number between 1 and "
                             f"{MAX_QTY} (e.g. <code>5</code>).")
        return
    qty = int(text)
    if not (1 <= qty <= MAX_QTY):
        await message.answer(f"⚠️ Quantity must be between 1 and {MAX_QTY}. "
                             "Try again.")
        return
    await state.update_data(qty=qty)
    await state.set_state(None)
    await after_setting_selected(message, state, "fmt")


# ----- step 5: output format ------------------------------------------------ #

@router.callback_query(F.data.startswith("fm:"))
async def cb_fmt(cq: CallbackQuery, state: FSMContext):
    if await _need_photo(cq, state):
        return
    val = cq.data.split(":", 1)[1]
    if val not in ("pdf", "jpg", "png", "pdf_jpg", "pdf_png"):
        await cq.answer("Invalid format.")
        return
    await state.update_data(fmt=val)
    await show_summary(cq, state)
    await cq.answer()


# ----- generate --------------------------------------------------------------- #

@router.callback_query(F.data == "sum:generate")
async def cb_generate(cq: CallbackQuery, state: FSMContext):
    path, data = await get_data_photo_path(state)
    if path is None:
        await safe_edit(cq, "⌛ Your session expired or the photo was cleaned "
                            "up. Please send it again to start over.",
                        kb_main_menu())
        await cq.answer()
        return

    required = ("page_w", "page_h", "ps_w", "ps_h", "qty", "fmt")
    if any(k not in data for k in required):
        await safe_edit(cq, "⚠️ Some settings are missing. Let's go through "
                            "them again.", kb_page())
        await cq.answer()
        return

    page_w, page_h = data["page_w"], data["page_h"]
    ps_w, ps_h = data["ps_w"], data["ps_h"]
    qty, fmt = data["qty"], data["fmt"]
    bg_hex = data.get("bg")
    bg_rgb = parse_hex_color(bg_hex) if bg_hex else None

    grid = compute_grid(page_w, page_h, ps_w, ps_h)
    if grid is None:
        await safe_edit(cq,
            "⚠️ <b>Selected photo size does not fit on the selected page.</b>\n"
            "I won't shrink photos — your print size would be wrong. "
            "Please adjust one of these:", kb_fit_error())
        await cq.answer()
        return

    await safe_edit(cq, "⏳ <b>Processing your photo…</b>\nThis can take a "
                        "moment, especially with background removal.")
    await cq.answer()

    try:
        image_bytes = path.read_bytes()
        loop = asyncio.get_running_loop()
        files, pages = await loop.run_in_executor(
            None, generate_files, image_bytes, page_w, page_h,
            bg_rgb, ps_w, ps_h, qty, fmt)
    except ValueError as exc:
        await cq.message.answer(f"⚠️ {exc}")
        return
    except Exception:
        log.exception("generation failed")
        await cq.message.answer("⚠️ Sorry, something went wrong while "
                                "generating your files. Please try again.")
        return

    try:
        caption = (f"🖨 Done! {qty} photo(s), {pages} page(s), "
                   f"{ps_w:g} × {ps_h:g} mm @ {DPI} DPI.")
        for fname, blob in files:
            await cq.message.answer_document(
                BufferedInputFile(blob, filename=fname), caption=caption)
        await cq.message.answer("✅ All files sent! Send another photo "
                                "anytime to make a new sheet.",
                                reply_markup=kb_main_menu())
    except TelegramAPIError:
        log.warning("failed to send output files", exc_info=True)
        await cq.message.answer("⚠️ Files were generated but couldn't be "
                                "sent (they may be too large for Telegram). "
                                "Try fewer pages or a different format.")
    finally:
        await state.clear()
        cleanup_user(cq.from_user.id)


# ----- catch-all for stray input --------------------------------------------- #

@router.message()
async def fallback_message(message: Message):
    await message.answer("Send me a <b>photo</b> to create a passport photo "
                         "sheet, or use /start.", reply_markup=kb_main_menu())


@router.callback_query()
async def fallback_callback(cq: CallbackQuery):
    await cq.answer("That action is no longer available. Send a photo or use "
                    "/start to begin again.", show_alert=False)

# --------------------------------------------------------------------------- #
# Telegram Mini App (all HTML/CSS/JS embedded right here)
# --------------------------------------------------------------------------- #

MINIAPP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Passport Photo Maker</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<style>
:root{--bg:#0f1115;--card:#1a1e27;--accent:#4da3ff;--txt:#f2f5fa;--mut:#9aa3b2;--ok:#2ecc71;--err:#ff6b6b}
*{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
body{background:var(--bg);color:var(--txt);min-height:100vh;padding:16px}
h1{font-size:20px;margin-bottom:4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
.step{display:none;animation:fade .2s ease}
.step.on{display:block}
@keyframes fade{from{opacity:0}to{opacity:1}}
.card{background:var(--card);border-radius:14px;padding:16px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button.opt{background:#242a37;color:var(--txt);border:1px solid #303849;border-radius:12px;
  padding:16px 10px;font-size:15px;cursor:pointer;transition:.15s;min-height:52px}
button.opt.sel{border-color:var(--accent);background:#233048;color:#fff}
button.opt:active{transform:scale(.97)}
input[type=text],input[type=number],input[type=file]{width:100%;padding:14px;border-radius:12px;
  border:1px solid #303849;background:#11141b;color:var(--txt);font-size:15px;margin:8px 0}
.nav{display:flex;gap:10px;margin-top:6px}
.nav button{flex:1;padding:15px;border-radius:12px;border:none;font-size:15px;font-weight:600;cursor:pointer;min-height:50px}
.back{background:#2a3040;color:var(--txt)}
.next{background:var(--accent);color:#06121f}
.gen{background:var(--ok);color:#04150a}
.err{color:var(--err);font-size:13px;min-height:18px;margin:6px 0}
.info{font-size:14px;line-height:1.7}
.badge{display:inline-block;background:#233048;border-radius:8px;padding:2px 8px;font-size:12px;color:var(--accent)}
#preview{max-width:100%;border-radius:12px;margin-top:10px}
#dz{border:2px dashed #3a445a;border-radius:14px;padding:30px;text-align:center;color:var(--mut);cursor:pointer}
#spin{display:none;text-align:center;padding:20px;color:var(--mut)}
.lds{display:inline-block;width:26px;height:26px;border:3px solid #3a445a;border-top-color:var(--accent);border-radius:50%;animation:sp 1s linear infinite;vertical-align:middle}
@keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<h1>📸 Passport Photo Maker</h1>
<div class="sub">Print-ready sheets • 300 DPI • exact physical sizes</div>

<!-- Step 1: upload -->
<div class="step on" id="s1">
  <div class="card">
    <div id="dz">Tap to choose a photo<br><small>JPG / PNG</small></div>
    <input type="file" id="file" accept="image/*" style="display:none">
    <img id="preview" style="display:none">
    <div class="err" id="e1"></div>
  </div>
  <div class="nav"><button class="next" onclick="go(2)" id="n1" disabled>Next ➜</button></div>
</div>

<!-- Step 2: page size -->
<div class="step" id="s2">
  <div class="card">
    <h3>📄 Page size</h3><br>
    <div class="grid" id="pageGrid">
      <button class="opt" data-v="A4">A4<br><small>210 × 297 mm</small></button>
      <button class="opt" data-v="A3">A3<br><small>297 × 420 mm</small></button>
      <button class="opt" data-v="A5">A5<br><small>148 × 210 mm</small></button>
      <button class="opt" data-v="Letter">Letter</button>
      <button class="opt" data-v="Legal">Legal</button>
      <button class="opt" data-v="custom">📐 Custom</button>
    </div>
    <div id="pageCustom" style="display:none">
      <input type="number" id="pw" placeholder="Width" min="1">
      <input type="number" id="ph" placeholder="Height" min="1">
      <div class="grid">
        <button class="opt" data-u="mm">mm</button>
        <button class="opt" data-u="cm">cm</button>
        <button class="opt" data-u="inch">inch</button>
      </div>
    </div>
    <div class="err" id="e2"></div>
  </div>
  <div class="nav"><button class="back" onclick="go(1)">⬅ Back</button><button class="next" onclick="go(3)">Next ➜</button></div>
</div>

<!-- Step 3: background -->
<div class="step" id="s3">
  <div class="card">
    <h3>🎨 Background</h3><br>
    <div class="grid" id="bgGrid">
      <button class="opt" data-v="skip">⏭ Keep original</button>
      <button class="opt" data-v="blue">🔵 Blue</button>
      <button class="opt" data-v="red">🔴 Red</button>
      <button class="opt" data-v="white">⚪ White</button>
      <button class="opt" data-v="green">🟢 Green</button>
      <button class="opt" data-v="custom">🟣 Custom HEX</button>
    </div>
    <div id="bgCustom" style="display:none"><input type="text" id="hex" placeholder="#FFFFFF" maxlength="7"></div>
    <div class="err" id="e3"></div>
  </div>
  <div class="nav"><button class="back" onclick="go(2)">⬅ Back</button><button class="next" onclick="go(4)">Next ➜</button></div>
</div>

<!-- Step 4: photo size -->
<div class="step" id="s4">
  <div class="card">
    <h3>📐 Passport photo size</h3><br>
    <div class="grid" id="psGrid">
      <button class="opt" data-v="25x35">25 × 35 mm</button>
      <button class="opt" data-v="30x40">30 × 40 mm</button>
      <button class="opt" data-v="35x45">35 × 45 mm</button>
      <button class="opt" data-v="2x2">2 × 2 inch</button>
      <button class="opt" data-v="custom">📐 Custom</button>
    </div>
    <div id="psCustom" style="display:none">
      <input type="number" id="psw" placeholder="Width" min="1">
      <input type="number" id="psh" placeholder="Height" min="1">
      <div class="grid">
        <button class="opt" data-u="mm">mm</button>
        <button class="opt" data-u="cm">cm</button>
        <button class="opt" data-u="inch">inch</button>
      </div>
    </div>
    <div class="err" id="e4"></div>
  </div>
  <div class="nav"><button class="back" onclick="go(3)">⬅ Back</button><button class="next" onclick="go(5)">Next ➜</button></div>
</div>

<!-- Step 5: quantity -->
<div class="step" id="s5">
  <div class="card">
    <h3>🖼 How many photos?</h3><br>
    <div class="grid" id="qtyGrid">
      <button class="opt" data-v="1">1</button><button class="opt" data-v="2">2</button>
      <button class="opt" data-v="3">3</button><button class="opt" data-v="4">4</button>
      <button class="opt" data-v="6">6</button><button class="opt" data-v="8">8</button>
      <button class="opt" data-v="10">10</button><button class="opt" data-v="12">12</button>
      <button class="opt" data-v="custom">🔢 Custom</button>
    </div>
    <div id="qtyCustom" style="display:none"><input type="number" id="qcustom" placeholder="e.g. 15" min="1"></div>
    <div class="err" id="e5"></div>
  </div>
  <div class="nav"><button class="back" onclick="go(4)">⬅ Back</button><button class="next" onclick="go(6)">Next ➜</button></div>
</div>

<!-- Step 6: format -->
<div class="step" id="s6">
  <div class="card">
    <h3>📦 Output format</h3><br>
    <div class="grid" id="fmtGrid">
      <button class="opt" data-v="pdf">📄 PDF</button>
      <button class="opt" data-v="jpg">🖼 JPEG</button>
      <button class="opt" data-v="png">🖼 PNG</button>
      <button class="opt" data-v="pdf_jpg">📄 PDF + JPEG</button>
      <button class="opt" data-v="pdf_png">📄 PDF + PNG</button>
    </div>
    <div class="err" id="e6"></div>
  </div>
  <div class="nav"><button class="back" onclick="go(5)">⬅ Back</button><button class="next" onclick="go(7)">Next ➜</button></div>
</div>

<!-- Step 7: preview / generate -->
<div class="step" id="s7">
  <div class="card info" id="summary"></div>
  <div class="err" id="e7"></div>
  <div id="spin"><span class="lds"></span> Processing… (background removal can take a minute)</div>
  <div class="nav"><button class="back" onclick="go(6)">⬅ Back</button><button class="gen" id="genBtn" onclick="generate()">✅ Generate</button></div>
</div>

<script>
const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;
if (tg){ tg.ready(); tg.expand(); }
const P = {file:null, page:"A4", pw:null, ph:null, pu:"mm", bg:"skip", hex:null,
           ps:"35x45", psw:null, psh:null, psu:"mm", qty:12, fmt:"pdf"};

function go(n){
  if(!validate(cur())) return;
  document.querySelectorAll(".step").forEach(s=>s.classList.remove("on"));
  document.getElementById("s"+n).classList.add("on");
  if(n===7) renderSummary();
  window.scrollTo(0,0);
}
function cur(){ for(let i=1;i<=7;i++) if(document.getElementById("s"+i).classList.contains("on")) return i; return 1; }
function err(id,msg){ document.getElementById("e"+id).textContent = msg||""; }

// upload
const dz=document.getElementById("dz"), fi=document.getElementById("file");
dz.onclick=()=>fi.click();
fi.onchange=()=>{
  const f=fi.files[0]; err(1);
  if(!f) return;
  if(f.size>25*1024*1024){ err(1,"File too large (max 25 MB)."); return; }
  P.file=f;
  const img=document.getElementById("preview");
  img.src=URL.createObjectURL(f); img.style.display="block";
  document.getElementById("n1").disabled=false;
};

function grid(id, key, customBox){
  document.querySelectorAll("#"+id+" button.opt[data-v]").forEach(b=>{
    b.onclick=()=>{
      document.querySelectorAll("#"+id+" button.opt").forEach(x=>x.classList.remove("sel"));
      b.classList.add("sel"); P[key]=b.dataset.v;
      if(customBox) document.getElementById(customBox).style.display = b.dataset.v==="custom"?"block":"none";
    };
  });
}
grid("pageGrid","page","pageCustom");
grid("bgGrid","bg","bgCustom");
grid("psGrid","ps","psCustom");
grid("qtyGrid","qty","qtyCustom");
grid("fmtGrid","fmt");
document.querySelector("#pageGrid button").classList.add("sel");
document.querySelector('#bgGrid button[data-v="skip"]').classList.add("sel");
document.querySelector('#psGrid button[data-v="35x45"]').classList.add("sel");
document.querySelector('#qtyGrid button[data-v="12"]').classList.add("sel");
document.querySelector('#fmtGrid button[data-v="pdf"]').classList.add("sel");
document.querySelectorAll("#pageCustom button.opt[data-u], #psCustom button.opt[data-u]").forEach(b=>{
  b.onclick=()=>{ const box=b.parentElement.parentElement.id;
    document.querySelectorAll("#"+box+" button.opt[data-u]").forEach(x=>x.classList.remove("sel"));
    b.classList.add("sel");
    if(box==="pageCustom") P.pu=b.dataset.u; else P.psu=b.dataset.u; };
});
document.querySelector('#pageCustom button[data-u="mm"]').classList.add("sel");
document.querySelector('#psCustom button[data-u="mm"]').classList.add("sel");

function unitToMm(v,u){ v=parseFloat(v); return u==="cm"?v*10 : u==="inch"?v*25.4 : v; }

function validate(step){
  if(step===2){
    if(P.page==="custom"){
      const w=unitToMm(document.getElementById("pw").value,P.pu),
            h=unitToMm(document.getElementById("ph").value,P.pu);
      if(!(w>=50&&w<=1200&&h>=50&&h<=1200)){ err(2,"Enter valid page dimensions (50–1200 mm)."); return false; }
      P.pw=w; P.ph=h;
    }
    err(2);
  }
  if(step===3){
    if(P.bg==="custom"){
      const hex=document.getElementById("hex").value.trim();
      if(!/^#?[0-9a-fA-F]{6}$/.test(hex)){ err(3,"Invalid HEX. Use like #FFFFFF."); return false; }
      P.hex=hex.startsWith("#")?hex:"#"+hex;
    }
    err(3);
  }
  if(step===4){
    if(P.ps==="custom"){
      const w=unitToMm(document.getElementById("psw").value,P.psu),
            h=unitToMm(document.getElementById("psh").value,P.psu);
      if(!(w>=10&&w<=400&&h>=10&&h<=400)){ err(4,"Enter valid photo dimensions (10–400 mm)."); return false; }
      P.psw=w; P.psh=h;
    }
    err(4);
  }
  if(step===5){
    if(P.qty==="custom"){
      const q=parseInt(document.getElementById("qcustom").value,10);
      if(!(q>=1&&q<=500)){ err(5,"Quantity must be 1–500."); return false; }
      P.qty=q;
    } else P.qty=parseInt(P.qty,10);
    err(5);
  }
  return true;
}

const PAGE_DIMS={A4:"210 × 297 mm",A3:"297 × 420 mm",A5:"148 × 210 mm",Letter:"215.9 × 279.4 mm",Legal:"215.9 × 355.6 mm"};
const PS_DIMS={"25x35":"25 × 35 mm","30x40":"30 × 40 mm","35x45":"35 × 45 mm","2x2":"2 × 2 inch"};
const BG_NAMES={skip:"Skip (original)",blue:"🔵 Blue",red:"🔴 Red",white:"⚪ White",green:"🟢 Green",custom:"Custom HEX"};
const FMT_NAMES={pdf:"PDF",jpg:"JPEG",png:"PNG",pdf_jpg:"PDF + JPEG",pdf_png:"PDF + PNG"};

function renderSummary(){
  const page = P.page==="custom" ? `Custom (${P.pw} × ${P.ph} mm)` : `${P.page} (${PAGE_DIMS[P.page]})`;
  const ps = P.ps==="custom" ? `${P.psw} × ${P.psh} mm` : PS_DIMS[P.ps];
  const bg = P.bg==="custom" ? P.hex : BG_NAMES[P.bg];
  document.getElementById("summary").innerHTML =
    `✅ <b>PHOTO READY</b><br><br>
     📄 Page: <span class="badge">${page}</span><br>
     📐 Photo size: <span class="badge">${ps}</span><br>
     🖼 Quantity: <span class="badge">${P.qty}</span><br>
     🎨 Background: <span class="badge">${bg}</span><br>
     📦 Format: <span class="badge">${FMT_NAMES[P.fmt]}</span><br>
     🖨 Quality: <span class="badge">300 DPI</span>`;
}

async function generate(){
  err(7);
  const spin=document.getElementById("spin"), btn=document.getElementById("genBtn");
  spin.style.display="block"; btn.disabled=true;
  try{
    const fd=new FormData();
    fd.append("photo",P.file);
    fd.append("payload",JSON.stringify({
      page:P.page, page_w:P.pw, page_h:P.ph,
      bg:P.bg, hex:P.hex, ps:P.ps, ps_w:P.psw, ps_h:P.psh,
      qty:P.qty, fmt:P.fmt }));
    const r=await fetch("/api/generate",{method:"POST",body:fd});
    if(!r.ok){
      let msg="Generation failed. Try different settings.";
      try{ msg=(await r.json()).error||msg; }catch(e){}
      throw new Error(msg);
    }
    const blob=await r.blob();
    const cd=r.headers.get("Content-Disposition")||"";
    const m=cd.match(/filename="?([^";]+)"?/);
    const name=m?m[1]:(P.fmt==="pdf"?"passport_photos.pdf":"passport_photos.zip");
    const a=document.createElement("a");
    a.href=URL.createObjectURL(blob); a.download=name; document.body.appendChild(a); a.click(); a.remove();
    spin.style.display="none";
    document.getElementById("summary").innerHTML="🎉 <b>Done!</b> Your file has been downloaded.";
    if(tg) tg.showPopup({title:"Done",message:"Your print sheet was downloaded."});
  }catch(e){
    spin.style.display="none";
    err(7,e.message||"Something went wrong.");
    if(tg) tg.showAlert(e.message||"Something went wrong.");
  }finally{ btn.disabled=false; }
}
</script>
</body>
</html>
"""

# --------------------------------------------------------------------------- #
# Mini App API (aiohttp) — same processing functions as the bot
# --------------------------------------------------------------------------- #

API_PAGE_DIMS = {k: v for k, v in PAGE_SIZES.items()}
API_PS_DIMS = {"25x35": (25.0, 35.0), "30x40": (30.0, 40.0),
               "35x45": (35.0, 45.0), "2x2": (50.8, 50.8)}
API_BG = {k: rgb_to_hex(v[1]) for k, v in BG_PRESETS.items()}


async def http_index(request: web.Request) -> web.Response:
    return web.Response(text=MINIAPP_HTML, content_type="text/html")


async def http_favicon(request: web.Request) -> web.Response:
    return web.Response(status=204)


async def http_info(request: web.Request) -> web.Response:
    """Small public diagnostics endpoint useful for Render and uptime monitors."""
    return web.json_response({
        "service": "passport-photo-maker",
        "version": "2.1",
        "uptime_seconds": round(time.time() - _started_at, 1),
        "telegram_configured": bool(BOT_TOKEN),
        "rembg_available": REMBG_AVAILABLE,
        "generation_concurrency": GENERATION_CONCURRENCY,
        "generations_completed": _generation_count,
    })


async def http_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "passport-photo-maker"})


async def http_generate(request: web.Request) -> web.Response:
    """Server-side validated generation endpoint for the Mini App."""
    try:
        reader = await request.multipart()
        photo_bytes = None
        payload = None
        async for part in reader:
            if part.name == "photo":
                photo_bytes = await part.read(decode=False)
            elif part.name == "payload":
                try:
                    payload = json.loads((await part.read(decode=False)).decode("utf-8"))
                except Exception:
                    payload = None
        if not photo_bytes or not isinstance(payload, dict):
            return web.json_response({"error": "Photo and settings are required."},
                                     status=400)

        # --- validate everything server-side; never trust the client -------
        page_key = str(payload.get("page", ""))
        if page_key in API_PAGE_DIMS:
            page_w, page_h = API_PAGE_DIMS[page_key]
        elif page_key == "custom":
            page_w = parse_float(str(payload.get("page_w", "")))
            page_h = parse_float(str(payload.get("page_h", "")))
            if (page_w is None or page_h is None or
                    not (MM_MIN_PAGE <= page_w <= MM_MAX_PAGE) or
                    not (MM_MIN_PAGE <= page_h <= MM_MAX_PAGE)):
                return web.json_response({"error": "Invalid custom page size."},
                                         status=400)
        else:
            return web.json_response({"error": "Invalid page size."}, status=400)

        bg_key = str(payload.get("bg", "skip"))
        if bg_key == "skip":
            bg_rgb = None
        elif bg_key in API_BG:
            bg_rgb = parse_hex_color(API_BG[bg_key])
        elif bg_key == "custom":
            bg_rgb = parse_hex_color(str(payload.get("hex", "")))
            if bg_rgb is None:
                return web.json_response({"error": "Invalid HEX background color."},
                                         status=400)
        else:
            return web.json_response({"error": "Invalid background option."},
                                     status=400)

        ps_key = str(payload.get("ps", ""))
        if ps_key in API_PS_DIMS:
            ps_w, ps_h = API_PS_DIMS[ps_key]
        elif ps_key == "custom":
            ps_w = parse_float(str(payload.get("ps_w", "")))
            ps_h = parse_float(str(payload.get("ps_h", "")))
            if (ps_w is None or ps_h is None or
                    not (MM_MIN_PHOTO <= ps_w <= MM_MAX_PHOTO) or
                    not (MM_MIN_PHOTO <= ps_h <= MM_MAX_PHOTO)):
                return web.json_response({"error": "Invalid custom photo size."},
                                         status=400)
        else:
            return web.json_response({"error": "Invalid photo size."}, status=400)

        try:
            qty = int(payload.get("qty"))
        except (TypeError, ValueError):
            return web.json_response({"error": "Invalid quantity."}, status=400)
        if not (1 <= qty <= MAX_QTY):
            return web.json_response({"error": "Quantity must be 1–500."},
                                     status=400)

        fmt = str(payload.get("fmt", ""))
        if fmt not in ("pdf", "jpg", "png", "pdf_jpg", "pdf_png"):
            return web.json_response({"error": "Invalid output format."},
                                     status=400)

        try:
            image_bytes = photo_bytes
            load_image(image_bytes).close()  # early validation
        except Exception:
            return web.json_response({"error": "The uploaded file is not a valid image."},
                                     status=400)

        loop = asyncio.get_running_loop()
        try:
            async with _generation_semaphore:
                files, pages = await loop.run_in_executor(
                    None, generate_files, image_bytes, page_w, page_h,
                    bg_rgb, ps_w, ps_h, qty, fmt)
            global _generation_count
            _generation_count += 1
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        if len(files) == 1:
            fname, blob = files[0]
            return web.Response(
                body=blob,
                headers={"Content-Disposition": f'attachment; filename="{fname}"',
                         "Content-Type": "application/octet-stream"})

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, blob in files:
                zf.writestr(fname, blob)
        return web.Response(
            body=buf.getvalue(),
            headers={"Content-Disposition": 'attachment; filename="passport_photos.zip"',
                     "Content-Type": "application/zip"})

    except Exception:
        log.exception("mini app generate failed")
        return web.json_response(
            {"error": "Something went wrong while generating. Please try again."},
            status=500)


async def start_web_server() -> web.AppRunner:
    app = web.Application(client_max_size=MAX_IMAGE_BYTES + 2 * 1024 * 1024)
    app.router.add_get("/", http_index)
    app.router.add_get("/health", http_health)
    app.router.add_get("/healthz", http_health)
    app.router.add_get("/api/info", http_info)
    app.router.add_get("/favicon.ico", http_favicon)
    app.router.add_post("/api/generate", http_generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info("HTTP server listening on port %s", PORT)
    return runner


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

async def main() -> None:
    # Start the HTTP listener first. Render detects readiness by scanning the port;
    # Telegram setup must never prevent /health from becoming available.
    runner = await start_web_server()
    if not BOT_TOKEN:
        log.error("BOT_TOKEN is not set; running in web-only mode. Set BOT_TOKEN to enable Telegram polling.")
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
        return

    bot = Bot(token=BOT_TOKEN,
              default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        log.info("Bot started (rembg available: %s)", REMBG_AVAILABLE)
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await runner.cleanup()
        await bot.session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
