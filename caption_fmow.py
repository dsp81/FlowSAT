"""
caption_fmow.py — Detailed VLM captioning for the FMoW dataset.

Generates rich, satellite-domain captions for every image in the FMoW
training corpus and writes them to a NEW directory tree that mirrors the
original layout. The original dataset is never touched.

Output layout
-------------
    <output_dir>/
        train/
            airport/
                airport_0/
                    airport_0_0_rgb.txt        ← caption (UTF-8 text)
                    airport_0_0_rgb.meta.json  ← prompt + model + timing
                    ...
        manifest.shard{S}.jsonl                ← per-shard manifest
        progress.shard{S}.log                  ← progress log
        FAILED.shard{S}.txt                    ← list of failed paths

Resume
------
    Re-running the same command skips any image that already has a
    non-empty caption file. The script is safe to interrupt with Ctrl-C
    at any time — every caption is written atomically (tmp+rename).

Multi-GPU (no NCCL)
-------------------
    Launch one process per GPU and shard the work deterministically:

       

    Sharding is by sample index (deterministic sort), so processes never
    touch the same file. All four can run concurrently and resume
    independently.

Models supported
----------------
    qwen2.5-vl-7b   (default)  — best quality, ~14 GB bf16
    qwen2.5-vl-3b              — faster, ~6 GB bf16, fits 16 GB GPUs
    qwen2-vl-7b                — fallback
    florence2-large            — much faster but shorter captions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("caption_fmow")


# ---------------------------------------------------------------------------
# FMoW constants (kept local so this script is self-contained)
# ---------------------------------------------------------------------------

FMOW_CATEGORIES = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field",
    "dam", "debris_or_rubble", "educational_institution",
    "electric_substation", "factory_or_powerplant", "fire_station",
    "flooded_road", "fountain", "gas_station", "golf_course",
    "ground_transportation_station", "helipad", "hospital",
    "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential",
    "nuclear_powerplant", "office_building", "oil_or_gas_facility",
    "park", "parking_lot_or_garage", "place_of_worship",
    "police_station", "port", "prison", "race_track",
    "railway_bridge", "recreational_facility", "road_bridge",
    "runway", "shipyard", "shopping_mall",
    "single-unit_residential", "smokestack", "solar_farm",
    "space_facility", "stadium", "storage_tank",
    "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal",
    "water_treatment_facility", "wind_farm", "zoo",
]

COUNTRY_CODE_MAP = {
    "US": "United States", "GB": "United Kingdom", "FR": "France",
    "DE": "Germany", "CN": "China", "JP": "Japan", "IN": "India",
    "BR": "Brazil", "RU": "Russia", "AU": "Australia", "CA": "Canada",
    "IT": "Italy", "ES": "Spain", "KR": "South Korea", "MX": "Mexico",
    "ID": "Indonesia", "TR": "Turkey", "SA": "Saudi Arabia",
    "ZA": "South Africa", "AR": "Argentina", "EG": "Egypt",
    "NG": "Nigeria", "PK": "Pakistan", "TH": "Thailand",
    "NL": "Netherlands", "CH": "Switzerland", "SE": "Sweden",
    "PL": "Poland", "BE": "Belgium", "AT": "Austria",
    "NO": "Norway", "DK": "Denmark", "FI": "Finland",
    "IE": "Ireland", "PT": "Portugal", "GR": "Greece",
    "CZ": "Czech Republic", "RO": "Romania", "HU": "Hungary",
    "NZ": "New Zealand", "SG": "Singapore", "MY": "Malaysia",
    "PH": "Philippines", "VN": "Vietnam", "CL": "Chile",
    "CO": "Colombia", "PE": "Peru", "UA": "Ukraine",
    "IL": "Israel", "AE": "United Arab Emirates",
}


# ---------------------------------------------------------------------------
# Image I/O — matches flowsat/data/fmow_dataset.py::_read_image
# ---------------------------------------------------------------------------

def _percentile_normalization(
    img: np.ndarray, lo: float = 2.0, hi: float = 98.0, axis=(0, 1)
) -> np.ndarray:
    img = img.astype(np.float32)
    p_lo = np.percentile(img, lo, axis=axis, keepdims=True)
    p_hi = np.percentile(img, hi, axis=axis, keepdims=True)
    denom = np.maximum(p_hi - p_lo, 1e-6)
    return np.clip((img - p_lo) / denom, 0.0, 1.0)


def read_image_robust(path: str) -> Optional[np.ndarray]:
    """Read an FMoW image (TIF/JPG/PNG) and return (H, W, 3) uint8.

    Returns None on unrecoverable failure (corrupted file).
    """
    # First try rasterio (handles GeoTIFF + 16-bit + multispectral)
    try:
        import rasterio  # noqa
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
            if img.shape[0] >= 3:
                img = img[:3]
            elif img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)

            if img.dtype == np.uint16:
                img = (_percentile_normalization(img) * 255).astype(np.uint8)
            elif img.dtype in (np.float32, np.float64):
                if img.max() > 1.0:
                    img = _percentile_normalization(img)
                else:
                    img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = (_percentile_normalization(img) * 255).astype(np.uint8)
            return img
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"rasterio failed for {path}: {e}; falling back to PIL")

    # Fallback to PIL (handles standard JPG/PNG, sometimes regular TIFs)
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im)
    except Exception as e:
        logger.warning(f"Failed to read image {path}: {e}")
        return None


def read_metadata_robust(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read metadata {path}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Sample scanning (mirrors FMoWDataset._scan_dataset)
# ---------------------------------------------------------------------------

@dataclass
class SampleRecord:
    image_path: str
    metadata_path: str
    category: str
    location_dir: str
    stem: str  # filename without extension

    @property
    def relative_dir(self) -> str:
        # e.g. "airport/airport_0"
        return f"{self.category}/{Path(self.location_dir).name}"


def scan_fmow_dataset(
    fmow_root: Path,
    categories: Optional[List[str]] = None,
) -> List[SampleRecord]:
    """Walk root/<category>/<location>/*_rgb.{tif,jpg,png} pairs."""
    valid = set(categories) if categories else set(FMOW_CATEGORIES)
    samples: List[SampleRecord] = []

    if not fmow_root.exists():
        raise FileNotFoundError(f"FMoW root not found: {fmow_root}")

    cat_dirs = sorted(p for p in fmow_root.iterdir() if p.is_dir())
    for cat_dir in cat_dirs:
        if cat_dir.name not in valid:
            continue
        loc_dirs = sorted(p for p in cat_dir.iterdir() if p.is_dir())
        for loc_dir in loc_dirs:
            imgs: List[Path] = []
            for ext in ("*_rgb.tif", "*_rgb.jpg", "*_rgb.png"):
                imgs.extend(sorted(loc_dir.glob(ext)))
            for img_path in imgs:
                json_path = img_path.with_suffix(".json")
                if not json_path.exists():
                    continue
                samples.append(SampleRecord(
                    image_path=str(img_path),
                    metadata_path=str(json_path),
                    category=cat_dir.name,
                    location_dir=str(loc_dir),
                    stem=img_path.stem,  # e.g. "airport_0_0_rgb"
                ))

    # Deterministic sort so sharding is stable across runs
    samples.sort(key=lambda s: s.image_path)
    return samples


def output_paths_for(out_root: Path, sample: SampleRecord) -> Tuple[Path, Path]:
    """Return (caption_path, meta_path) inside the output mirror tree."""
    rel_dir = out_root / "train" / sample.category / Path(sample.location_dir).name
    cap = rel_dir / f"{sample.stem}.txt"
    meta = rel_dir / f"{sample.stem}.meta.json"
    return cap, meta


def caption_already_exists(cap_path: Path) -> bool:
    try:
        return cap_path.exists() and cap_path.stat().st_size > 0
    except OSError:
        return False


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_context_block(category: str, metadata: Dict[str, Any]) -> str:
    """Compact context line shown to the VLM as ground truth conditioning."""
    cls_name = " ".join(category.split("_"))
    cc = metadata.get("country_code", "") or ""
    country = COUNTRY_CODE_MAP.get(cc, cc) if cc else ""

    gsd = metadata.get("gsd", None)
    cloud = metadata.get("cloud_cover", None)
    ts = metadata.get("timestamp", "") or ""
    date = ts[:10] if isinstance(ts, str) and len(ts) >= 10 else ""

    parts = [f"Category label: {cls_name}"]
    if country:
        parts.append(f"Country: {country}")
    if date:
        parts.append(f"Date: {date}")
    if isinstance(gsd, (int, float)):
        parts.append(f"Ground sample distance: {gsd:.2f} m/pixel")
    if isinstance(cloud, (int, float)):
        parts.append(f"Cloud cover: {cloud:.1f}%")
    return " | ".join(parts)


CAPTION_INSTRUCTION = (
    "You are an expert remote-sensing analyst writing training captions for a "
    "satellite image generation model. Look carefully at the image and write "
    "ONE long, dense, factual paragraph of about 120 to 200 words.\n\n"
    "Cover, in this order:\n"
    "1. The dominant land cover and surface materials you actually see "
    "(paved areas, rooftops, vegetation type, bare soil, water, sand, snow).\n"
    "2. The principal man-made structures, their shapes, geometry, and how "
    "they are laid out (e.g. straight runways, curved roads, gridded blocks, "
    "circular tanks, rectangular buildings, parking lots, piers, fields).\n"
    "3. Spatial composition: where things sit relative to each other, "
    "approximate density, and any axes or symmetries.\n"
    "4. Colors, textures, and any notable patterns (striping, mottling, "
    "shadows, haze, glare, seasonal cues).\n"
    "5. The surrounding context (urban, suburban, rural, coastal, forested, "
    "arid, mountainous, etc.).\n\n"
    "Strict rules:\n"
    "- Use the provided context as ground truth. Do not contradict the "
    "category label.\n"
    "- Describe only what is visible. Do NOT invent specific names of "
    "places, people, companies, models of vehicles, or aircraft.\n"
    "- Do NOT begin with phrases like 'This image shows', 'The image is', "
    "'In this satellite image', or any meta-commentary. Start directly with "
    "the description.\n"
    "- Write a single flowing paragraph. No bullet points. No headings. "
    "No markdown.\n"
    "- Be concrete and specific. Avoid filler like 'various features' or "
    "'a number of things'."
)


def build_user_prompt(category: str, metadata: Dict[str, Any]) -> str:
    return f"{CAPTION_INSTRUCTION}\n\nContext: {build_context_block(category, metadata)}"


# ---------------------------------------------------------------------------
# Dataset for parallel I/O
# ---------------------------------------------------------------------------

# Maximum side fed to the VLM. 448 is a sweet spot for Qwen2.5-VL: large
# enough to preserve fine satellite detail, small enough to keep token
# count manageable.
VLM_INPUT_SIZE = 448


class FMoWCaptionDataset(Dataset):
    """Loads images + metadata from disk and packages them for the VLM."""

    def __init__(self, samples: List[SampleRecord], input_size: int = VLM_INPUT_SIZE):
        self.samples = samples
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        try:
            arr = read_image_robust(s.image_path)
            if arr is None:
                return {"sample": s, "image": None, "metadata": {}, "error": "read_failed"}

            img = Image.fromarray(arr).convert("RGB")
            # Resize: long side = input_size (preserves aspect, then we let
            # the VLM processor handle final cropping/padding).
            w, h = img.size
            if max(w, h) > self.input_size:
                if w >= h:
                    new_w = self.input_size
                    new_h = max(1, int(round(h * self.input_size / w)))
                else:
                    new_h = self.input_size
                    new_w = max(1, int(round(w * self.input_size / h)))
                img = img.resize((new_w, new_h), Image.BILINEAR)

            md = read_metadata_robust(s.metadata_path)
            return {"sample": s, "image": img, "metadata": md, "error": None}
        except Exception as e:
            logger.debug(f"Loader error on {s.image_path}: {e}")
            return {"sample": s, "image": None, "metadata": {}, "error": str(e)}


def caption_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # No tensor stacking — VLM processors batch internally.
    return batch


# ---------------------------------------------------------------------------
# VLM backends
# ---------------------------------------------------------------------------

class BaseCaptioner:
    name: str = "base"
    def caption_batch(
        self, images: List[Image.Image], prompts: List[str], max_new_tokens: int
    ) -> List[str]:
        raise NotImplementedError


class Gemma4Captioner(BaseCaptioner):
    """Gemma 4 multimodal — uses HuggingFace standard image-text-to-text pattern.

    Notes:
      - Gemma models are GATED on HF; you must accept the license and
        `huggingface-cli login` before first use.
      - Thinking mode is intentionally disabled (no `<|think|>` token) so the
        model emits the caption directly instead of an internal reasoning trace.
      - Gemma 4 uses learned 2D position embeddings and handles variable
        aspect ratios natively; no manual normalization needed.
    """

    def __init__(self, model_id: str, device: str, dtype: torch.dtype):
        self.name = model_id
        self.device = device
        self.dtype = dtype
        logger.info(f"Loading {model_id} on {device} ({dtype})...")

        from transformers import AutoProcessor
        # Gemma 4 ships its own classes; fall back to AutoModelForImageTextToText
        # if a specific class isn't directly importable.
        try:
            from transformers import Gemma4ForConditionalGeneration as ModelCls
            logger.info("Using Gemma4ForConditionalGeneration.")
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as ModelCls
                logger.info("Using AutoModelForImageTextToText.")
            except ImportError:
                from transformers import AutoModelForCausalLM as ModelCls
                logger.info("Using AutoModelForCausalLM (last-resort fallback).")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        # Try flash-attn 2 if available — silently skip otherwise.
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("flash_attention_2 enabled.")
        except Exception:
            pass

        self.model = ModelCls.from_pretrained(model_id, **load_kwargs)
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def caption_batch(
        self, images: List[Image.Image], prompts: List[str], max_new_tokens: int
    ) -> List[str]:
        # Build per-sample chat messages. NO system prompt with <|think|> —
        # we want a direct caption, not a reasoning trace.
        messages_batch = []
        for img, prompt in zip(images, prompts):
            messages_batch.append([
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}
            ])

        # Per-item chat-template render, then batch-process.
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_batch
        ]

        inputs = self.processor(
            text=texts,
            images=[[img] for img in images],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Match dtype on floating-point tensors for bf16 inference.
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                inputs[k] = v.to(self.dtype)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            repetition_penalty=1.05,
        )

        # Strip prompt tokens.
        input_lens = inputs["input_ids"].shape[1]
        gen_ids = out_ids[:, input_lens:]
        outputs = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # Sometimes Gemma still emits a leftover thought block even with
        # thinking disabled; strip it defensively.
        cleaned = []
        for o in outputs:
            o = o.strip()
            # Remove anything between <|channel|> ... <|channel|> markers
            for marker in ("<|channel|>", "<channel|>", "<|channel>"):
                while marker in o:
                    start = o.find(marker)
                    end = o.find(marker, start + len(marker))
                    if end == -1:
                        o = o[:start].strip()
                        break
                    o = (o[:start] + o[end + len(marker):]).strip()
            cleaned.append(o)
        return cleaned


def build_captioner(model_choice: str, device: str, dtype: torch.dtype) -> BaseCaptioner:
    aliases = {
        "gemma4-e4b":   "google/gemma-4-E4B-it",
        "gemma4-e2b":   "google/gemma-4-E2B-it",
        "gemma4-26b":   "google/gemma-4-26B-A4B-it",
        "gemma4-31b":   "google/gemma-4-31B-it",
    }
    model_id = aliases.get(model_choice.lower(), model_choice)
    return Gemma4Captioner(model_id, device, dtype)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    fmow_root = Path(args.fmow_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- shard-specific bookkeeping files ----
    manifest_path = out_root / f"manifest.shard{args.shard_id}of{args.num_shards}.jsonl"
    progress_path = out_root / f"progress.shard{args.shard_id}of{args.num_shards}.log"
    failed_path   = out_root / f"FAILED.shard{args.shard_id}of{args.num_shards}.txt"

    # ---- scan dataset ----
    t0 = time.time()
    logger.info(f"Scanning {fmow_root}...")
    cat_subset = args.categories.split(",") if args.categories else None
    if cat_subset:
        cat_subset = [c.strip() for c in cat_subset if c.strip()]
    samples = scan_fmow_dataset(fmow_root, categories=cat_subset)
    logger.info(f"Found {len(samples):,} samples in {time.time() - t0:.1f}s")

    # ---- shard ----
    if args.num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_id]
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(samples):,} samples")

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
        logger.info(f"--limit {args.limit}: trimmed to {len(samples):,} samples")

    # ---- skip already-captioned (resume) ----
    if args.resume:
        before = len(samples)
        samples = [s for s in samples if not caption_already_exists(output_paths_for(out_root, s)[0])]
        logger.info(f"Resume: {before:,} → {len(samples):,} after skipping existing captions")

    if len(samples) == 0:
        logger.info("Nothing to do. Exiting.")
        return

    # ---- model + device ----
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Inference on CPU will be very slow.")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda"
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    captioner = build_captioner(args.model, device=device, dtype=dtype)

    # ---- dataloader ----
    ds = FMoWCaptionDataset(samples, input_size=args.input_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=caption_collate,
        shuffle=False,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # ---- main loop ----
    total = len(samples)
    n_done = 0
    n_fail = 0
    t_start = time.time()
    last_log = t_start

    with open(manifest_path, "a", encoding="utf-8") as manifest_f, \
         open(progress_path, "a", encoding="utf-8") as progress_f, \
         open(failed_path, "a", encoding="utf-8") as failed_f:

        progress_f.write(f"# run started at {time.strftime('%Y-%m-%d %H:%M:%S')} model={captioner.name}\n")
        progress_f.flush()

        for batch in loader:
            # Filter out load failures
            valid_indices = [i for i, item in enumerate(batch) if item["image"] is not None]
            failed_items = [item for item in batch if item["image"] is None]

            for it in failed_items:
                s = it["sample"]
                failed_f.write(f"{s.image_path}\t{it.get('error') or 'unknown'}\n")
                failed_f.flush()
                n_fail += 1

            if not valid_indices:
                n_done += len(batch)
                continue

            valid = [batch[i] for i in valid_indices]
            images = [it["image"] for it in valid]
            prompts = [build_user_prompt(it["sample"].category, it["metadata"]) for it in valid]

            # Inference (with retry-on-OOM by halving)
            try:
                t_inf = time.time()
                captions = captioner.caption_batch(images, prompts, max_new_tokens=args.max_new_tokens)
                inf_time = time.time() - t_inf
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM. Retrying batch one image at a time...")
                torch.cuda.empty_cache()
                captions = []
                inf_time = 0.0
                for img, prompt in zip(images, prompts):
                    try:
                        t_inf = time.time()
                        c = captioner.caption_batch([img], [prompt], max_new_tokens=args.max_new_tokens)[0]
                        inf_time += time.time() - t_inf
                        captions.append(c)
                    except Exception as e:
                        logger.warning(f"Single-item retry failed: {e}")
                        captions.append("")
            except Exception as e:
                logger.warning(f"Batch inference error: {e}\n{traceback.format_exc()}")
                captions = ["" for _ in valid]
                inf_time = 0.0

            # Persist
            for it, cap in zip(valid, captions):
                s: SampleRecord = it["sample"]
                cap_path, meta_path = output_paths_for(out_root, s)

                if not cap or len(cap.strip()) < 10:
                    failed_f.write(f"{s.image_path}\tempty_caption\n")
                    failed_f.flush()
                    n_fail += 1
                    continue

                try:
                    atomic_write_text(cap_path, cap.strip() + "\n")
                    atomic_write_json(meta_path, {
                        "image_path": s.image_path,
                        "category": s.category,
                        "model": captioner.name,
                        "input_size": args.input_size,
                        "max_new_tokens": args.max_new_tokens,
                        "context": build_context_block(s.category, it["metadata"]),
                        "wall_time_s": round(inf_time / max(len(valid), 1), 3),
                    })
                    manifest_f.write(json.dumps({
                        "image_path": s.image_path,
                        "caption_path": str(cap_path),
                        "category": s.category,
                        "caption": cap.strip(),
                    }, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.warning(f"Write failed for {cap_path}: {e}")
                    n_fail += 1

            n_done += len(batch)
            manifest_f.flush()

            # Periodic progress log
            now = time.time()
            if now - last_log > args.log_every_seconds:
                elapsed = now - t_start
                rate = n_done / max(elapsed, 1e-6)
                remaining = total - n_done
                eta_s = remaining / max(rate, 1e-6)
                msg = (
                    f"[shard {args.shard_id}/{args.num_shards}] "
                    f"{n_done:,}/{total:,} done "
                    f"({100.0 * n_done / total:.1f}%) "
                    f"| {rate:.2f} img/s "
                    f"| failed={n_fail} "
                    f"| elapsed={elapsed/3600:.2f}h "
                    f"| eta={eta_s/3600:.2f}h"
                )
                logger.info(msg)
                progress_f.write(msg + "\n")
                progress_f.flush()
                last_log = now

        # final summary line
        elapsed = time.time() - t_start
        summary = (
            f"[shard {args.shard_id}/{args.num_shards}] DONE "
            f"total={total:,} captioned={n_done - n_fail:,} failed={n_fail:,} "
            f"elapsed={elapsed/3600:.2f}h"
        )
        logger.info(summary)
        progress_f.write(summary + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate detailed VLM captions for the FMoW dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--fmow_root", type=str, required=True,
                   help="FMoW split root, e.g. /mnt/nas/rishabh_sat/fmow-full/train")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output root for captions (DO NOT point at the dataset).")
    # Model
    p.add_argument("--model", type=str, default="gemma4-e4b",
                   help="Model alias or full HF ID. "
                        "Aliases: gemma4-e4b, gemma4-e2b, gemma4-26b, gemma4-31b.")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--max_new_tokens", type=int, default=320,
                   help="Generation budget. Gemma 4 produces ~150 words at ~300 tokens.")
    p.add_argument("--input_size", type=int, default=VLM_INPUT_SIZE,
                   help="Long-side resize before VLM. 448 = good detail/speed tradeoff.")
    # Run shape
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--categories", type=str, default="",
                   help="Optional comma-separated subset, e.g. 'airport,stadium,port'.")
    p.add_argument("--limit", type=int, default=0,
                   help="Optional cap on samples (for smoke tests). 0 = no limit.")
    # Resume / logging
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip images that already have a non-empty caption.")
    p.add_argument("--log_every_seconds", type=float, default=60.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Refuse to write into the dataset folder by accident.
    fmow_root_resolved = Path(args.fmow_root).resolve()
    out_root_resolved = Path(args.output_dir).resolve()
    if out_root_resolved == fmow_root_resolved or fmow_root_resolved in out_root_resolved.parents:
        logger.error(
            f"--output_dir ({out_root_resolved}) lives inside or equals "
            f"--fmow_root ({fmow_root_resolved}). Refusing to write inside "
            f"the source dataset. Choose a different --output_dir."
        )
        sys.exit(2)

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        logger.error(f"Invalid shard_id {args.shard_id} for num_shards {args.num_shards}")
        sys.exit(2)

    logger.info(
        f"Args: model={args.model} dtype={args.dtype} "
        f"batch={args.batch_size} workers={args.num_workers} "
        f"shard={args.shard_id}/{args.num_shards} "
        f"input_size={args.input_size} max_new_tokens={args.max_new_tokens}"
    )
    run(args)


if __name__ == "__main__":
    main()
