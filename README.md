# FlowSat

**FlowSat: A Flow-Matching Diffusion Transformer for Satellite Imagery
Generation with Metadata Conditioning**

FlowSat is a text + metadata-conditioned satellite image generator. It
fine-tunes the Sana-0.6B Diffusion Transformer with a *flow-matching*
objective and a *zero-initialised AdaLN metadata graft* that injects the
seven standard FMoW metadata fields (longitude, latitude, GSD, cloud
cover, year, month, day) into the network without disrupting the
pretrained image prior. The result is a 20-step Euler sampler that
matches or exceeds DiffusionSat's 100-step DDIM baseline on FMoW-RGB
while remaining competitive zero-shot on RSICD.

This repository contains the training code, model definitions, the
flow-matching loss / Euler-Heun samplers, the geometry-aware metadata
encoder, the FMoW dataset loader with rich-caption support, and an
inference pipeline.

---

## Repository layout

```
flowsat/
  flow/
    flow_matching.py        # Linear-interpolant flow-matching loss
    flow_sampler.py         # Euler and Heun ODE samplers
  models/
    sat_sana.py             # SatSana: Sana-0.6B + zero-init AdaLN metadata graft
    sat_clip.py             # SatCLIPMetadataEncoder (spherical lift + Fourier)
  data/
    fmow_dataset.py         # FMoW directory traversal + caption-mix curriculum
    fmow_caption.py         # Caption file loaders (rich / short / empty)
    sat_data_util.py        # Metadata normalisation, image transforms
  training/
    train_flowsat_sana.py   # Multi-GPU training loop (Accelerate)
  inference/
    flowsat_pipeline.py     # End-to-end text + metadata -> image pipeline
```

---

## Method in one paragraph

FlowSat treats text-to-satellite-image generation as flow matching:
sample noise `z ~ N(0, I)` and a clean VAE latent `x0`, interpolate
`x_t = (1-t) x0 + t z`, regress the velocity `v = z - x0`. The
backbone is Sana-0.6B (28 DiT blocks, 1152 hidden, linear self-attention
with QK-norm), with Gemma-2-2B as the frozen text encoder feeding
cross-attention in every block, and a frozen DC-AE-f32c32 VAE
(`32x32x32` latents at 512 px). The seven-field metadata vector enters
through a *zero-initialised* parallel projection `W_md` into the
AdaLN-single modulation, alongside the timestep path: at step 0
`W_md = 0`, so the network output is byte-identical to vanilla Sana and
the model inherits its pretrained image prior immediately. Training is
125K optimiser steps with AdamW, peak LR 1e-5 cosine, effective batch
24, bf16, gradient checkpointing. Inference is a 20-step Euler ODE with
CFG scale 2.5.

The metadata encoder is geometry-aware: longitude and latitude are
lifted to the 2-sphere `(cos phi cos lambda, cos phi sin lambda,
sin phi)` and Fourier-encoded; temporal and sensor fields each get
their own Fourier basis; the features are concatenated (195 + 99 + 33 +
33 = 360) and projected through a 360 -> 512 -> 512 -> 1152 MLP.

---

## Quick start

### 1. Environment

```bash
# Python 3.10 recommended
conda create -n flowsat python=3.10 -y
conda activate flowsat
pip install -r requirements.txt
```

You need an NVIDIA GPU with bf16 support (A100, A5000, H100, RTX 4090,
or similar). Training was performed on 3x A100 80GB.

### 2. Data

The dataset loader expects FMoW-RGB laid out as:

```
fmow_root/
  train/
    <category>/                  # 62 categories, e.g. airport, port, solar_farm, ...
      <category>_<instance>/
        <category>_<instance>_<i>_rgb.tif
        <category>_<instance>_<i>_rgb.json   # metadata
```

Each image carries a JSON sidecar with longitude, latitude, GSD,
cloud cover, and timestamp. The loader normalises these to `[0, 1000]`
following DiffusionSat's convention.

Rich captions (one `.txt` per image, ~100-200 words, VLM-generated)
should be placed in a parallel tree:

```
caption_dir/
  train/
    <category>/
      <category>_<instance>/
        <category>_<instance>_<i>_rgb.txt
```

Short metadata-derived captions ("A `<category>` in `<city>`,
`<country>` in `<season>`") are produced on the fly by the dataset
loader.

### 3. Pretrained components

The first training run will download:
- `Efficient-Large-Model/Sana_600M_512px_diffusers` (Sana backbone,
  Gemma-2-2B-IT tokenizer + text encoder, DC-AE VAE)

Pre-cache it offline:

```bash
huggingface-cli download Efficient-Large-Model/Sana_600M_512px_diffusers \
    --local-dir ./pretrained/sana-600m-512px-diffusers
```

### 4. Training

Single-machine, multi-GPU via Accelerate:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_HUB_OFFLINE=1   # if you pre-cached the pretrained checkpoint

accelerate launch \
    --num_processes=3 --mixed_precision=bf16 \
    flowsat/training/train_flowsat_sana.py \
    --pixart_id ./pretrained/sana-600m-512px-diffusers \
    --pretrained_model_name_or_path ./pretrained/sana-600m-512px-diffusers \
    --fmow_root /path/to/fmow-rgb/train \
    --caption_dir /path/to/fmow_captions/train \
    --output_dir ./flowsat_outputs/run1 \
    --resolution 512 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler cosine \
    --lr_warmup_steps 1000 \
    --max_train_steps 125000 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 5 \
    --validation_steps 2500 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --use_satclip_encoder \
    --dataloader_num_workers 4 \
    --max_grad_norm 1.0 \
    --caption_drop_prob 0.0 \
    --seed 42
```

Important flags:

- `--use_satclip_encoder`: use the spherical-lift + Fourier metadata
  encoder described in the paper. **Always enable this for the reported
  configuration.** The default per-field encoder exists only for
  backward compatibility with DiffusionSat's encoding style.
- `--caption_drop_prob 0.0`: the dataset's caption mix
  (70% rich, 20% short, 10% empty) already provides the unconditional
  signal for CFG.
- `--gradient_checkpointing`: required for batch size 8 on 80 GB GPUs;
  disable for 1-shot speed if memory allows.

Resuming from a checkpoint:

```bash
accelerate launch ... train_flowsat_sana.py \
    ... \
    --resume_from_checkpoint latest
```

### 5. Inference

Generate one image from a caption and metadata vector:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoencoderDC

from flowsat.models.sat_sana import SatSana
from flowsat.models.sat_clip import SatCLIPMetadataEncoder
from flowsat.flow.flow_sampler import EulerSampler

device = "cuda"
dtype = torch.bfloat16

sana_dir = "./pretrained/sana-600m-512px-diffusers"
ckpt_dir = "./flowsat_outputs/run1/checkpoint-125000"

# Frozen components
tokenizer = AutoTokenizer.from_pretrained(sana_dir, subfolder="tokenizer")
tokenizer.model_max_length = 256
text_encoder = AutoModelForCausalLM.from_pretrained(
    sana_dir, subfolder="text_encoder", torch_dtype=dtype
).to(device).eval()
vae = AutoencoderDC.from_pretrained(
    sana_dir, subfolder="vae", torch_dtype=dtype
).to(device).eval()

# SatSana + metadata encoder
model = SatSana(
    latent_size=16, in_channels=32, num_metadata=7,
    use_metadata=True, cross_attention_dim=2304,
    pretrained_sana_id=sana_dir, load_pretrained=True,
)
satclip = SatCLIPMetadataEncoder(embed_dim=1152, num_metadata=7)
model.set_metadata_encoder(satclip)

# Load EMA weights (preferred) or model_0.pt
import pathlib
ema_path = pathlib.Path(ckpt_dir) / "ema.pt"
ema = torch.load(ema_path, map_location="cpu", weights_only=False)
with torch.no_grad():
    for p, s in zip(model.parameters(), ema["shadow_params"]):
        p.data.copy_(s)
model = model.to(device, dtype=dtype).eval()

# Caption + metadata
caption = "A satellite image of a commercial port along a coastline."
metadata = torch.tensor([[
    (-122.32 + 180) / 360 * 1000,   # lon
    ( 37.74 +  90) / 180 * 1000,    # lat
    0.5 * 1000,                     # gsd (m)
    0.0,                            # cloud %
    (2018 - 1980) / 120 * 1000,     # year
    6 / 12 * 1000,                  # month
    15 / 31 * 1000,                 # day
]], device=device, dtype=dtype)

with torch.no_grad():
    tok = tokenizer([caption], padding="max_length",
                    max_length=256, truncation=True, return_tensors="pt").to(device)
    cond = text_encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask,
                        output_hidden_states=True).hidden_states[-1]
    tok_u = tokenizer([""], padding="max_length",
                       max_length=256, truncation=True, return_tensors="pt").to(device)
    uncond = text_encoder(input_ids=tok_u.input_ids, attention_mask=tok_u.attention_mask,
                          output_hidden_states=True).hidden_states[-1]

    emb_cfg  = torch.cat([uncond, cond], dim=0)
    mask_cfg = torch.cat([tok_u.attention_mask, tok.attention_mask], dim=0)
    md_cfg   = torch.cat([torch.zeros_like(metadata), metadata], dim=0)

    z = torch.randn(1, 32, 16, 16, device=device, dtype=dtype,
                    generator=torch.Generator(device).manual_seed(42))

    def cfg(x, t, **kw):
        out = model(torch.cat([x, x], 0), torch.cat([t, t], 0),
                    encoder_hidden_states=emb_cfg, metadata=md_cfg,
                    attention_mask=mask_cfg).sample
        u, c = out.chunk(2, dim=0)
        return u + 2.5 * (c - u)

    latents = EulerSampler(num_steps=20).sample(
        cfg, z, num_steps=20, show_progress=False,
        encoder_hidden_states=emb_cfg, metadata=md_cfg, attention_mask=mask_cfg,
    )
    img = (vae.decode(latents / vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)

from PIL import Image
import numpy as np
arr = (img[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
Image.fromarray(arr).save("flowsat_sample.png")
```

A higher-level wrapper is available in
`flowsat/inference/flowsat_pipeline.py`.

---

## Citing

If you use this code, please cite:

```bibtex
@inproceedings{flowsat2026,
  title     = {FlowSat: A Flow-Matching Diffusion Transformer for Satellite
               Imagery Generation with Metadata Conditioning},
  year      = {2026}
}
```

We build on Sana, DC-AE, Gemma-2, Flow Matching, SatCLIP, and
DiffusionSat; please cite the corresponding papers.

---

## License

Code released under the MIT License. The pretrained Sana-0.6B
checkpoint, DC-AE VAE, and Gemma-2 text encoder are governed by their
respective upstream licenses. FMoW-RGB is released under its own
data-use agreement.
