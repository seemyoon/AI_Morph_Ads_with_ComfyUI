from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from shutil import copy2

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub is required. Run:  pip install -r requirements.txt")

HF_DOWNLOADS: list[tuple[str, str, str, str]] = [

    ("runwayml/stable-diffusion-v1-5",
     "v1-5-pruned-emaonly.safetensors",
     "checkpoints",
     "v1-5-pruned-emaonly.safetensors"),
    ("stabilityai/sd-vae-ft-mse-original",
     "vae-ft-mse-840000-ema-pruned.safetensors",
     "vae",
     "vae-ft-mse-840000-ema-pruned.safetensors"),

    ("lllyasviel/ControlNet-v1-1",
     "control_v11p_sd15_canny.pth",
     "controlnet",
     "control_v11p_sd15_canny.pth"),
    ("lllyasviel/ControlNet-v1-1",
     "control_v11f1p_sd15_depth.pth",
     "controlnet",
     "control_v11f1p_sd15_depth.pth"),

    ("h94/IP-Adapter",
     "models/ip-adapter-plus_sd15.safetensors",
     "ipadapter",
     "ip-adapter-plus_sd15.safetensors"),
    ("h94/IP-Adapter",
     "models/image_encoder/model.safetensors",
     "clip_vision",
     "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"),
]

HF_DOWNLOADS_SDXL: list[tuple[str, str, str, str]] = [
    ("stabilityai/stable-diffusion-xl-base-1.0",
     "sd_xl_base_1.0.safetensors",
     "checkpoints",
     "sd_xl_base_1.0.safetensors"),
]

HF_DOWNLOADS_LORAS: list[tuple[str, str, str, str]] = [
    ("mnemic/dAIversityLoRA15-PhotoSemiReal-SD1.5-LoRA",
     "dAIversityLoRA15-PhotoSemiReal.safetensors",
     "loras",
     "epiCRealism_Luxury.safetensors"),

    ("mnemic/CyberpunkWorld-SD1.5-LoRA",
     "CyberpunkWorld.safetensors",
     "loras",
     "neon_street_night.safetensors"),

    ("shawn323/sd-v1.5-lora-vintage",
     "pytorch_lora_weights.safetensors",
     "loras",
     "film_kodak_portra_v3.safetensors"),
]

LORA_ALIAS_COPIES: list[tuple[str, str]] = [
    ("epiCRealism_Luxury.safetensors", "minimalist_studio_v2.safetensors"),
]


def get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def download_one(
    repo_id: str,
    filename: str,
    target_dir: Path,
    final_name: str,
    hf_token: str | None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / final_name
    if final_path.exists():
        print(f"  [skip] {final_path.name} already present")
        return
    print(f"  [get]  {repo_id} :: {filename}")
    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
        token=hf_token,
    )
    local_path = Path(local)

    if local_path.name != final_name or local_path.parent != target_dir:
        final_path.unlink(missing_ok=True)
        local_path.rename(final_path)
    print(f"  [ok]   {final_path}")


def ensure_lora_aliases(loras_dir: Path) -> None:
    loras_dir.mkdir(parents=True, exist_ok=True)
    for source_name, alias_name in LORA_ALIAS_COPIES:
        source_path = loras_dir / source_name
        alias_path = loras_dir / alias_name
        if alias_path.exists():
            print(f"  [skip] {alias_path.name} already present")
            continue
        if not source_path.exists():
            print(f"  [warn] missing source for alias copy: {source_path}")
            continue
        copy2(source_path, alias_path)
        print(f"  [ok]   {alias_path} (copied from {source_path.name})")


def main() -> None:
    p = argparse.ArgumentParser(description="Download all models for AI Morph Ads.")
    p.add_argument("--comfy-root", required=True, type=Path,
                   help="Path to your ComfyUI installation (the folder containing main.py).")
    p.add_argument("--skip-sdxl", action="store_true",
                   help="Skip SDXL base download (~6.5 GB). Useful if you only want the SD1.5 stage.")
    p.add_argument("--skip-loras", action="store_true",
                   help="Skip auto-downloading style LoRAs from HuggingFace. Use when you plan to "
                        "provide your own LoRA files manually under ComfyUI/models/loras/.")
    args = p.parse_args()

    comfy_root: Path = args.comfy_root.expanduser().resolve()
    if not comfy_root.exists():
        sys.exit(f"ComfyUI root not found: {comfy_root}")

    models_root = comfy_root / "models"
    hf_token = get_hf_token()
    print(f"Target: {models_root}\n")
    if hf_token:
        print("[auth] HF token found in env (HF_TOKEN or HUGGINGFACE_HUB_TOKEN)")
    else:
        print("[auth] No HF token found. Downloads will run unauthenticated (lower rate limits).")

    todo = list(HF_DOWNLOADS)
    if not args.skip_sdxl:
        todo += HF_DOWNLOADS_SDXL
    if not args.skip_loras:
        todo += HF_DOWNLOADS_LORAS

    for repo_id, filename, subdir, final_name in todo:
        print(f"--> {subdir}/{final_name}")
        download_one(repo_id, filename, models_root / subdir, final_name, hf_token)

    if args.skip_loras:
        print("\n" + "=" * 72)
        print("LoRAs skipped (--skip-loras). Place your own files into:")
        print(f"  {models_root / 'loras'}")
        print("And make sure the filenames match 'lora:' fields in configs/variants.yaml.")
    else:
        print("\n" + "=" * 72)
        print("Creating LoRA aliases...")
        print("=" * 72)
        ensure_lora_aliases(models_root / "loras")

    print("\nDone.")


if __name__ == "__main__":
    main()
