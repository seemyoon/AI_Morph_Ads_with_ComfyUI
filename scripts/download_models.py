from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

LORA_FILENAMES: list[str] = [
    "epiCRealism_Luxury.safetensors",
    "minimalist_studio_v2.safetensors",
    "neon_street_night.safetensors",
    "film_kodak_portra_v3.safetensors",
]


def download_one(repo_id: str, filename: str, target_dir: Path, final_name: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / final_name
    if final_path.exists():
        print(f"  [skip] {final_path.name} already present")
        return
    print(f"  [get]  {repo_id} :: {filename}")
    local = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target_dir))
    local_path = Path(local)

    if local_path.name != final_name or local_path.parent != target_dir:
        final_path.unlink(missing_ok=True)
        local_path.rename(final_path)
    print(f"  [ok]   {final_path}")


def print_lora_instructions(loras_dir: Path) -> None:
    print("\n" + "=" * 72)
    print("LoRA files — manual download required (CivitAI)")
    print("=" * 72)
    print(f"Place the following 4 files into:  {loras_dir}\n")
    for fn in LORA_FILENAMES:
        print(f"  - {fn}")
    print("\nSuggested sources (browse + pick any style LoRA that matches each mood):")
    print("  Luxury   : https://civitai.com/search/models?sortBy=models_v9&query=luxury")
    print("  Minimal  : https://civitai.com/search/models?sortBy=models_v9&query=minimalist+studio")
    print("  Urban    : https://civitai.com/search/models?sortBy=models_v9&query=neon+night")
    print("  Nostalgic: https://civitai.com/search/models?sortBy=models_v9&query=kodak+portra+film")
    print("\nIf the LoRA filenames you download differ from the above, edit")
    print("configs/variants.yaml to match — no other code changes needed.")


def main() -> None:
    p = argparse.ArgumentParser(description="Download all models for AI Morph Ads.")
    p.add_argument("--comfy-root", required=True, type=Path,
                   help="Path to your ComfyUI installation (the folder containing main.py).")
    p.add_argument("--skip-sdxl", action="store_true",
                   help="Skip SDXL base download (~6.5 GB). Useful if you only want the SD1.5 stage.")
    args = p.parse_args()

    comfy_root: Path = args.comfy_root.expanduser().resolve()
    if not comfy_root.exists():
        sys.exit(f"ComfyUI root not found: {comfy_root}")

    models_root = comfy_root / "models"
    print(f"Target: {models_root}\n")

    todo = list(HF_DOWNLOADS)
    if not args.skip_sdxl:
        todo += HF_DOWNLOADS_SDXL

    for repo_id, filename, subdir, final_name in todo:
        print(f"--> {subdir}/{final_name}")
        download_one(repo_id, filename, models_root / subdir, final_name)

    print_lora_instructions(models_root / "loras")
    print("\nDone.")


if __name__ == "__main__":
    main()
