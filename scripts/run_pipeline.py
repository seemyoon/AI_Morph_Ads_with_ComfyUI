from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.parse
import uuid
from pathlib import Path

import requests
import websocket
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / "workflows" / "ai_morph_ads.json"
CONFIG_PATH = ROOT / "configs" / "variants.yaml"
OUTPUT_DIR = ROOT / "outputs"


class ComfyClient:
    def __init__(self, server: str):
        self.server = server.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, workflow: dict) -> str:
        """POST /prompt and return the prompt_id."""
        r = requests.post(
            f"{self.server}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"ComfyUI rejected workflow: {data}")
        return data["prompt_id"]

    def wait_for(self, prompt_id: str, timeout_s: int = 600) -> dict:
        """Block on the ComfyUI websocket until the given prompt finishes."""
        ws_url = f"ws://{urllib.parse.urlparse(self.server).netloc}/ws?clientId={self.client_id}"
        ws = websocket.create_connection(ws_url, timeout=timeout_s)
        deadline = time.time() + timeout_s
        try:
            while time.time() < deadline:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue
                payload = json.loads(msg)
                if payload.get("type") == "executing":
                    d = payload["data"]
                    if d.get("node") is None and d.get("prompt_id") == prompt_id:
                        break
        finally:
            ws.close()
        return self.history(prompt_id)

    def history(self, prompt_id: str) -> dict:
        r = requests.get(f"{self.server}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        return r.json().get(prompt_id, {})

    def download_image(self, filename: str, subfolder: str, folder_type: str, dest: Path) -> None:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = requests.get(f"{self.server}/view", params=params, stream=True, timeout=120)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)

    def upload_image(self, path: Path) -> str:
        """POST /upload/image so the workflow's LoadImage node can see it. Returns the server-side filename."""
        with path.open("rb") as f:
            r = requests.post(
                f"{self.server}/upload/image",
                files={"image": (path.name, f, "image/png")},
                data={"overwrite": "true"},
                timeout=60,
            )
        r.raise_for_status()
        return r.json()["name"]


def patch_workflow(template: dict, values: dict) -> dict:
    """Replace every '{{key}}' string or numeric-in-string with values[key]."""
    raw = json.dumps(template)
    for k, v in values.items():
        placeholder = "{{" + k + "}}"
        raw = raw.replace(placeholder, str(v).replace('"', '\\"'))
    wf = json.loads(raw)

    wf = {k: v for k, v in wf.items() if not k.startswith("_")}

    wf["3"]["inputs"]["strength_model"] = values["lora_weight"]
    wf["3"]["inputs"]["strength_clip"] = values["lora_weight"]
    wf["7"]["inputs"]["weight"] = values["ipadapter_weight"]
    wf["14"]["inputs"]["strength"] = values["canny_strength"]
    wf["15"]["inputs"]["strength"] = values["depth_strength"]
    wf["16"]["inputs"]["width"] = values["width"]
    wf["16"]["inputs"]["height"] = values["height"]
    wf["17"]["inputs"]["seed"] = values["seed"]
    wf["17"]["inputs"]["steps"] = values["steps"]
    wf["17"]["inputs"]["cfg"] = values["cfg"]
    wf["17"]["inputs"]["sampler_name"] = values["sampler_name"]
    wf["17"]["inputs"]["scheduler"] = values["scheduler"]
    wf["19"]["inputs"]["width"] = values["refine_width"]
    wf["19"]["inputs"]["height"] = values["refine_height"]
    wf["24"]["inputs"]["seed"] = values["seed"] + 1
    wf["24"]["inputs"]["steps"] = values["refine_steps"]
    wf["24"]["inputs"]["cfg"] = values["refine_cfg"]
    wf["24"]["inputs"]["denoise"] = values["refine_denoise"]
    return wf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--product", required=True, type=Path, help="Path to product image (PNG/JPG).")
    p.add_argument("--brand", required=True, help="Brand name (injected into prompt + overlay).")
    p.add_argument("--tagline", required=True, help="Tagline (injected into prompt + overlay).")
    p.add_argument("--product-name", default="product",
                   help="Generic noun for the object (e.g. 'perfume bottle'). Injected into prompt.")
    p.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server URL.")
    p.add_argument("--seed", type=int, default=None, help="Base seed (default: random).")
    p.add_argument("--no-overlay", action="store_true", help="Skip PIL text overlay step.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()

    if not args.product.exists():
        sys.exit(f"Product image not found: {args.product}")

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    client = ComfyClient(args.server)

    print(f"[1/3] Uploading product image -> {args.server}")
    server_filename = client.upload_image(args.product)

    base_seed = args.seed if args.seed is not None else random.randint(0, 2 ** 31 - 1)
    shared = cfg["shared"]
    overlay = shared["overlay"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_outputs: list[tuple[str, Path]] = []

    print(f"[2/3] Running {len(cfg['variants'])} variants (base seed={base_seed})")
    for i, v in enumerate(cfg["variants"], start=1):
        print(f"      - variant {i}/{len(cfg['variants'])}: {v['id']} ({v['label']})")
        prompt_text = v["prompt"].format(brand=args.brand, product=args.product_name).strip()
        values = {
            "sd15_checkpoint": shared["sd15_checkpoint"],
            "sd15_vae": shared["sd15_vae"],
            "sdxl_checkpoint": shared["sdxl_checkpoint"],
            "controlnet_canny": shared["controlnet_canny"],
            "controlnet_depth": shared["controlnet_depth"],
            "ipadapter_model": shared["ipadapter_model"],
            "clip_vision": shared["clip_vision"],
            "lora": v["lora"],
            "lora_weight": float(v["lora_weight"]),
            "ipadapter_weight": float(v.get("ipadapter_weight", shared["ipadapter_weight"])),
            "canny_strength": float(shared["controlnet"]["canny_strength"]),
            "depth_strength": float(shared["controlnet"]["depth_strength"]),
            "product_image": server_filename,
            "variant_id": v["id"],
            "positive_prompt": prompt_text,
            "negative_prompt": shared["negative_prompt"].strip(),
            "seed": base_seed + i,
            "steps": int(shared["sampler"]["steps"]),
            "cfg": float(shared["sampler"]["cfg"]),
            "sampler_name": shared["sampler"]["sampler_name"],
            "scheduler": shared["sampler"]["scheduler"],
            "width": int(shared["sampler"]["width"]),
            "height": int(shared["sampler"]["height"]),
            "refine_steps": int(shared["refine"]["steps"]),
            "refine_cfg": float(shared["refine"]["cfg"]),
            "refine_denoise": float(shared["refine"]["denoise"]),
            "refine_width": int(shared["refine"]["width"]),
            "refine_height": int(shared["refine"]["height"]),
        }
        workflow = patch_workflow(template, values)
        prompt_id = client.queue_prompt(workflow)
        history = client.wait_for(prompt_id)

        outputs = history.get("outputs", {}).get("26", {}).get("images", [])
        if not outputs:
            print(f"      ! variant {v['id']} produced no images", file=sys.stderr)
            continue
        img = outputs[0]
        dest = args.output_dir / f"{v['id']}_raw.png"
        client.download_image(img["filename"], img.get("subfolder", ""), img.get("type", "output"), dest)
        raw_outputs.append((v["id"], dest))
        print(f"        saved -> {dest}")

    if args.no_overlay or not raw_outputs:
        print("[3/3] Skipping text overlay.")
        return

    print(f"[3/3] Applying PIL text overlay on {len(raw_outputs)} images")
    for variant_id, src in raw_outputs:
        final = args.output_dir / f"{variant_id}.png"
        cmd = [
            sys.executable, str(ROOT / "scripts" / "text_overlay.py"),
            "--input", str(src),
            "--output", str(final),
            "--brand", args.brand,
            "--tagline", args.tagline,
            "--position", str(overlay["position"]),
            "--padding", str(overlay["padding"]),
            "--brand-size", str(overlay["brand_size"]),
            "--tagline-size", str(overlay["tagline_size"]),
            "--color", str(overlay["color"]),
        ]
        if overlay.get("font"):
            cmd += ["--font", str(ROOT / "assets" / "fonts" / overlay["font"])]
        if overlay.get("shadow"):
            cmd += ["--shadow"]
        subprocess.run(cmd, check=True)
        print(f"  overlay -> {final}")

    print("Done. Final creatives in:", args.output_dir)


if __name__ == "__main__":
    main()
