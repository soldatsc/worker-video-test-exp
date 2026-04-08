"""
RunPod handler for Stefan Falkok SVI 2 PRO workflow (WAN 2.2 I2V A14B).

All workflow parameters are configurable via API — no Docker rebuild needed.
For workflow overrides, place JSON files on the network volume at:
  /runpod-volume/workflows/<name>.json

Video stitching: set num_chunks > 1 to generate multiple clips sequentially,
each using the last frame of the previous clip as the start image. Clips are
concatenated via ffmpeg. Max ~60s at 16fps = 12 chunks × 81 frames.
"""
import base64
import glob
import json
import os
import random
import subprocess
import tempfile
import time
import uuid

import requests
import runpod

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_INPUT_DIR = "/comfyui/input"
COMFYUI_OUTPUT_DIR = "/comfyui/output"
VOLUME_WORKFLOWS_DIR = "/runpod-volume/workflows"
BAKED_WORKFLOW_PATH = "/workflow_api.json"

# ---------- SVI 2 PRO workflow node IDs ----------
NODE_LOAD_IMAGE = "17"
NODE_RESIZE = "48"
NODE_POSITIVE = "77"
NODE_NEGATIVE = "81"
NODE_SAMPLER_HIGH = "84"          # KSamplerAdvanced (high noise)
NODE_FRAMES = "95"                 # INTConstant — length in frames
NODE_CFG_HIGH = "69"               # PrimitiveFloat — cfg high noise
NODE_CFG_LOW = "72"                # PrimitiveFloat — cfg low noise
NODE_SAMPLER_CONFIG = "371"        # KSampler Config (rgthree) — steps + sampler
NODE_UNET_HIGH = "377"             # UNETLoader — HIGH checkpoint
NODE_UNET_LOW = "378"              # UNETLoader — LOW checkpoint
NODE_LORA_HIGH = "385"             # Lora Loader (LoraManager) — HIGH path
NODE_LORA_LOW = "79"               # Lora Loader (LoraManager) — LOW path
NODE_SVI_LORA_HIGH = "365"         # LoraLoaderModelOnly — SVI PRO HIGH strength
NODE_SVI_LORA_LOW = "368"          # LoraLoaderModelOnly — SVI PRO LOW strength
NODE_VIDEO_COMBINE = "102"         # VHS_VideoCombine


# ---------- Helpers ----------

def wait_for_comfyui(timeout: int = 120) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def free_comfy_memory() -> None:
    try:
        requests.post(
            f"{COMFYUI_URL}/free",
            json={"unload_models": False, "free_memory": True},
            timeout=10,
        )
    except Exception:
        pass


def save_input_image(image_base64: str) -> str:
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(COMFYUI_INPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(image_base64))
    return filename


def save_image_from_path(video_path: str) -> str:
    """Extract last frame from a video file and save as PNG input image."""
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(COMFYUI_INPUT_DIR, filename)
    subprocess.run(
        ["ffmpeg", "-sseof", "-0.5", "-i", video_path, "-frames:v", "1",
         "-update", "1", "-y", filepath],
        check=True, capture_output=True,
    )
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        raise RuntimeError(f"ffmpeg produced empty frame from {video_path}")
    return filename


def queue_prompt(workflow: dict) -> str:
    r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    r.raise_for_status()
    return r.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 900) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            if r.status_code == 200:
                history = r.json()
                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("status_str") == "success":
                        return history[prompt_id]
                    if status.get("status_str") == "error":
                        msgs = status.get("messages", [])
                        raise RuntimeError(f"ComfyUI error: {msgs}")
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"Generation timed out after {timeout}s")


def get_video_path(history: dict) -> str:
    for node_output in history.get("outputs", {}).values():
        for key in ("gifs", "videos"):
            for item in node_output.get(key, []):
                fullpath = item.get("fullpath")
                if fullpath and os.path.exists(fullpath):
                    return fullpath
                subfolder = item.get("subfolder", "")
                path = os.path.join(COMFYUI_OUTPUT_DIR, subfolder, item["filename"])
                if os.path.exists(path):
                    return path
    # fallback: newest mp4 (warning: non-deterministic if multiple files present)
    print("WARNING: No video path in history — falling back to glob newest mp4")
    videos = glob.glob(os.path.join(COMFYUI_OUTPUT_DIR, "**/*.mp4"), recursive=True)
    # exclude stitched outputs so we don't accidentally pick a prior stitch
    videos = [v for v in videos if "stitched_" not in os.path.basename(v)]
    if videos:
        return max(videos, key=os.path.getmtime)
    raise FileNotFoundError("No video output found in history or output directory")


def stitch_videos(video_paths: list[str], fps: int) -> str:
    """Concatenate multiple mp4 clips into one using ffmpeg concat demuxer.
    Uses lossless stream copy when possible (same codec/resolution assumed).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
        concat_list = f.name

    output_path = os.path.join(COMFYUI_OUTPUT_DIR, f"stitched_{uuid.uuid4().hex}.mp4")
    try:
        # Try lossless copy first (no re-encode, no fps drift)
        subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list,
             "-c", "copy", "-y", output_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Fallback: re-encode if codecs/resolutions differ between chunks
        subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list,
             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             "-vsync", "vfr", "-y", output_path],
            check=True, capture_output=True,
        )
    os.unlink(concat_list)
    return output_path


def build_lora_manager_value(loras: list[dict]) -> dict:
    """
    Convert API lora list to comfyui-lora-manager __value__ format.
    Input:  [{"name": "new2026/Instareal_high", "strength": 0.90, "clip_strength": 0.90}]
    Output: {"__value__": [...]}
    """
    items = []
    for lora in loras:
        strength_str = str(lora.get("strength", 1.0))
        clip_str = str(lora.get("clip_strength", lora.get("strength", 1.0)))
        items.append({
            "name": lora["name"],
            "strength": strength_str,
            "active": lora.get("active", True),
            "expanded": False,
            "clipStrength": clip_str,
            "locked": False,
        })
    return {"__value__": items}


def load_workflow(workflow_name: str | None) -> dict:
    if workflow_name:
        path = os.path.join(VOLUME_WORKFLOWS_DIR, f"{workflow_name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Workflow '{workflow_name}.json' not found at {VOLUME_WORKFLOWS_DIR}/"
            )
    else:
        path = BAKED_WORKFLOW_PATH
    with open(path, "r") as f:
        return json.load(f)


def patch_workflow(workflow: dict, image_filename: str, params: dict) -> dict:
    """Apply all configurable parameters to workflow dict. Returns modified workflow."""
    wf = workflow  # mutate in place (loaded fresh each job)

    # Input image
    wf[NODE_LOAD_IMAGE]["inputs"]["image"] = image_filename

    # Resize / resolution
    if "width" in params:
        wf[NODE_RESIZE]["inputs"]["width"] = int(params["width"])
    if "height" in params:
        wf[NODE_RESIZE]["inputs"]["height"] = int(params["height"])

    # Prompts
    if "prompt" in params:
        wf[NODE_POSITIVE]["inputs"]["text"] = params["prompt"]
    if "negative_prompt" in params:
        wf[NODE_NEGATIVE]["inputs"]["text"] = params["negative_prompt"]

    # Seed
    seed = params.get("seed", -1)
    if seed == -1:
        seed = random.randint(0, 2**50)
    wf[NODE_SAMPLER_HIGH]["inputs"]["noise_seed"] = seed
    params["_resolved_seed"] = seed

    # Frames
    if "frames" in params:
        wf[NODE_FRAMES]["inputs"]["value"] = int(params["frames"])

    # CFG values
    if "cfg_high" in params:
        wf[NODE_CFG_HIGH]["inputs"]["value"] = float(params["cfg_high"])
    if "cfg_low" in params:
        wf[NODE_CFG_LOW]["inputs"]["value"] = float(params["cfg_low"])

    # Steps
    if "steps" in params:
        wf[NODE_SAMPLER_CONFIG]["inputs"]["steps_total"] = int(params["steps"])

    # Sampler / scheduler
    if "sampler_name" in params:
        wf[NODE_SAMPLER_CONFIG]["inputs"]["sampler_name"] = params["sampler_name"]
    if "scheduler" in params:
        wf[NODE_SAMPLER_CONFIG]["inputs"]["scheduler"] = params["scheduler"]

    # Video FPS (both output nodes)
    if "fps" in params:
        wf[NODE_VIDEO_COMBINE]["inputs"]["frame_rate"] = int(params["fps"])
        if "357" in wf:
            wf["357"]["inputs"]["frame_rate"] = int(params["fps"])

    # Checkpoints
    if "checkpoint_high" in params:
        wf[NODE_UNET_HIGH]["inputs"]["unet_name"] = params["checkpoint_high"]
    if "checkpoint_low" in params:
        wf[NODE_UNET_LOW]["inputs"]["unet_name"] = params["checkpoint_low"]

    # SVI PRO LoRA strengths
    if "svi_strength_high" in params:
        wf[NODE_SVI_LORA_HIGH]["inputs"]["strength_model"] = float(params["svi_strength_high"])
    if "svi_strength_low" in params:
        wf[NODE_SVI_LORA_LOW]["inputs"]["strength_model"] = float(params["svi_strength_low"])

    # Action LoRAs (comfyui-lora-manager format)
    if "loras_high" in params and params["loras_high"]:
        wf[NODE_LORA_HIGH]["inputs"]["loras"] = build_lora_manager_value(params["loras_high"])
        # Rebuild text field to match lora names (for trigger word detection)
        wf[NODE_LORA_HIGH]["inputs"]["text"] = ", ".join(
            f"<lora:{l['name']}:{l.get('strength', 1.0)}>"
            for l in params["loras_high"]
        )
    if "loras_low" in params and params["loras_low"]:
        wf[NODE_LORA_LOW]["inputs"]["loras"] = build_lora_manager_value(params["loras_low"])
        wf[NODE_LORA_LOW]["inputs"]["text"] = ", ".join(
            f"<lora:{l['name']}:{l.get('strength', 1.0)}>"
            for l in params["loras_low"]
        )

    return wf


def clear_outputs() -> None:
    """Remove all mp4s from output dir (recursive) except stitched outputs."""
    for f in glob.glob(os.path.join(COMFYUI_OUTPUT_DIR, "**/*.mp4"), recursive=True):
        if "stitched_" not in os.path.basename(f):
            try:
                os.remove(f)
            except OSError:
                pass


# ---------- Main handler ----------

def handler(job: dict) -> dict:
    inp = job["input"]

    image_base64 = inp.get("image")
    if not image_base64:
        return {"error": "Missing 'image' (base64 PNG)"}

    if not wait_for_comfyui(timeout=30):
        return {"error": "ComfyUI not ready after 30s"}

    free_comfy_memory()

    # Stitching params
    num_chunks = int(inp.get("num_chunks", 1))
    fps = int(inp.get("fps", 16))
    workflow_name = inp.get("workflow_name")  # load from volume if set

    # Build params dict (all optional — defaults come from baked workflow)
    params = {}
    for key in (
        "prompt", "negative_prompt", "seed",
        "width", "height",
        "steps", "cfg_high", "cfg_low",
        "frames",
        "sampler_name", "scheduler",
        "checkpoint_high", "checkpoint_low",
        "svi_strength_high", "svi_strength_low",
        "loras_high", "loras_low",
    ):
        if key in inp:
            params[key] = inp[key]

    params["fps"] = fps

    # Single chunk or multi-chunk stitching
    clip_paths: list[str] = []
    current_image_b64 = image_base64
    base_seed = params.get("seed", -1)
    # Resolve base seed once up-front so all chunks are deterministic
    if base_seed == -1:
        base_seed = random.randint(0, 2**50)
    resolved_seed = base_seed

    try:
        for chunk_idx in range(num_chunks):
            clear_outputs()

            workflow = load_workflow(workflow_name)
            image_filename = save_input_image(current_image_b64)

            # Each chunk gets a deterministic seed offset from the base seed
            params["seed"] = (base_seed + chunk_idx) % (2**50)

            patch_workflow(workflow, image_filename, params)

            prompt_id = queue_prompt(workflow)
            history = wait_for_completion(prompt_id, timeout=900)
            video_path = get_video_path(history)
            clip_paths.append(video_path)

            # Extract last frame for next chunk input
            if chunk_idx < num_chunks - 1:
                last_frame_filename = save_image_from_path(video_path)
                # Load last frame back as base64 for next iteration
                frame_path = os.path.join(COMFYUI_INPUT_DIR, last_frame_filename)
                with open(frame_path, "rb") as f:
                    current_image_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Stitch if multiple chunks
        if len(clip_paths) > 1:
            final_video_path = stitch_videos(clip_paths, fps)
        else:
            final_video_path = clip_paths[0]

        with open(final_video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "video": video_b64,
            "format": "mp4",
            "seed": resolved_seed,
            "num_chunks": num_chunks,
            "prompt": params.get("prompt"),
            "frames_per_chunk": params.get("frames", 81),
            "fps": fps,
            "checkpoint_high": params.get("checkpoint_high"),
            "checkpoint_low": params.get("checkpoint_low"),
            "workflow_name": workflow_name,
        }

    except FileNotFoundError as e:
        return {"error": str(e)}
    except TimeoutError as e:
        return {"error": str(e)}
    except subprocess.CalledProcessError as e:
        return {"error": f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


runpod.serverless.start({"handler": handler})
