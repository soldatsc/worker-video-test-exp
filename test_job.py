"""
Local test script — sends a job to the RunPod endpoint (or local handler).

Usage:
    # Against RunPod serverless endpoint
    RUNPOD_API_KEY=xxx ENDPOINT_ID=yyy python3 test_job.py --image test.png

    # Against local handler directly (set LOCAL=1)
    LOCAL=1 python3 test_job.py --image test.png
"""
import argparse
import base64
import json
import os
import sys
import time

import requests


def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_local(payload: dict) -> dict:
    """Import handler directly and run synchronously."""
    sys.path.insert(0, "/")
    import handler as h
    return h.handler({"input": payload})


def run_runpod(payload: dict, endpoint_id: str, api_key: str, timeout: int = 600) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"https://api.runpod.io/v2/{endpoint_id}/run"

    r = requests.post(url, json={"input": payload}, headers=headers)
    r.raise_for_status()
    job_id = r.json()["id"]
    print(f"Job submitted: {job_id}")

    status_url = f"https://api.runpod.io/v2/{endpoint_id}/status/{job_id}"
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(status_url, headers=headers)
        data = r.json()
        status = data.get("status")
        print(f"  status: {status} ({int(time.time() - start)}s)", end="\r")
        if status == "COMPLETED":
            print()
            return data.get("output", {})
        if status == "FAILED":
            print()
            return {"error": data.get("error", "FAILED")}
        time.sleep(5)
    return {"error": "timeout"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", default="Instareal, woman slowly moves, natural skin texture, soft lighting")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-high", type=float, default=None)
    parser.add_argument("--cfg-low", type=float, default=None)
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of clips to stitch (1=single)")
    parser.add_argument("--checkpoint-high", default=None)
    parser.add_argument("--checkpoint-low", default=None)
    parser.add_argument("--svi-strength-high", type=float, default=None)
    parser.add_argument("--svi-strength-low", type=float, default=None)
    parser.add_argument("--loras-high", default=None,
                        help='JSON array: [{"name":"new2026/Instareal_high","strength":0.9}]')
    parser.add_argument("--loras-low", default=None,
                        help='JSON array: [{"name":"new2026/Instareal_low","strength":0.95}]')
    parser.add_argument("--workflow-name", default=None, help="Load workflow from volume (name without .json)")
    parser.add_argument("--output", default="output.mp4")
    args = parser.parse_args()

    payload: dict = {
        "image": load_image_b64(args.image),
        "prompt": args.prompt,
        "seed": args.seed,
        "frames": args.frames,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "num_chunks": args.num_chunks,
    }

    if args.negative_prompt is not None:
        payload["negative_prompt"] = args.negative_prompt
    if args.steps is not None:
        payload["steps"] = args.steps
    if args.cfg_high is not None:
        payload["cfg_high"] = args.cfg_high
    if args.cfg_low is not None:
        payload["cfg_low"] = args.cfg_low
    if args.checkpoint_high is not None:
        payload["checkpoint_high"] = args.checkpoint_high
    if args.checkpoint_low is not None:
        payload["checkpoint_low"] = args.checkpoint_low
    if args.svi_strength_high is not None:
        payload["svi_strength_high"] = args.svi_strength_high
    if args.svi_strength_low is not None:
        payload["svi_strength_low"] = args.svi_strength_low
    if args.loras_high:
        payload["loras_high"] = json.loads(args.loras_high)
    if args.loras_low:
        payload["loras_low"] = json.loads(args.loras_low)
    if args.workflow_name:
        payload["workflow_name"] = args.workflow_name

    print("Payload (no image):", json.dumps({k: v for k, v in payload.items() if k != "image"}, indent=2))

    if os.environ.get("LOCAL") == "1":
        result = run_local(payload)
    else:
        endpoint_id = os.environ.get("ENDPOINT_ID")
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not endpoint_id or not api_key:
            print("Set ENDPOINT_ID and RUNPOD_API_KEY, or LOCAL=1 for local run")
            sys.exit(1)
        result = run_runpod(payload, endpoint_id, api_key)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    video_b64 = result.get("video")
    if not video_b64:
        print("No video in response:", json.dumps({k: v for k, v in result.items() if k != "video"}))
        sys.exit(1)

    with open(args.output, "wb") as f:
        f.write(base64.b64decode(video_b64))

    print(f"\nSaved: {args.output}")
    print(f"Seed: {result.get('seed')}")
    print(f"Chunks: {result.get('num_chunks')}")
    print(f"Frames/chunk: {result.get('frames_per_chunk')}")
    print(f"FPS: {result.get('fps')}")


if __name__ == "__main__":
    main()
