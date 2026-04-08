#!/bin/bash
set -e

VOLUME="/runpod-volume"
MODELS="/comfyui/models"

echo "=== SVI 2 PRO Video Worker ==="
echo "Image: $(cat /IMAGE_TAG 2>/dev/null || echo 'dev')"

# ── Link models from Network Volume ──────────────────────────────────────────
if [ -d "$VOLUME/ComfyUI/models" ]; then
    echo "Linking models from Network Volume..."
    for dir in unet clip vae loras upscale_models checkpoints; do
        if [ -d "$VOLUME/ComfyUI/models/$dir" ]; then
            rm -rf "$MODELS/$dir"
            ln -sf "$VOLUME/ComfyUI/models/$dir" "$MODELS/$dir"
            echo "  linked: $dir"
        fi
    done
else
    echo "WARNING: No models at $VOLUME/ComfyUI/models/"
    ls -la "$VOLUME/" 2>/dev/null || echo "Volume not mounted!"
fi

# ── Link custom nodes from volume (e.g. comfyui-lora-manager) ────────────────
VOLUME_NODES="$VOLUME/ComfyUI/custom_nodes"
if [ -d "$VOLUME_NODES" ]; then
    for node_dir in "$VOLUME_NODES"/*/; do
        node_name=$(basename "$node_dir")
        target="/comfyui/custom_nodes/$node_name"
        if [ ! -e "$target" ]; then
            ln -sf "$node_dir" "$target"
            echo "  linked custom node: $node_name"
        fi
    done
fi

# ── Workflow overrides from volume (swap workflow without rebuilding) ─────────
# Place any <name>.json in /runpod-volume/workflows/ and pass workflow_name=<name>
if [ -d "$VOLUME/workflows" ]; then
    ln -sfn "$VOLUME/workflows" /workspace/workflows
    echo "Workflow overrides linked from $VOLUME/workflows"
else
    mkdir -p /workspace/workflows
    echo "No workflow overrides at $VOLUME/workflows (using baked-in default)"
fi

# ── Model check ───────────────────────────────────────────────────────────────
echo "=== Model check ==="
ls -lh "$MODELS/unet/wan2.2/"*.safetensors 2>/dev/null | head -4 || echo "WARNING: No wan2.2 UNETs!"
ls -lh "$MODELS/clip/"*.safetensors 2>/dev/null | head -2 || echo "WARNING: No CLIP!"
ls -lh "$MODELS/vae/"*.safetensors 2>/dev/null | head -2 || echo "WARNING: No VAE!"

echo "=== LoRA folders on volume ==="
ls "$MODELS/loras/" 2>/dev/null || echo "WARNING: loras dir missing!"
echo "--- new2026/ ---"
ls "$MODELS/loras/new2026/" 2>/dev/null || echo "MISSING: new2026/ not on volume — copy LoRAs from old pod!"
echo "--- SVI PRO ---"
ls "$MODELS/loras/wan2.2/"*SVI* 2>/dev/null || echo "MISSING: SVI PRO LoRAs not on volume!"

# ── Start ComfyUI ─────────────────────────────────────────────────────────────
echo "=== Starting ComfyUI ==="

EXTRA_ARGS=""
if [ "${DISABLE_SAGE_ATTENTION}" != "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-sage-attention"
    echo "SageAttention: ON"
else
    echo "SageAttention: OFF (DISABLE_SAGE_ATTENTION=1)"
fi

cd /comfyui
python3 main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    $EXTRA_ARGS \
    --disable-cuda-malloc \
    --gpu-only \
    &
COMFY_PID=$!

# ── Wait for ComfyUI to be ready ──────────────────────────────────────────────
echo "Waiting for ComfyUI..."
MAX_WAIT=120
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI ready (${i}s)"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "ERROR: ComfyUI did not start in ${MAX_WAIT}s"
        exit 1
    fi
    sleep 1
done

# ── Watchdog: restart ComfyUI if it dies ─────────────────────────────────────
(
    while true; do
        sleep 10
        if ! kill -0 $COMFY_PID 2>/dev/null; then
            echo "WARNING: ComfyUI died (pid $COMFY_PID) — restarting..."
            cd /comfyui
            python3 main.py --listen 127.0.0.1 --port 8188 $EXTRA_ARGS --disable-cuda-malloc --gpu-only &
            COMFY_PID=$!
            sleep 30  # give it time to reload models
        fi
    done
) &

# ── Start RunPod handler ──────────────────────────────────────────────────────
echo "=== Starting RunPod Handler ==="
python3 /handler.py
