FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ARG IMAGE_TAG=dev
RUN echo "$IMAGE_TAG" > /IMAGE_TAG

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget curl ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch (CUDA 12.4) ───────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ── RunPod + inference libs ───────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    runpod requests \
    sageattention triton \
    gguf

# ── ComfyUI core ──────────────────────────────────────────────────────────────
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
RUN cd /comfyui && pip3 install --no-cache-dir -r requirements.txt

# ── Custom nodes ──────────────────────────────────────────────────────────────
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    git clone https://github.com/rgthree/rgthree-comfy && \
    git clone https://github.com/ashtar1984/comfyui-find-perfect-resolution && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation && \
    git clone https://github.com/willchil/comfyui-lora-manager

RUN cd /comfyui/custom_nodes/ComfyUI-KJNodes && \
    pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /comfyui/custom_nodes/comfyui-lora-manager && \
    pip3 install --no-cache-dir -r requirements.txt || true
RUN pip3 install --no-cache-dir imageio imageio-ffmpeg

# ── Workspace dirs ────────────────────────────────────────────────────────────
RUN mkdir -p /workspace/workflows /comfyui/input /comfyui/output

# ── Bake in default workflow + handler ───────────────────────────────────────
# The workflow can be overridden at runtime from /runpod-volume/workflows/
COPY workflow_api.json /workflow_api.json
COPY handler.py /handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
