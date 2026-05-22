# vLLM Exploration

A hands-on exploration of vLLM — running a real model server on a GPU instance in the cloud.

---

## What is vLLM

vLLM is a fast, production-grade serving engine for large language models. It is not a model itself — it is the software that sits in front of a model and serves requests to users efficiently.

The problem it solves: when multiple users send requests to an LLM at the same time, a naive server queues them and processes them one by one. Response times degrade quickly under load. vLLM solves this with two core mechanisms.

---

## How It Works

### Continuous Batching

Instead of processing each request separately and waiting for it to finish before starting the next, vLLM groups multiple incoming requests into a batch and processes them together. When one request finishes mid-batch, a new one slots in immediately — the GPU is never idle waiting for a single request to complete.

### PagedAttention

LLMs need GPU RAM to hold the context of each active request. Traditionally each request reserves a large contiguous block of memory upfront, even if it doesn't use it all — wasting VRAM and limiting concurrency.

PagedAttention borrows the virtual memory paging concept from operating systems. It breaks memory into small pages and allocates only what each request actually needs. More requests fit in the same VRAM, and memory is released page by page as requests complete.

The **v** in vLLM stands for **virtual** — as in virtual memory.

---

## vLLM vs Ollama

| | Ollama | vLLM |
|---|---|---|
| Designed for | Developer laptop | Production server |
| Strength | Easy setup, model management | High throughput, concurrency |
| API | OpenAI-compatible | OpenAI-compatible |
| Batching | Basic | Continuous batching |
| Memory management | Standard | PagedAttention |
| Typical use | Prototyping, local dev | Serving users at scale |

Both expose an OpenAI-compatible API — the same curl commands and Python code work with either, just change the base URL.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Lambda Labs Instance                    │
│                  (Ubuntu 22.04 LTS)                     │
│                                                         │
│   ┌─────────────┐     ┌──────────────────────────────┐ │
│   │ NVIDIA GPU  │     │       Docker Container        │ │
│   │    (A10)    │◄────│                              │ │
│   │   24GB VRAM │     │  ┌────────┐   ┌───────────┐ │ │
│   └──────┬──────┘     │  │  vLLM  │──►│    LLM    │ │ │
│          │            │  │ Server │   │ (opt-125m)│ │ │
│   ┌──────┴──────┐     │  └────────┘   └───────────┘ │ │
│   │   NVIDIA    │     │                              │ │
│   │  Container  ├────►│         Port 8000            │ │
│   │  Toolkit    │     └──────────────────────────────┘ │
│   └─────────────┘                                       │
└─────────────────────────────────────────────────────────┘
                              ▲
                              │ HTTP requests
                              │ curl / Python client
                         Your Machine
```

**How the layers connect:**
- **Lambda Labs instance** — the cloud server with physical NVIDIA A10 GPU
- **NVIDIA drivers** — installed on the host OS so the GPU is accessible
- **NVIDIA Container Toolkit** — bridges Docker to the GPU driver; enables `--gpus all` flag
- **Docker container** — isolated environment running the vLLM image
- **vLLM** — the serving engine inside the container; loads the model and handles requests
- **LLM (OPT-125m)** — the actual model loaded into GPU VRAM, doing the inference

---

## Hardware Requirements

- **OS:** Linux (Ubuntu 22.04 LTS recommended)
- **GPU:** NVIDIA GPU with CUDA support (tested on A10 24GB)
- **Software:** Docker, NVIDIA drivers, NVIDIA Container Toolkit

> **Note:** vLLM does not run on Apple Silicon. The official Docker image requires NVIDIA CUDA. For local development on Mac, use Ollama instead.

---

## Setup Guide

### 1. Launch a GPU Server (Lambda Labs)

[Lambda Labs](https://lambdalabs.com) is the recommended provider for hands-on vLLM work. Unlike AWS or GCP, there are no quota approvals required — GPU instances are available instantly.

**Instance configuration:**
- Type: `1x A10 (24GB)`
- Region: us-east-1 or us-west-2 (pick whichever shows available)
- Image: Ubuntu 22.04.5 LTS
- Filesystem: none
- Firewall: allow TCP port 22 (SSH) and TCP port 8000 (vLLM API) on `0.0.0.0/0`

> **Cost:** ~$0.75/hour. Terminate the instance immediately when done.

Generate and add an SSH key before launching:

```bash
ssh-keygen -t ed25519 -C "lambda-labs" -f ~/.ssh/id_ed25519_lambda
cat ~/.ssh/id_ed25519_lambda.pub  # paste this into Lambda Labs dashboard
```

Connect once the instance is running:

```bash
ssh -i ~/.ssh/id_ed25519_lambda ubuntu@<IP_ADDRESS>
```

---

### 2. Verify the GPU

```bash
nvidia-smi
```

If `nvidia-smi` is not found, install the drivers:

```bash
sudo apt update && sudo apt install -y nvidia-utils-550-server docker.io
sudo apt install -y nvidia-driver-580-server
sudo reboot
```

Reconnect after reboot and run `nvidia-smi` again. You should see the A10 listed with 23028MiB of VRAM.

---

### 3. Install NVIDIA Container Toolkit

Docker containers are isolated from host hardware by default. The NVIDIA Container Toolkit bridges Docker to the GPU driver so containers can use the GPU.

```bash
# Add NVIDIA's package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install and configure
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify Docker can see the GPU:

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

You should see the A10 listed inside the container output.

---

### 4. Run vLLM

```bash
sudo docker run -d \
  --name vllm \
  --gpus all \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model facebook/opt-125m
```

**Flag breakdown:**
- `-d` — run in background
- `--name vllm` — name the container for easy reference
- `--gpus all` — pass all GPUs through to the container
- `-p 8000:8000` — expose vLLM's API on port 8000
- `--ipc=host` — share host memory space (required for vLLM's inter-process communication)
- `--model facebook/opt-125m` — model to download from HuggingFace and serve

Check logs until you see `Application startup complete`:

```bash
sudo docker logs vllm
```

---

## Usage

### Single request

```bash
curl http://<IP_ADDRESS>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Machine learning is",
    "max_tokens": 50
  }'
```

> Note: OPT-125m is a pre-chat-era model and does not support the `/v1/chat/completions` endpoint. Use `/v1/completions` with a `prompt` field instead.

### Concurrent requests

```bash
for i in {1..20}; do
  curl -s http://<IP_ADDRESS>:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "facebook/opt-125m",
      "prompt": "Machine learning is",
      "max_tokens": 200
    }' &
done
wait
echo "All done"
```

The `&` fires each request in the background simultaneously. vLLM batches them and processes concurrently.

---

## What I Observed

Running 20 concurrent requests while monitoring `nvidia-smi` (`watch -n 1 nvidia-smi`):

- **GPU memory increased** as vLLM allocated pages for each concurrent request (PagedAttention in action)
- **GPU memory dropped** as requests completed and pages were released
- All 20 requests completed successfully — no queuing, no failures

With a tiny model like OPT-125m on an A10, requests complete in milliseconds. A larger model would make the memory allocation and release more visible.

---

## Stopping the Instance

```bash
# Stop the container
sudo docker stop vllm

# Then terminate the instance in the Lambda Labs dashboard
```

> Always terminate the Lambda Labs instance when done. GPU instances charge by the minute.

---

## Project Structure

```
vllm-exploration/
└── README.md   # this file — setup guide and observations
```
