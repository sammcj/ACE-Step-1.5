# ACE-Step 1.5 Docker Deployment

This guide covers running ACE-Step 1.5 using Docker.

## Prerequisites

- Docker Engine 24.0+
- Docker Compose v2.0+
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with CUDA 12.x support
- At least 4GB VRAM (16GB+ recommended)

### Installing NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick Start

### Option 1: Gradio Web UI (Default)

```bash
# Build and start the Gradio web interface
docker compose up --build

# Or run in background
docker compose up -d --build
```

Access the web UI at: http://localhost:7860

### Option 2: REST API Server Only

```bash
# Using profile
docker compose --profile api up acestep-api --build

# Or directly
docker compose up acestep-api --build
```

API available at: http://localhost:8001

### Option 3: Both Services (Single Container)

```bash
docker compose --profile full up acestep-full --build
```

- Web UI: http://localhost:7860
- API: http://localhost:8001

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Model Configuration
ACESTEP_CONFIG_PATH=acestep-v15-turbo          # DiT model
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B      # LM model
ACESTEP_DEVICE=auto                            # Device: auto, cuda, cpu
ACESTEP_LM_BACKEND=vllm                        # LM backend: vllm, pt

# Performance
ACESTEP_OFFLOAD_CPU=false                      # Enable CPU offloading for low VRAM
ACESTEP_INIT_SERVICE=false                     # Pre-initialize models on startup
ACESTEP_SKIP_MODEL_DOWNLOAD=false              # Skip auto-download (for pre-mounted models)
ACESTEP_MULTI_GPU=                             # Multi-GPU: auto (default), true, or false

# UI Settings
ACESTEP_UI_LANGUAGE=en                         # Language: en, zh, ja

# API Settings
ACESTEP_API_LOG_LEVEL=info                     # Log level: debug, info, warning, error

# Authentication (optional)
ACESTEP_API_KEY=your-secret-key               # API authentication key
ACESTEP_AUTH_USERNAME=admin                    # Gradio UI username
ACESTEP_AUTH_PASSWORD=password                 # Gradio UI password

# Port Configuration (host ports)
GRADIO_HOST_PORT=7860                          # Host port for Gradio UI
API_HOST_PORT=8001                             # Host port for REST API
```

### Available Models

**DiT Models:**
- `acestep-v15-turbo` (default, included in main download)
- `acestep-v15-base`
- `acestep-v15-sft`
- `acestep-v15-turbo-shift1`
- `acestep-v15-turbo-shift3`
- `acestep-v15-turbo-continuous`

**LM Models:**
- `acestep-5Hz-lm-0.6B` (smallest, good for low VRAM)
- `acestep-5Hz-lm-1.7B` (default, included in main download)
- `acestep-5Hz-lm-4B` (largest, best quality)

### GPU Memory Requirements

| VRAM | Capabilities |
|------|-------------|
| 4GB  | DiT only, max 3 min, CPU offload required |
| 6GB  | DiT only, max 6 min |
| 8GB  | LM 0.6B option, max 4 min with LM |
| 12GB | LM 0.6B/1.7B, max 4-6 min |
| 16GB | All models, max 8 min |
| 24GB+| Full performance, max 10 min, batch 8 |

For systems with <16GB VRAM, enable CPU offloading:

```bash
ACESTEP_OFFLOAD_CPU=true docker compose up
```

### Multi-GPU Systems

ACE-Step currently uses a **single GPU** for inference. Multi-GPU model splitting (`device_map="auto"`) is experimental and disabled by default due to device synchronization issues in the codebase.

**For systems with multiple GPUs**, you can:

```bash
# Use a specific GPU (e.g., the one with more free VRAM)
CUDA_VISIBLE_DEVICES=0 docker compose up
CUDA_VISIBLE_DEVICES=1 docker compose up

# Enable experimental multi-GPU (may cause errors)
ACESTEP_MULTI_GPU=true docker compose up
```

**Memory optimization alternatives:**
```bash
# Enable CPU offloading (recommended for memory issues)
ACESTEP_OFFLOAD_CPU=true docker compose up

# Use smaller LM model (saves ~2-4GB VRAM)
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B docker compose up

# Combine both
ACESTEP_OFFLOAD_CPU=true ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B docker compose up
```

## Volume Management

### Default Named Volumes

The compose file uses named volumes for persistence:

```bash
# List volumes
docker volume ls | grep acestep

# Inspect volume location
docker volume inspect acestep-checkpoints

# Remove volumes (WARNING: deletes downloaded models)
docker volume rm acestep-checkpoints acestep-cache
```

### Using Local Directories

To use local directories instead of named volumes, modify `docker-compose.yml`:

```yaml
volumes:
  - ./checkpoints:/app/checkpoints
  - ./cache:/app/.cache
```

Or override via command line:

```bash
docker compose run -v $(pwd)/checkpoints:/app/checkpoints acestep-ui
```

### Pre-downloading Models

Download models before starting services:

```bash
# Download main model (includes default DiT, LM 1.7B, VAE, text encoder)
docker compose run --rm acestep-ui download

# Or download all available models
docker compose run --rm acestep-ui python -m acestep.model_downloader --all

# Download specific model
docker compose run --rm acestep-ui python -m acestep.model_downloader --model acestep-5Hz-lm-4B
```

## Building

### Build Only

```bash
docker compose build
```

### Build with Custom Tag

```bash
docker build -t acestep:custom .
```

### Build Notes

- The build process compiles flash-attention which can be slow (10-30 minutes)
- Use `MAX_JOBS=4` is set to prevent OOM during compilation
- The image is ~15-20GB due to CUDA and ML dependencies

## Commands Reference

| Command | Description |
|---------|-------------|
| `gradio` | Start Gradio web UI (default) |
| `api` | Start REST API server |
| `both` | Start both services in one container |
| `download` | Download all models |
| `shell` | Open bash shell in container |

Examples:

```bash
# Open shell for debugging
docker compose run --rm acestep-ui shell

# Run with different model
docker compose run -e ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B acestep-ui

# Run arbitrary command
docker compose run --rm acestep-ui python -c "import torch; print(torch.cuda.is_available())"
```

## API Usage

### Health Check

```bash
curl http://localhost:8001/health
```

### List Models

```bash
curl http://localhost:8001/v1/models
```

### Generate Music

```bash
curl -X POST http://localhost:8001/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cheerful pop song with piano and drums",
    "duration": 60,
    "task_type": "text2music"
  }'
```

### Query Result

```bash
curl -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d '{"task_ids": ["your-task-id"]}'
```

### With Authentication

```bash
curl -X POST http://localhost:8001/release_task \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"prompt": "...", "duration": 60}'
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# Check Docker GPU support
docker info | grep -i gpu
```

### Out of Memory (OOM)

1. Enable CPU offloading:
   ```bash
   ACESTEP_OFFLOAD_CPU=true docker compose up
   ```

2. Use smaller LM model:
   ```bash
   ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B docker compose up
   ```

3. Reduce batch size in the Gradio UI settings

### Model Download Fails

```bash
# Check network connectivity
docker compose run --rm acestep-ui curl -I https://huggingface.co

# Retry download
docker compose run --rm acestep-ui download

# Skip auto-download and mount pre-downloaded models
ACESTEP_SKIP_MODEL_DOWNLOAD=true docker compose up
```

### Container Logs

```bash
# View all logs
docker compose logs -f

# View specific service
docker compose logs -f acestep-ui

# View last 100 lines
docker compose logs --tail 100 acestep-ui
```

### Shared Memory Issues

If you see errors about shared memory, the compose file already sets `shm_size: 8gb`. For larger workloads:

```bash
docker compose run --shm-size=16g acestep-ui
```

## Production Deployment

### Security

1. **Use authentication:**
   ```bash
   ACESTEP_API_KEY=your-secure-key-here
   ACESTEP_AUTH_USERNAME=admin
   ACESTEP_AUTH_PASSWORD=secure-password-here
   ```

2. **Use a reverse proxy** (nginx/traefik) for TLS termination

3. **Run as non-root** (already configured in the Dockerfile)

### Resource Limits

Add to docker-compose.yml:

```yaml
services:
  acestep-ui:
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 16G
```

### Monitoring

The containers expose health check endpoints:
- Gradio: `http://localhost:7860/`
- API: `http://localhost:8001/health`

### Scaling

The API server runs with `workers=1` due to its in-memory queue design. For horizontal scaling, run multiple containers behind a load balancer with session affinity.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Docker Container          │
                    │                                     │
┌─────────┐        │  ┌─────────────────────────────┐   │
│  User   │◄──────►│  │     Gradio Web UI (:7860)   │   │
└─────────┘  HTTP  │  └─────────────────────────────┘   │
                    │                                     │
┌─────────┐        │  ┌─────────────────────────────┐   │
│  App    │◄──────►│  │     REST API (:8001)        │   │
└─────────┘  HTTP  │  └─────────────────────────────┘   │
                    │                                     │
                    │  ┌─────────────────────────────┐   │
                    │  │     ACE-Step Core           │   │
                    │  │  ┌───────┐  ┌───────────┐   │   │
                    │  │  │  DiT  │  │  LM (5Hz) │   │   │
                    │  │  └───────┘  └───────────┘   │   │
                    │  └─────────────────────────────┘   │
                    │                                     │
                    │         Volumes:                    │
                    │  /app/checkpoints (models)          │
                    │  /app/.cache (cache, outputs)       │
                    └─────────────────────────────────────┘
```
