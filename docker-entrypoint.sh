#!/bin/bash
set -e

# =============================================================================
# ACE-Step 1.5 Docker Entrypoint
# Uses uv run as per official README instructions
# =============================================================================

# Colors for output (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

log_info() {
    echo -e "${GREEN}[entrypoint]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[entrypoint]${NC} $1"
}

log_error() {
    echo -e "${RED}[entrypoint]${NC} $1"
}

# =============================================================================
# Model Download Functions
# =============================================================================

# Check if main model components exist
check_main_model() {
    local checkpoints_dir="/app/checkpoints"
    local required_components=("acestep-v15-turbo" "vae" "Qwen3-Embedding-0.6B" "acestep-5Hz-lm-1.7B")

    for component in "${required_components[@]}"; do
        if [ ! -d "${checkpoints_dir}/${component}" ]; then
            return 1
        fi
    done
    return 0
}

# Check if a specific submodel exists
check_submodel() {
    local model_name="$1"
    local checkpoints_dir="/app/checkpoints"
    [ -d "${checkpoints_dir}/${model_name}" ]
}

# Download models if not present
download_models_if_needed() {
    if [ "${ACESTEP_SKIP_MODEL_DOWNLOAD:-false}" = "true" ]; then
        log_warn "Skipping model download (ACESTEP_SKIP_MODEL_DOWNLOAD=true)"
        return 0
    fi

    local need_download=false

    # Check if main model exists (contains core components)
    if ! check_main_model; then
        log_info "Main model components not found, will download..."
        need_download=true
    fi

    # Download main model if needed (using uv run as per README)
    if [ "$need_download" = true ]; then
        log_info "Downloading main model (includes vae, text encoder, DiT, and LM 1.7B)..."
        if ! uv run acestep-download; then
            log_error "Failed to download main model"
            return 1
        fi
        log_info "Main model download complete"
    fi

    # Check for non-default LM model if specified
    local lm_model="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
    if [ "$lm_model" != "acestep-5Hz-lm-1.7B" ] && ! check_submodel "$lm_model"; then
        log_info "Downloading LM model: $lm_model"
        if ! uv run acestep-download --model "$lm_model" --skip-main; then
            log_warn "Failed to download LM model $lm_model, falling back to default"
            export ACESTEP_LM_MODEL_PATH="acestep-5Hz-lm-1.7B"
        fi
    fi

    # Check for non-default DiT model if specified
    local dit_model="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
    if [ "$dit_model" != "acestep-v15-turbo" ] && ! check_submodel "$dit_model"; then
        log_info "Downloading DiT model: $dit_model"
        if ! uv run acestep-download --model "$dit_model" --skip-main; then
            log_warn "Failed to download DiT model $dit_model, falling back to default"
            export ACESTEP_CONFIG_PATH="acestep-v15-turbo"
        fi
    fi

    return 0
}

# =============================================================================
# Service Start Functions
# =============================================================================

# Build Gradio arguments from environment variables
build_gradio_args() {
    local args="--server-name ${GRADIO_SERVER_NAME:-0.0.0.0} --port ${GRADIO_SERVER_PORT:-7860}"

    [ -n "$ACESTEP_CONFIG_PATH" ] && args="$args --config_path $ACESTEP_CONFIG_PATH"
    [ -n "$ACESTEP_LM_MODEL_PATH" ] && args="$args --lm_model_path $ACESTEP_LM_MODEL_PATH"
    [ "${ACESTEP_INIT_SERVICE:-false}" = "true" ] && args="$args --init_service"
    [ "${ACESTEP_OFFLOAD_CPU:-false}" = "true" ] && args="$args --offload_to_cpu"
    [ "${ACESTEP_ENABLE_API:-true}" = "true" ] && args="$args --enable-api"
    [ -n "$ACESTEP_API_KEY" ] && args="$args --api-key $ACESTEP_API_KEY"
    [ -n "$ACESTEP_UI_LANGUAGE" ] && args="$args --language $ACESTEP_UI_LANGUAGE"

    if [ -n "$ACESTEP_AUTH_USERNAME" ] && [ -n "$ACESTEP_AUTH_PASSWORD" ]; then
        args="$args --auth-username $ACESTEP_AUTH_USERNAME --auth-password $ACESTEP_AUTH_PASSWORD"
    fi

    echo "$args"
}

# Start Gradio UI (using uv run acestep as per README)
start_gradio() {
    log_info "Starting Gradio UI on port ${GRADIO_SERVER_PORT:-7860}..."

    local gradio_args
    gradio_args=$(build_gradio_args)

    # shellcheck disable=SC2086
    exec uv run acestep $gradio_args
}

# Start API server (using uv run acestep-api as per README)
start_api() {
    log_info "Starting API server on port ${ACESTEP_API_PORT:-8001}..."

    exec uv run acestep-api \
        --host "${ACESTEP_API_HOST:-0.0.0.0}" \
        --port "${ACESTEP_API_PORT:-8001}"
}

# Start both services (API in background, Gradio in foreground)
start_both() {
    log_info "Starting both Gradio UI and API server..."

    # Track child PIDs for cleanup
    local api_pid=""

    # Cleanup function
    cleanup() {
        log_info "Shutting down services..."
        if [ -n "$api_pid" ] && kill -0 "$api_pid" 2>/dev/null; then
            kill -TERM "$api_pid" 2>/dev/null || true
            wait "$api_pid" 2>/dev/null || true
        fi
        exit 0
    }

    # Set up signal handlers
    trap cleanup SIGTERM SIGINT SIGQUIT

    # Start API server in background
    uv run acestep-api \
        --host "${ACESTEP_API_HOST:-0.0.0.0}" \
        --port "${ACESTEP_API_PORT:-8001}" &
    api_pid=$!

    log_info "API server started with PID $api_pid"

    # Give API server time to start
    sleep 2

    # Check if API server is still running
    if ! kill -0 "$api_pid" 2>/dev/null; then
        log_error "API server failed to start"
        exit 1
    fi

    # Build Gradio arguments (without --enable-api since we have standalone API)
    local gradio_args="--server-name ${GRADIO_SERVER_NAME:-0.0.0.0} --port ${GRADIO_SERVER_PORT:-7860}"
    [ -n "$ACESTEP_CONFIG_PATH" ] && gradio_args="$gradio_args --config_path $ACESTEP_CONFIG_PATH"
    [ -n "$ACESTEP_LM_MODEL_PATH" ] && gradio_args="$gradio_args --lm_model_path $ACESTEP_LM_MODEL_PATH"
    [ "${ACESTEP_INIT_SERVICE:-false}" = "true" ] && gradio_args="$gradio_args --init_service"
    [ "${ACESTEP_OFFLOAD_CPU:-false}" = "true" ] && gradio_args="$gradio_args --offload_to_cpu"
    [ -n "$ACESTEP_UI_LANGUAGE" ] && gradio_args="$gradio_args --language $ACESTEP_UI_LANGUAGE"

    if [ -n "$ACESTEP_AUTH_USERNAME" ] && [ -n "$ACESTEP_AUTH_PASSWORD" ]; then
        gradio_args="$gradio_args --auth-username $ACESTEP_AUTH_USERNAME --auth-password $ACESTEP_AUTH_PASSWORD"
    fi

    log_info "Starting Gradio UI on port ${GRADIO_SERVER_PORT:-7860}..."

    # Start Gradio in foreground (but don't exec so we keep the trap active)
    # shellcheck disable=SC2086
    uv run acestep $gradio_args &
    gradio_pid=$!

    # Wait for either process to exit
    wait -n "$api_pid" "$gradio_pid" 2>/dev/null || true

    # If we get here, one process exited - trigger cleanup
    cleanup
}

# =============================================================================
# Main entrypoint logic
# =============================================================================

echo "============================================="
echo "  ACE-Step 1.5 - Music Generation Framework"
echo "============================================="
echo "  DiT Model:  ${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
echo "  LM Model:   ${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
echo "  Device:     ${ACESTEP_DEVICE:-auto}"
echo "============================================="
echo ""

# Download models if needed
if ! download_models_if_needed; then
    log_error "Model download failed. Please check your network connection or mount pre-downloaded models."
    exit 1
fi

# Parse command
case "${1:-gradio}" in
    gradio|ui|web)
        start_gradio
        ;;
    api|server)
        start_api
        ;;
    both|all)
        start_both
        ;;
    download)
        log_info "Downloading all models..."
        uv run acestep-download --all
        log_info "Model download complete."
        ;;
    shell|bash)
        exec /bin/bash
        ;;
    *)
        # Pass through any other command
        exec "$@"
        ;;
esac
