"""
Gradio UI Generation Section Module
Contains generation section component definitions
"""
import torch
import gradio as gr
from acestep.constants import (
    VALID_LANGUAGES,
    TRACK_NAMES,
    TASK_TYPES_TURBO,
    TASK_TYPES_BASE,
    DEFAULT_DIT_INSTRUCTION,
)
from acestep.gradio_ui.i18n import t
from acestep.gpu_config import get_global_gpu_config, GPUConfig


def create_generation_section(dit_handler, llm_handler, init_params=None, language='en') -> dict:
    """Create generation section
    
    Args:
        dit_handler: DiT handler instance
        llm_handler: LM handler instance
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
        language: UI language code ('en', 'zh', 'ja')
    """
    # Check if service is pre-initialized
    service_pre_initialized = init_params is not None and init_params.get('pre_initialized', False)
    
    # Check if running in service mode (restricted UI)
    service_mode = init_params is not None and init_params.get('service_mode', False)
    
    # Get current language from init_params if available
    current_language = init_params.get('language', language) if init_params else language
    
    # Get GPU configuration
    gpu_config: GPUConfig = init_params.get('gpu_config') if init_params else None
    if gpu_config is None:
        gpu_config = get_global_gpu_config()
    
    # Determine if LM is initialized (for setting appropriate limits)
    lm_initialized = init_params.get('init_llm', False) if init_params else False
    
    # Calculate UI limits based on GPU config and LM state
    max_duration = gpu_config.max_duration_with_lm if lm_initialized else gpu_config.max_duration_without_lm
    max_batch_size = gpu_config.max_batch_size_with_lm if lm_initialized else gpu_config.max_batch_size_without_lm
    default_batch_size = min(1, max_batch_size)  # Default to 1 for stability
    init_lm_default = gpu_config.init_lm_default
    
    with gr.Group():
        # Service Configuration - collapse if pre-initialized, hide if in service mode
        accordion_open = not service_pre_initialized
        accordion_visible = not service_pre_initialized  # Hide when running in service mode
        with gr.Accordion(t("service.title"), open=accordion_open, visible=accordion_visible) as service_config_accordion:
            # Language selector at the top
            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=[
                        ("English", "en"),
                        ("中文", "zh"),
                        ("日本語", "ja"),
                    ],
                    value=current_language,
                    label=t("service.language_label"),
                    info=t("service.language_info"),
                    scale=1,
                )
            
            # Dropdown options section - all dropdowns grouped together
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    # Set checkpoint value from init_params if pre-initialized
                    checkpoint_value = init_params.get('checkpoint') if service_pre_initialized else None
                    checkpoint_dropdown = gr.Dropdown(
                        label=t("service.checkpoint_label"),
                        choices=dit_handler.get_available_checkpoints(),
                        value=checkpoint_value,
                        info=t("service.checkpoint_info")
                    )
                with gr.Column(scale=1, min_width=90):
                    refresh_btn = gr.Button(t("service.refresh_btn"), size="sm")
            
            with gr.Row():
                # Get available acestep-v15- model list
                available_models = dit_handler.get_available_acestep_v15_models()
                default_model = "acestep-v15-turbo" if "acestep-v15-turbo" in available_models else (available_models[0] if available_models else None)
                
                # Set config_path value from init_params if pre-initialized
                config_path_value = init_params.get('config_path', default_model) if service_pre_initialized else default_model
                config_path = gr.Dropdown(
                    label=t("service.model_path_label"),
                    choices=available_models,
                    value=config_path_value,
                    info=t("service.model_path_info")
                )
                # Set device value from init_params if pre-initialized
                device_value = init_params.get('device', 'auto') if service_pre_initialized else 'auto'
                device = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value=device_value,
                    label=t("service.device_label"),
                    info=t("service.device_info")
                )
            
            with gr.Row():
                # Get available 5Hz LM model list
                available_lm_models = llm_handler.get_available_5hz_lm_models()
                default_lm_model = "acestep-5Hz-lm-0.6B" if "acestep-5Hz-lm-0.6B" in available_lm_models else (available_lm_models[0] if available_lm_models else None)
                
                # Set lm_model_path value from init_params if pre-initialized
                lm_model_path_value = init_params.get('lm_model_path', default_lm_model) if service_pre_initialized else default_lm_model
                lm_model_path = gr.Dropdown(
                    label=t("service.lm_model_path_label"),
                    choices=available_lm_models,
                    value=lm_model_path_value,
                    info=t("service.lm_model_path_info")
                )
                # Set backend value from init_params if pre-initialized
                backend_value = init_params.get('backend', 'vllm') if service_pre_initialized else 'vllm'
                backend_dropdown = gr.Dropdown(
                    choices=["vllm", "pt"],
                    value=backend_value,
                    label=t("service.backend_label"),
                    info=t("service.backend_info")
                )
            
            # Checkbox options section - all checkboxes grouped together
            with gr.Row():
                # Set init_llm value from init_params if pre-initialized, otherwise use GPU config default
                init_llm_value = init_params.get('init_llm', init_lm_default) if service_pre_initialized else init_lm_default
                init_llm_checkbox = gr.Checkbox(
                    label=t("service.init_llm_label"),
                    value=init_llm_value,
                    info=t("service.init_llm_info"),
                )
                # Auto-detect flash attention availability
                flash_attn_available = dit_handler.is_flash_attention_available()
                # Set use_flash_attention value from init_params if pre-initialized
                use_flash_attention_value = init_params.get('use_flash_attention', flash_attn_available) if service_pre_initialized else flash_attn_available
                use_flash_attention_checkbox = gr.Checkbox(
                    label=t("service.flash_attention_label"),
                    value=use_flash_attention_value,
                    interactive=flash_attn_available,
                    info=t("service.flash_attention_info_enabled") if flash_attn_available else t("service.flash_attention_info_disabled")
                )
                # Set offload_to_cpu value from init_params if pre-initialized (default True)
                offload_to_cpu_value = init_params.get('offload_to_cpu', True) if service_pre_initialized else True
                offload_to_cpu_checkbox = gr.Checkbox(
                    label=t("service.offload_cpu_label"),
                    value=offload_to_cpu_value,
                    info=t("service.offload_cpu_info")
                )
                # Set offload_dit_to_cpu value from init_params if pre-initialized (default True)
                offload_dit_to_cpu_value = init_params.get('offload_dit_to_cpu', True) if service_pre_initialized else True
                offload_dit_to_cpu_checkbox = gr.Checkbox(
                    label=t("service.offload_dit_cpu_label"),
                    value=offload_dit_to_cpu_value,
                    info=t("service.offload_dit_cpu_info")
                )
                # Set compile_model value from init_params if pre-initialized (default True)
                compile_model_value = init_params.get('compile_model', True) if service_pre_initialized else True
                compile_model_checkbox = gr.Checkbox(
                    label=t("service.compile_model_label"),
                    value=compile_model_value,
                    info=t("service.compile_model_info")
                )
                # Set quantization value from init_params if pre-initialized (default True for int8_weight_only)
                quantization_value = init_params.get('quantization', True) if service_pre_initialized else True
                quantization_checkbox = gr.Checkbox(
                    label=t("service.quantization_label"),
                    value=quantization_value,
                    info=t("service.quantization_info")
                )
            
            init_btn = gr.Button(t("service.init_btn"), variant="primary", size="lg")
            # Set init_status value from init_params if pre-initialized
            init_status_value = init_params.get('init_status', '') if service_pre_initialized else ''
            init_status = gr.Textbox(label=t("service.status_label"), interactive=False, lines=3, value=init_status_value)
            
            # LoRA Configuration Section
            gr.HTML("<hr><h4>🔧 LoRA Adapter</h4>")
            with gr.Row():
                lora_path = gr.Textbox(
                    label="LoRA Path",
                    placeholder="./lora_output/final/adapter",
                    info="Path to trained LoRA adapter directory",
                    scale=3,
                )
                load_lora_btn = gr.Button("📥 Load LoRA", variant="secondary", scale=1)
                unload_lora_btn = gr.Button("🗑️ Unload", variant="secondary", scale=1)
            with gr.Row():
                use_lora_checkbox = gr.Checkbox(
                    label="Use LoRA",
                    value=False,
                    info="Enable LoRA adapter for inference",
                    scale=1,
                )
                lora_scale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    label="LoRA Scale",
                    info="LoRA influence strength (0=disabled, 1=full)",
                    scale=2,
                )
                lora_status = gr.Textbox(
                    label="LoRA Status",
                    value="No LoRA loaded",
                    interactive=False,
                    scale=2,
                )

        # Model Selection Section - Always visible for switching models
        with gr.Accordion("🔄 Model Selection", open=True) as model_selection_accordion:
            # Current model status
            current_dit_name = dit_handler.get_current_model_name() if hasattr(dit_handler, 'get_current_model_name') else "Unknown"
            current_lm_name = llm_handler.get_current_model_name() if hasattr(llm_handler, 'get_current_model_name') else "Unknown"

            with gr.Row():
                model_status_display = gr.Textbox(
                    label="Current Models",
                    value=f"DiT: {current_dit_name}\nLM: {current_lm_name}",
                    interactive=False,
                    lines=2,
                )

            with gr.Row():
                # DiT model selection - include all available models (downloaded + downloadable)
                # Full list of official DiT models from ACE-Step
                ALL_DIT_MODELS = [
                    "acestep-v15-turbo",           # Default turbo (8 steps, fast)
                    "acestep-v15-turbo-shift1",    # Turbo with shift1
                    "acestep-v15-turbo-shift3",    # Turbo with shift3
                    "acestep-v15-turbo-continuous", # Turbo with continuous shift (1-5)
                    "acestep-v15-base",            # Base model (50 steps, high diversity)
                    "acestep-v15-sft",             # SFT model (50 steps, high quality)
                ]
                # Add any locally downloaded models that might not be in the official list
                downloaded_dit_models = dit_handler.get_available_acestep_v15_models()
                all_dit_choices = list(dict.fromkeys(ALL_DIT_MODELS + downloaded_dit_models))  # Dedupe, preserve order

                current_dit = dit_handler.get_current_model_name() if hasattr(dit_handler, 'get_current_model_name') else None
                default_dit = current_dit if current_dit and current_dit in all_dit_choices else "acestep-v15-turbo"

                switch_dit_dropdown = gr.Dropdown(
                    label="DiT Model",
                    choices=all_dit_choices,
                    value=default_dit,
                    info="turbo=8 steps (fast), base/sft=50 steps (quality). Will auto-download if needed.",
                )

                # LM model selection with "None" option - include all available models
                # Full list of official LM models from ACE-Step
                ALL_LM_MODELS = [
                    "acestep-5Hz-lm-0.6B",   # Lightweight (6-12GB VRAM)
                    "acestep-5Hz-lm-1.7B",   # Balanced (12-16GB VRAM) - default
                    "acestep-5Hz-lm-4B",     # Best quality (16GB+ VRAM)
                ]
                # Add any locally downloaded models
                downloaded_lm_models = llm_handler.get_available_5hz_lm_models()
                all_lm_models = list(dict.fromkeys(ALL_LM_MODELS + downloaded_lm_models))  # Dedupe
                lm_choices = ["None (DiT only)"] + all_lm_models

                current_lm = llm_handler.get_current_model_name() if hasattr(llm_handler, 'get_current_model_name') else None
                default_lm = current_lm if current_lm and current_lm in all_lm_models else (
                    "acestep-5Hz-lm-1.7B" if "acestep-5Hz-lm-1.7B" in all_lm_models else
                    (all_lm_models[0] if all_lm_models else "None (DiT only)")
                )
                switch_lm_dropdown = gr.Dropdown(
                    label="LM Model",
                    choices=lm_choices,
                    value=default_lm,
                    info="0.6B=6-12GB, 1.7B=12-16GB, 4B=16GB+. Will auto-download if needed.",
                )

            # Multi-GPU options
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            with gr.Row(visible=num_gpus > 1):
                split_gpu_checkbox = gr.Checkbox(
                    label="🔀 Split GPUs (DiT on GPU 0, LM on GPU 1)",
                    value=num_gpus > 1,  # Default to True when multiple GPUs detected
                    info=f"Detected {num_gpus} GPUs. Enable to put DiT and LM on separate GPUs for better memory utilization.",
                )

            with gr.Row():
                switch_models_btn = gr.Button("🔄 Load / Switch Models", variant="primary", size="lg")
                unload_all_btn = gr.Button("🗑️ Unload All (Free VRAM)", variant="stop", size="lg")
                refresh_models_btn = gr.Button("🔃 Refresh", variant="secondary", size="sm")

            switch_status = gr.Textbox(
                label="Status",
                value="💡 No models loaded - select models above and click 'Load / Switch Models' to start, or just click Generate to auto-load.",
                interactive=False,
                lines=2,
            )

            # Model comparison tables in a collapsed accordion
            with gr.Accordion("📊 Model Comparison Guide", open=False):
                gr.Markdown("""
### VRAM Requirements

| Your GPU VRAM | Recommended LM Model | Notes |
|---------------|---------------------|-------|
| ≤6GB | None (DiT only) | LM disabled to save memory |
| 6-12GB | `acestep-5Hz-lm-0.6B` | Lightweight, good balance |
| 12-16GB | `acestep-5Hz-lm-1.7B` | Better quality (default) |
| ≥16GB | `acestep-5Hz-lm-4B` | Best quality and audio understanding |

---

### DiT Models (Diffusion Transformer)

| Model | CFG | Steps | Refer Audio | Quality | Diversity | Speed |
|-------|-----|-------|-------------|---------|-----------|-------|
| `acestep-v15-turbo` | ❌ | 8 | ✅ | ⭐⭐⭐⭐ Very High | ⭐⭐⭐ Medium | 🚀 Fast |
| `acestep-v15-turbo-shift1` | ❌ | 8 | ✅ | ⭐⭐⭐⭐ Very High | ⭐⭐⭐ Medium | 🚀 Fast |
| `acestep-v15-turbo-shift3` | ❌ | 8 | ✅ | ⭐⭐⭐⭐ Very High | ⭐⭐⭐ Medium | 🚀 Fast |
| `acestep-v15-turbo-continuous` | ❌ | 8 | ✅ | ⭐⭐⭐⭐ Very High | ⭐⭐⭐ Medium | 🚀 Fast |
| `acestep-v15-sft` | ✅ | 50 | ✅ | ⭐⭐⭐⭐ High | ⭐⭐⭐ Medium | 🐢 Slow |
| `acestep-v15-base` | ✅ | 50 | ✅ | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ High | 🐢 Slow |

**Turbo variants:** Fast generation (8 steps), no CFG needed, best for most use cases.
- `shift1/shift3`: Fixed shift values for consistent results
- `continuous`: Variable shift (1-5) for more control

**Base/SFT:** Slower (50 steps) but support CFG for finer control. Base has higher diversity, SFT has higher quality.

---

### LM Models (Language Model)

| Model | VRAM | Audio Understanding | Composition | Copy Melody | CoT Metas |
|-------|------|---------------------|-------------|-------------|-----------|
| `acestep-5Hz-lm-0.6B` | 6-12GB | ⭐⭐ Medium | ⭐⭐ Medium | ⭐ Weak | ✅ |
| `acestep-5Hz-lm-1.7B` | 12-16GB | ⭐⭐ Medium | ⭐⭐ Medium | ⭐⭐ Medium | ✅ |
| `acestep-5Hz-lm-4B` | 16GB+ | ⭐⭐⭐ Strong | ⭐⭐⭐ Strong | ⭐⭐⭐ Strong | ✅ |

**LM Features:**
- **Audio Understanding**: Analyze reference audio to extract style/mood
- **Composition**: Generate coherent musical structures
- **Copy Melody**: Replicate melodic patterns from reference
- **CoT Metas**: Chain-of-thought metadata generation (BPM, key, etc.)

**None (DiT only)**: Fastest generation, lowest VRAM, but no AI-enhanced metadata or audio understanding.
                """)

        # Inputs
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion(t("generation.required_inputs"), open=True):
                    # Task type
                    # Determine initial task_type choices based on actual model in use
                    # When service is pre-initialized, use config_path from init_params
                    actual_model = init_params.get('config_path', default_model) if service_pre_initialized else default_model
                    actual_model_lower = (actual_model or "").lower()
                    if "turbo" in actual_model_lower:
                        initial_task_choices = TASK_TYPES_TURBO
                    else:
                        initial_task_choices = TASK_TYPES_BASE
                    
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            task_type = gr.Dropdown(
                                choices=initial_task_choices,
                                value="text2music",
                                label=t("generation.task_type_label"),
                                info=t("generation.task_type_info"),
                            )
                        with gr.Column(scale=7):
                            instruction_display_gen = gr.Textbox(
                                label=t("generation.instruction_label"),
                                value=DEFAULT_DIT_INSTRUCTION,
                                interactive=False,
                                lines=1,
                                info=t("generation.instruction_info"),
                            )
                        with gr.Column(scale=1, min_width=100):
                            load_file = gr.UploadButton(
                                t("generation.load_btn"),
                                file_types=[".json"],
                                file_count="single",
                                variant="secondary",
                                size="sm",
                            )
                    
                    track_name = gr.Dropdown(
                        choices=TRACK_NAMES,
                        value=None,
                        label=t("generation.track_name_label"),
                        info=t("generation.track_name_info"),
                        visible=False
                    )
                    
                    complete_track_classes = gr.CheckboxGroup(
                        choices=TRACK_NAMES,
                        label=t("generation.track_classes_label"),
                        info=t("generation.track_classes_info"),
                        visible=False
                    )
                    
                    # Audio uploads
                    audio_uploads_accordion = gr.Accordion(t("generation.audio_uploads"), open=False)
                    with audio_uploads_accordion:
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2):
                                reference_audio = gr.Audio(
                                    label=t("generation.reference_audio"),
                                    type="filepath",
                                )
                            with gr.Column(scale=7):
                                src_audio = gr.Audio(
                                    label=t("generation.source_audio"),
                                    type="filepath",
                                )
                            with gr.Column(scale=1, min_width=80):
                                convert_src_to_codes_btn = gr.Button(
                                    t("generation.convert_codes_btn"),
                                    variant="secondary",
                                    size="sm"
                                )
                        
                    # Audio Codes for text2music - single input for transcription or cover task
                    with gr.Accordion(t("generation.lm_codes_hints"), open=False, visible=True) as text2music_audio_codes_group:
                        with gr.Row(equal_height=True):
                            text2music_audio_code_string = gr.Textbox(
                                label=t("generation.lm_codes_label"),
                                placeholder=t("generation.lm_codes_placeholder"),
                                lines=6,
                                info=t("generation.lm_codes_info"),
                                scale=9,
                            )
                            transcribe_btn = gr.Button(
                                t("generation.transcribe_btn"),
                                variant="secondary",
                                size="sm",
                                scale=1,
                            )
                    
                    # Repainting controls
                    with gr.Group(visible=False) as repainting_group:
                        gr.HTML(f"<h5>{t('generation.repainting_controls')}</h5>")
                        with gr.Row():
                            repainting_start = gr.Number(
                                label=t("generation.repainting_start"),
                                value=0.0,
                                step=0.1,
                            )
                            repainting_end = gr.Number(
                                label=t("generation.repainting_end"),
                                value=-1,
                                minimum=-1,
                                step=0.1,
                            )
                    
                    # Simple/Custom Mode Toggle
                    # Default to Custom mode - Simple is for users who need help with lyrics/style
                    with gr.Row(visible=not service_mode):
                        generation_mode = gr.Radio(
                            choices=[
                                ("✨ Simple (AI helps write lyrics & style)", "simple"),
                                ("🎛️ Custom (full control)", "custom"),
                            ],
                            value="custom",
                            label="Generation Mode",
                            info="Simple: AI generates lyrics & style from a brief description. Custom: You provide the full details.",
                        )
                    
                    # Simple Mode Components - hidden in service mode
                    with gr.Group(visible=not service_mode) as simple_mode_group:
                        with gr.Row(equal_height=True):
                            simple_query_input = gr.Textbox(
                                label=t("generation.simple_query_label"),
                                placeholder=t("generation.simple_query_placeholder"),
                                lines=2,
                                info=t("generation.simple_query_info"),
                                scale=12,
                            )

                            with gr.Column(scale=1, min_width=100):
                                random_desc_btn = gr.Button(
                                    "🎲",
                                    variant="secondary",
                                    size="sm",
                                    scale=2
                                )
                        
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, variant="compact"):
                                simple_instrumental_checkbox = gr.Checkbox(
                                    label=t("generation.instrumental_label"),
                                    value=False,
                                )
                            with gr.Column(scale=18):
                                create_sample_btn = gr.Button(
                                    t("generation.create_sample_btn"),
                                    variant="primary",
                                    size="lg",
                                )
                            with gr.Column(scale=1, variant="compact"):
                                simple_vocal_language = gr.Dropdown(
                                    choices=VALID_LANGUAGES,
                                    value="unknown",
                                    allow_custom_value=True,
                                    label=t("generation.simple_vocal_language_label"),
                                    interactive=True,
                                )
                    
                    # State to track if sample has been created in Simple mode
                    simple_sample_created = gr.State(value=False)
                
                # Music Caption - wrapped in accordion that can be collapsed in Simple mode
                # Default to expanded for better UX
                with gr.Accordion(t("generation.caption_title"), open=True) as caption_accordion:
                    with gr.Row(equal_height=True):
                        captions = gr.Textbox(
                            label=t("generation.caption_label"),
                            placeholder=t("generation.caption_placeholder"),
                            lines=3,
                            info=t("generation.caption_info"),
                            scale=12,
                        )
                        with gr.Column(scale=1, min_width=100):
                            sample_btn = gr.Button(
                                "🎲",
                                variant="secondary",
                                size="sm",
                                scale=2,
                            )
                # Lyrics - wrapped in accordion that can be collapsed in Simple mode
                # Default to expanded for better UX
                with gr.Accordion(t("generation.lyrics_title"), open=True) as lyrics_accordion:
                    lyrics = gr.Textbox(
                        label=t("generation.lyrics_label"),
                        placeholder=t("generation.lyrics_placeholder"),
                        lines=8,
                        info=t("generation.lyrics_info")
                    )
                    
                    with gr.Row(variant="compact", equal_height=True):
                        instrumental_checkbox = gr.Checkbox(
                            label=t("generation.instrumental_label"),
                            value=False,
                            scale=1,
                            min_width=120,
                            container=True,
                        )
                        
                        # 中间：语言选择 (Dropdown)
                        # 移除 gr.HTML hack，直接使用 label 参数，Gradio 会自动处理对齐
                        vocal_language = gr.Dropdown(
                            choices=VALID_LANGUAGES,
                            value="unknown",
                            label=t("generation.vocal_language_label"),
                            show_label=False, 
                            container=True, 
                            allow_custom_value=True,
                            scale=3,
                        )
                        
                        # 右侧：格式化按钮 (Button)
                        # 放在同一行最右侧，操作更顺手
                        format_btn = gr.Button(
                            t("generation.format_btn"),
                            variant="secondary",
                            scale=1,
                            min_width=80,
                        )
                
                # Optional Parameters
                # In service mode: auto-expand
                with gr.Accordion(t("generation.optional_params"), open=service_mode) as optional_params_accordion:
                    with gr.Row():
                        bpm = gr.Number(
                            label=t("generation.bpm_label"),
                            value=None,
                            step=1,
                            info=t("generation.bpm_info")
                        )
                        key_scale = gr.Textbox(
                            label=t("generation.keyscale_label"),
                            placeholder=t("generation.keyscale_placeholder"),
                            value="",
                            info=t("generation.keyscale_info")
                        )
                        time_signature = gr.Dropdown(
                            choices=["", "2", "3", "4", "6", "N/A"],
                            value="",
                            label=t("generation.timesig_label"),
                            allow_custom_value=True,
                            info=t("generation.timesig_info")
                        )
                        audio_duration = gr.Number(
                            label=t("generation.duration_label"),
                            value=-1,
                            minimum=-1,
                            maximum=float(max_duration),
                            step=0.1,
                            info=t("generation.duration_info") + f" (Max: {max_duration}s / {max_duration // 60} min)"
                        )
                        batch_size_input = gr.Number(
                            label=t("generation.batch_size_label"),
                            value=default_batch_size,
                            minimum=1,
                            maximum=max_batch_size,
                            step=1,
                            info=t("generation.batch_size_info") + f" (Max: {max_batch_size})",
                            interactive=not service_mode  # Fixed in service mode
                        )
                        sequential_generation = gr.Checkbox(
                            label="Sequential Generation (Queue)",
                            value=False,
                            info="Generate one at a time instead of in parallel. Uses less VRAM but takes longer.",
                            interactive=not service_mode
                        )

        # Advanced Settings
        # Default UI settings use turbo mode (max 20 steps, default 8, show shift with default 3)
        # These will be updated after model initialization based on handler.is_turbo_model()
        with gr.Accordion(t("generation.advanced_settings"), open=False):
            with gr.Row():
                inference_steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=8,
                    step=1,
                    label=t("generation.inference_steps_label"),
                    info=t("generation.inference_steps_info")
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=7.0,
                    step=0.1,
                    label=t("generation.guidance_scale_label"),
                    info=t("generation.guidance_scale_info"),
                    visible=False
                )
                with gr.Column():
                    seed = gr.Textbox(
                        label=t("generation.seed_label"),
                        value="-1",
                        info=t("generation.seed_info")
                    )
                    random_seed_checkbox = gr.Checkbox(
                        label=t("generation.random_seed_label"),
                        value=True,
                        info=t("generation.random_seed_info")
                    )
                audio_format = gr.Dropdown(
                    choices=["mp3", "flac"],
                    value="mp3",
                    label=t("generation.audio_format_label"),
                    info=t("generation.audio_format_info"),
                    interactive=not service_mode  # Fixed in service mode
                )
            
            with gr.Row():
                use_adg = gr.Checkbox(
                    label=t("generation.use_adg_label"),
                    value=False,
                    info=t("generation.use_adg_info"),
                    visible=False
                )
                shift = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,
                    step=0.1,
                    label=t("generation.shift_label"),
                    info=t("generation.shift_info"),
                    visible=True
                )
                infer_method = gr.Dropdown(
                    choices=["ode", "sde"],
                    value="ode",
                    label=t("generation.infer_method_label"),
                    info=t("generation.infer_method_info"),
                )
            
            with gr.Row():
                custom_timesteps = gr.Textbox(
                    label=t("generation.custom_timesteps_label"),
                    placeholder="0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0",
                    value="",
                    info=t("generation.custom_timesteps_info"),
                )
            
            with gr.Row():
                cfg_interval_start = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    label=t("generation.cfg_interval_start"),
                    visible=False
                )
                cfg_interval_end = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label=t("generation.cfg_interval_end"),
                    visible=False
                )

            # LM (Language Model) Parameters
            gr.HTML(f"<h4>{t('generation.lm_params_title')}</h4>")
            with gr.Row():
                lm_temperature = gr.Slider(
                    label=t("generation.lm_temperature_label"),
                    minimum=0.0,
                    maximum=2.0,
                    value=0.85,
                    step=0.1,
                    scale=1,
                    info=t("generation.lm_temperature_info")
                )
                lm_cfg_scale = gr.Slider(
                    label=t("generation.lm_cfg_scale_label"),
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    scale=1,
                    info=t("generation.lm_cfg_scale_info")
                )
                lm_top_k = gr.Slider(
                    label=t("generation.lm_top_k_label"),
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    scale=1,
                    info=t("generation.lm_top_k_info")
                )
                lm_top_p = gr.Slider(
                    label=t("generation.lm_top_p_label"),
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.01,
                    scale=1,
                    info=t("generation.lm_top_p_info")
                )
            
            with gr.Row():
                lm_negative_prompt = gr.Textbox(
                    label=t("generation.lm_negative_prompt_label"),
                    value="NO USER INPUT",
                    placeholder=t("generation.lm_negative_prompt_placeholder"),
                    info=t("generation.lm_negative_prompt_info"),
                    lines=2,
                    scale=2,
                )
            
            with gr.Row():
                use_cot_metas = gr.Checkbox(
                    label=t("generation.cot_metas_label"),
                    value=True,
                    info=t("generation.cot_metas_info"),
                    scale=1,
                )
                use_cot_language = gr.Checkbox(
                    label=t("generation.cot_language_label"),
                    value=True,
                    info=t("generation.cot_language_info"),
                    scale=1,
                )
                constrained_decoding_debug = gr.Checkbox(
                    label=t("generation.constrained_debug_label"),
                    value=False,
                    info=t("generation.constrained_debug_info"),
                    scale=1,
                    interactive=not service_mode  # Fixed in service mode
                )
            
            with gr.Row():
                auto_score = gr.Checkbox(
                    label=t("generation.auto_score_label"),
                    value=False,
                    info=t("generation.auto_score_info"),
                    scale=1,
                    interactive=not service_mode  # Fixed in service mode
                )
                auto_lrc = gr.Checkbox(
                    label=t("generation.auto_lrc_label"),
                    value=False,
                    info=t("generation.auto_lrc_info"),
                    scale=1,
                    interactive=not service_mode  # Fixed in service mode
                )
                lm_batch_chunk_size = gr.Number(
                    label=t("generation.lm_batch_chunk_label"),
                    value=8,
                    minimum=1,
                    maximum=32,
                    step=1,
                    info=t("generation.lm_batch_chunk_info"),
                    scale=1,
                    interactive=not service_mode  # Fixed in service mode
                )
            
            with gr.Row():
                audio_cover_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label=t("generation.codes_strength_label"),
                    info=t("generation.codes_strength_info"),
                    scale=1,
                )
                score_scale = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label=t("generation.score_sensitivity_label"),
                    info=t("generation.score_sensitivity_info"),
                    scale=1,
                    visible=not service_mode  # Hidden in service mode
                )
        
        # Set generate_btn to interactive if service is pre-initialized
        generate_btn_interactive = init_params.get('enable_generate', False) if service_pre_initialized else False
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, variant="compact"):
                think_checkbox = gr.Checkbox(
                    label=t("generation.think_label"),
                    value=True,
                    scale=1,
                )
                allow_lm_batch = gr.Checkbox(
                    label=t("generation.parallel_thinking_label"),
                    value=True,
                    scale=1,
                )
            with gr.Column(scale=18):
                generate_btn = gr.Button(t("generation.generate_btn"), variant="primary", size="lg", interactive=generate_btn_interactive)
            with gr.Column(scale=1, variant="compact"):
                autogen_checkbox = gr.Checkbox(
                    label=t("generation.autogen_label"),
                    value=False,  # Default to False for both service and local modes
                    scale=1,
                    interactive=not service_mode  # Not selectable in service mode
                )
                use_cot_caption = gr.Checkbox(
                    label=t("generation.caption_rewrite_label"),
                    value=True,
                    scale=1,
                )
    
    return {
        "service_config_accordion": service_config_accordion,
        "language_dropdown": language_dropdown,
        "checkpoint_dropdown": checkpoint_dropdown,
        "refresh_btn": refresh_btn,
        "config_path": config_path,
        "device": device,
        "init_btn": init_btn,
        "init_status": init_status,
        "lm_model_path": lm_model_path,
        "init_llm_checkbox": init_llm_checkbox,
        "backend_dropdown": backend_dropdown,
        "use_flash_attention_checkbox": use_flash_attention_checkbox,
        "offload_to_cpu_checkbox": offload_to_cpu_checkbox,
        "offload_dit_to_cpu_checkbox": offload_dit_to_cpu_checkbox,
        "compile_model_checkbox": compile_model_checkbox,
        "quantization_checkbox": quantization_checkbox,
        # LoRA components
        "lora_path": lora_path,
        "load_lora_btn": load_lora_btn,
        "unload_lora_btn": unload_lora_btn,
        "use_lora_checkbox": use_lora_checkbox,
        "lora_scale_slider": lora_scale_slider,
        "lora_status": lora_status,
        # Model Selection components
        "model_selection_accordion": model_selection_accordion,
        "model_status_display": model_status_display,
        "switch_dit_dropdown": switch_dit_dropdown,
        "switch_lm_dropdown": switch_lm_dropdown,
        "switch_models_btn": switch_models_btn,
        "unload_all_btn": unload_all_btn,
        "split_gpu_checkbox": split_gpu_checkbox,
        "refresh_models_btn": refresh_models_btn,
        "switch_status": switch_status,
        "task_type": task_type,
        "instruction_display_gen": instruction_display_gen,
        "track_name": track_name,
        "complete_track_classes": complete_track_classes,
        "audio_uploads_accordion": audio_uploads_accordion,
        "reference_audio": reference_audio,
        "src_audio": src_audio,
        "convert_src_to_codes_btn": convert_src_to_codes_btn,
        "text2music_audio_code_string": text2music_audio_code_string,
        "transcribe_btn": transcribe_btn,
        "text2music_audio_codes_group": text2music_audio_codes_group,
        "lm_temperature": lm_temperature,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption,
        "use_cot_language": use_cot_language,
        "repainting_group": repainting_group,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "audio_cover_strength": audio_cover_strength,
        # Simple/Custom Mode Components
        "generation_mode": generation_mode,
        "simple_mode_group": simple_mode_group,
        "simple_query_input": simple_query_input,
        "random_desc_btn": random_desc_btn,
        "simple_instrumental_checkbox": simple_instrumental_checkbox,
        "simple_vocal_language": simple_vocal_language,
        "create_sample_btn": create_sample_btn,
        "simple_sample_created": simple_sample_created,
        "caption_accordion": caption_accordion,
        "lyrics_accordion": lyrics_accordion,
        "optional_params_accordion": optional_params_accordion,
        # Existing components
        "captions": captions,
        "sample_btn": sample_btn,
        "load_file": load_file,
        "lyrics": lyrics,
        "vocal_language": vocal_language,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "audio_duration": audio_duration,
        "batch_size_input": batch_size_input,
        "sequential_generation": sequential_generation,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "random_seed_checkbox": random_seed_checkbox,
        "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "shift": shift,
        "infer_method": infer_method,
        "custom_timesteps": custom_timesteps,
        "audio_format": audio_format,
        "think_checkbox": think_checkbox,
        "autogen_checkbox": autogen_checkbox,
        "generate_btn": generate_btn,
        "instrumental_checkbox": instrumental_checkbox,
        "format_btn": format_btn,
        "constrained_decoding_debug": constrained_decoding_debug,
        "score_scale": score_scale,
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score,
        "auto_lrc": auto_lrc,
        "lm_batch_chunk_size": lm_batch_chunk_size,
        # GPU config values for validation
        "gpu_config": gpu_config,
        "max_duration": max_duration,
        "max_batch_size": max_batch_size,
    }

