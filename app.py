import os
import time
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import traceback
import threading
from datetime import datetime
from clearml import Task
import tempfile # Import for temporary files

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer # Still imported but not used in this optimized version
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        """
        model_path: str, the path or name of the single VibeVoice model to load.
        """
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps

        self.is_generating = False

        # Single model and processor
        self.model = None
        self.processor = None

        self.available_voices = {}

        self.load_model()
        self.setup_voice_presets()

    def load_model(self):
        print(f"Loading processor and model {self.model_path} on {self.device}...")
        try:
            self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True
            ).to(self.device) # Load directly to device
            print(f"Model {self.model_path} successfully loaded on {self.device}.")
        except Exception as e:
            print(f"Error loading model {self.model_path}: {e}")
            raise

    def setup_voice_presets(self):
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            return
        wav_files = [f for f in os.listdir(voices_dir)
                     if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'))]
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            self.available_voices[name] = os.path.join(voices_dir, wav_file)
        print(f"Voices loaded: {list(self.available_voices.keys())}")

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

    @torch.inference_mode()
    def generate_podcast(self,
                         num_speakers: int,
                         script: str, # Script now comes directly as string
                         speaker_1: str = None,
                         speaker_2: str = None,
                         speaker_3: str = None,
                         speaker_4: str = None,
                         cfg_scale: float = 1.3):
        """
        Generates a podcast as a single audio file from a script (string) and saves it.
        Non-streaming.
        """
        try:
            if self.model is None or self.processor is None:
                raise gr.Error("Model and/or processor not loaded. Please check logs.")

            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            self.is_generating = True

            if not script.strip():
                raise gr.Error("Error: Please provide a script.")

            script = script.replace("’", "'")

            if not 1 <= num_speakers <= 4:
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")

            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            for i, speaker_name in enumerate(selected_speakers):
                if not speaker_name or speaker_name not in self.available_voices:
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")

            log = f"🎙️ Generating podcast with {num_speakers} speakers\n"
            log += f"🧠 Model: {self.model_path}\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"

            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)

            log += f"✅ Loaded {len(voice_samples)} voice samples\n"

            lines = script.strip().split('\n')
            formatted_script_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Assuming script lines might already be formatted or need simple speaker assignment
                if line.lower().startswith('speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    # Simple round-robin assignment if no explicit speaker is given
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id + 1}: {line}") # 1-indexed for clarity

            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n"
            log += "🔄 Processing with VibeVoice...\n"

            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device) # Move inputs to device

            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
            )
            generation_time = time.time() - start_time

            if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs[0] is not None:
                audio_tensor = outputs.speech_outputs[0]
                audio = audio_tensor.cpu().float().numpy()
            else:
                raise gr.Error("❌ Error: No audio was generated by the model. Please try again.")

            if audio.ndim > 1:
                audio = audio.squeeze()

            sample_rate = 24000

            total_duration = len(audio) / sample_rate
            log += f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
            log += f"🎵 Final audio duration: {total_duration:.2f} seconds\n"
            log += f"✅ Successfully generated podcast.\n"

            self.is_generating = False
            return (sample_rate, audio), log

        except gr.Error as e:
            self.is_generating = False
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            return None, error_msg

        except Exception as e:
            self.is_generating = False
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return None, error_msg


def create_demo_interface(demo_instance: VibeVoiceDemo):
    with gr.Blocks(
        title="VibeVoice - AI Podcast Generator",
        theme=gr.themes.Soft()
    ) as interface:

        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Vibe Podcasting</h1>
            <p>Generating Long-form Multi-speaker AI Podcast with VibeVoice</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ Podcast Settings")

                gr.Markdown(f"**Loaded Model:** `{demo_instance.model_path}`")

                num_speakers = gr.Slider(
                    minimum=1, maximum=4, value=2, step=1,
                    label="Number of Speakers",
                    elem_classes="slider-container"
                )

                gr.Markdown("### 🎭 Speaker Selection")
                available_speaker_names = list(demo_instance.available_voices.keys())
                default_speakers = ['raora', 'karen', 'kore', 'stephany']

                speaker_selections = []
                for i in range(4):
                    default_value = default_speakers[i] if i < len(default_speakers) else None
                    speaker = gr.Dropdown(
                        choices=available_speaker_names,
                        value=default_value,
                        label=f"Speaker {i+1}",
                        visible=(i < 2),
                        elem_classes="speaker-item"
                    )
                    speaker_selections.append(speaker)

                gr.Markdown("### ⚙️ Advanced Settings")
                with gr.Accordion("Generation Parameters", open=False):
                    cfg_scale = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.3, step=0.05,
                        label="CFG Scale (Guidance Strength)",
                        elem_classes="slider-container"
                    )

            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 Script Input")

                # Field for direct script input (string)
                script_text_input = gr.Textbox(
                    label="Enter Script Directly",
                    placeholder="Type or paste your podcast script here...",
                    lines=10,
                    max_lines=20,
                    elem_classes="script-input-container",
                    interactive=True # Ensure it's editable
                )

                gr.Markdown("--- OR ---") # Separator

                # Field for file upload
                script_file_upload = gr.File(
                    label="Upload Script (.txt)",
                    file_types=[".txt"],
                    elem_classes="script-input-container"
                )

                generate_btn = gr.Button(
                    "🚀 Generate Podcast", size="lg",
                    variant="primary", elem_classes="generate-btn", scale=1
                )

                gr.Markdown("### 🎵 Generated Podcast")
                complete_audio_output = gr.Audio(
                    label="Complete Podcast",
                    type="numpy",
                    elem_classes="audio-output",
                    autoplay=False,
                    show_download_button=True,
                    visible=True
                )

                log_output = gr.Textbox(
                    label="Generation Log",
                    lines=8, max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )

        def update_speaker_visibility(num_speakers):
            return [gr.update(visible=(i < num_speakers)) for i in range(4)]

        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_selections
        )

        # Helper function to read file content, if file is provided
        def get_script_content(uploaded_file = None, text_input_script = None):
            if uploaded_file is not None:
                try:
                    with open(uploaded_file.name, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    gr.Warning(f"Error reading uploaded file: {e}. Using text input instead.")
                    return text_input_script
            return text_input_script
            
        def generate_podcast_wrapper(num_speakers, uploaded_file=None, text_input_script=None, *speakers_and_params):    
            task = Task.init(
                project_name='Vibevoice',
                task_name=f'vibevoice'
            )
            
            script_content = get_script_content(uploaded_file, text_input_script)

            if not script_content.strip():
                task.mark_as_failed(status_reason="❌ Error: Please provide a script either by uploading a file or typing directly.", status_message=None)
                task.close()
                return None, "❌ Error: Please provide a script either by uploading a file or typing directly.", None

            try:
                speakers = speakers_and_params[:4]
                cfg_scale_val = speakers_and_params[4]

                audio, log = demo_instance.generate_podcast( # generate_podcast now returns (audio, log)
                    num_speakers=int(num_speakers),
                    script=script_content, # Pass the resolved script content
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale_val,
                )
                
                task.upload_artifact(
                    name='Audio Gerado',
                    artifact_object=audio
                )
                
                task.close()
                return audio, log
            except Exception as e:
                traceback.print_exc()
                task.mark_as_failed(status_reason=None, status_message=str(e))
                task.close()
                return None, f"❌ Error: {str(e)}"


        generate_btn.click(
            fn=generate_podcast_wrapper,
            inputs=[num_speakers, script_file_upload, script_text_input] + speaker_selections + [cfg_scale],
            outputs=[complete_audio_output, log_output],
            queue=True
        )

    return interface


def run_demo(
    model_path: str = None,
    device: str = "cpu",
    inference_steps: int = 5,
    share: bool = True,
):
    """
    model_path: str, the path or name of the single VibeVoice model to load.
    """
    if model_path is None:
        model_path = "microsoft/VibeVoice-1.5B" # Default to 1.5B

    set_seed(42)
    demo_instance = VibeVoiceDemo(model_path, device, inference_steps)
    interface = create_demo_interface(demo_instance)
    interface.queue().launch(
        share=share,
        server_name="0.0.0.0" if share else "127.0.0.1",
        show_error=True,
        show_api=True,
        mcp_server=True
    )


if __name__ == "__main__":
    run_demo(device="cpu")