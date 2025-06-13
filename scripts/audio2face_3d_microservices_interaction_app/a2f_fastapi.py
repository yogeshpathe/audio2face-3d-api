#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ================================================================================================
# This script is a FastAPI web server that acts as a client for NVIDIA's A2F pipeline.
#
# NEW IN THIS VERSION:
# - Added a /inference-from-prompt endpoint to generate a full conversational turn.
# - Integrates with the Google Gemini API to generate responses from user prompts.
# - The Gemini response is then used to generate audio and facial animation.
# ================================================================================================

import asyncio
import os
import time
import warnings
import io
import base64

import numpy as np
import grpc
import scipy.io.wavfile
import yaml
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form

from nvidia_ace.health_pb2_grpc import HealthStub
from nvidia_ace.health_pb2 import HealthCheckRequest, HealthCheckResponse
from nvidia_ace.animation_data.v1_pb2 import AnimationData, AnimationDataStreamHeader
from nvidia_ace.a2f.v1_pb2 import AudioWithEmotion, EmotionPostProcessingParameters, \
    FaceParameters, BlendShapeParameters, EmotionParameters
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
from nvidia_ace.controller.v1_pb2 import AudioStream, AudioStreamHeader
from nvidia_ace.emotion_with_timecode.v1_pb2 import EmotionWithTimeCode
from nvidia_ace.emotion_aggregate.v1_pb2 import EmotionAggregate


# --- New Imports for Text-to-Speech and Generative AI ---
from gtts import gTTS
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai


# --- FastAPI App Initialization ---
app = FastAPI(
    title="NVIDIA ACE A2F-3D Conversational Client API",
    description="An API wrapper that connects the Google Gemini API, Text-to-Speech, and the NVIDIA ACE A2F-3D pipeline.",
    version="3.0.0"
)

# --- Middleware Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Original Constants and Configuration ---
BITS_PER_BYTE = 8
REQUIRED_BITS_PER_SAMPLE = 16
REQUIRED_CHANNEL_COUNT = 1
REQUIRED_AUDIO_FORMAT = AudioHeader.AUDIO_FORMAT_PCM

# --- Helper Functions (Adapted and New) ---

async def generate_response_from_gemini(user_prompt: str) -> str:
    """
    Generates a short, polite customer service response using the Gemini API.
    Requires the GEMINI_API_KEY environment variable to be set.
    """
    api_key = ""
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable not set. Please provide the API key."
        )

    try:
        genai.configure(api_key=api_key)

        # System prompt to guide the model's behavior
        system_instruction = (
            "You are a friendly and polite customer service assistant for a digital human project. "
            "Your responses must be short and conversational, "
            "Do not use markdown, lists, or any special formatting. Just provide a natural, spoken response."
        )

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            system_instruction=system_instruction
        )

        response = await model.generate_content_async(user_prompt)
        
        # Clean up the response text
        clean_text = response.text.strip()
        return clean_text

    except Exception as e:
        # Handle potential API errors from Google
        raise HTTPException(
            status_code=503,
            detail=f"An error occurred with the Gemini API: {e}"
        )


from google.cloud import texttospeech

def generate_audio_from_text(text: str) -> bytes:
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16  # WAV PCM
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content

def get_audio_bit_format(audio_header: AudioHeader):
    if audio_header.audio_format == AudioHeader.AUDIO_FORMAT_PCM:
        if audio_header.bits_per_sample == 16:
            return np.int16
    return None

def create_wav_in_memory(audio_header: AudioHeader, audio_buffer: bytes) -> bytes:
    dtype = get_audio_bit_format(audio_header)
    if dtype is None:
        raise ValueError("Unknown format for audio output")
    audio_data_to_save = np.frombuffer(audio_buffer, dtype=dtype)
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, audio_header.samples_per_second, audio_data_to_save)
    wav_buffer.seek(0)
    return wav_buffer.read()


def parse_emotion_data(animation_data, emotion_key_frames):
    emotion_aggregate: EmotionAggregate = EmotionAggregate()
    if (animation_data.metadata["emotion_aggregate"] and
        animation_data.metadata["emotion_aggregate"].Unpack(emotion_aggregate)):
        for item in emotion_aggregate.a2e_output:
            emotion_key_frames["a2e_output"].append({"time_code": item.time_code, "emotion_values": dict(item.emotion)})
        for item in emotion_aggregate.input_emotions:
            emotion_key_frames["input"].append({"time_code": item.time_code, "emotion_values": dict(item.emotion)})
        for item in emotion_aggregate.a2f_smoothed_output:
            emotion_key_frames["a2f_smoothed_output"].append({"time_code": item.time_code, "emotion_values": dict(item.emotion)})

async def read_from_stream_api(stream):
    bs_names, animation_key_frames, audio_buffer = [], [], b''
    audio_header: AudioHeader = None
    emotion_key_frames = {"input": [], "a2e_output": [], "a2f_smoothed_output": []}
    final_status = None
    while True:
        message = await stream.read()
        if message == grpc.aio.EOF:
            return {"animation_key_frames": animation_key_frames, "emotion_key_frames": emotion_key_frames,
                    "audio_header": audio_header, "audio_buffer": audio_buffer, "final_status": final_status}
        if message.HasField("animation_data_stream_header"):
            header = message.animation_data_stream_header
            bs_names, audio_header = header.skel_animation_header.blend_shapes, header.audio_header
        elif message.HasField("animation_data"):
            data = message.animation_data
            parse_emotion_data(data, emotion_key_frames)
            for blendshapes in data.skel_animation.blend_shape_weights:
                bs_values_dict = dict(zip(bs_names, blendshapes.values))
                animation_key_frames.append({"timeCode": blendshapes.time_code, "blendShapes": bs_values_dict})
            audio_buffer += data.audio.audio_buffer
        elif message.HasField("status"):
            final_status = message.status

async def write_to_stream_api(stream, config_bytes: bytes, audio_bytes: bytes):
    try:
        samplerate, data = scipy.io.wavfile.read(io.BytesIO(audio_bytes))
        config = yaml.safe_load(io.BytesIO(config_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio or config file format: {e}")

    bit_depth = data.dtype.itemsize * BITS_PER_BYTE
    channels = data.shape[1] if len(data.shape) > 1 else 1

    if bit_depth != REQUIRED_BITS_PER_SAMPLE or channels != REQUIRED_CHANNEL_COUNT:
        raise HTTPException(status_code=400, detail="Unsupported audio. Must be 16-bit mono PCM.")

    header = AudioStreamHeader(
        audio_header=AudioHeader(samples_per_second=samplerate, bits_per_sample=bit_depth, channel_count=channels, audio_format=REQUIRED_AUDIO_FORMAT),
        emotion_post_processing_params=EmotionPostProcessingParameters(**config["post_processing_parameters"]),
        face_params=FaceParameters(float_params=config["face_parameters"]),
        blendshape_params=BlendShapeParameters(
                bs_weight_multipliers=config["blendshape_parameters"]["multipliers"],
                bs_weight_offsets=config["blendshape_parameters"]["offsets"],
                enable_clamping_bs_weight=config["blendshape_parameters"]["enable_clamping_bs_weight"]
            ),
        emotion_params=EmotionParameters(
                live_transition_time=config["live_transition_time"],
                beginning_emotion=config["beginning_emotion"],
        )
    )
    await stream.write(AudioStream(audio_stream_header=header))

    chunk_size = samplerate  # 1-second chunks
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        payload = AudioWithEmotion(audio_buffer=chunk.astype(np.int16).tobytes())
        if i == 0 and "emotion_with_timecode_list" in config:
            emotions = [EmotionWithTimeCode(emotion={**v["emotions"]}, time_code=v["time_code"]) for v in config.get("emotion_with_timecode_list", {}).values()]
            payload.emotions.extend(emotions)
        await stream.write(AudioStream(audio_with_emotion=payload))
    await stream.write(AudioStream(end_of_audio=AudioStream.EndOfAudio()))

def create_grpc_channel(server_address, secure_mode=None, root_cert_path=None, client_cert_path=None, client_key_path=None):
    if secure_mode == "mtls":
        if not all([root_cert_path, client_cert_path, client_key_path]):
            raise HTTPException(status_code=500, detail="mTLS mode requires root, client cert, and client key paths.")
        with open(root_cert_path, 'rb') as f: root_cert = f.read()
        with open(client_cert_path, 'rb') as f: client_cert = f.read()
        with open(client_key_path, 'rb') as f: client_key = f.read()
        credentials = grpc.ssl_channel_credentials(root_certificates=root_cert, private_key=client_key, certificate_chain=client_cert)
        return grpc.aio.secure_channel(server_address, credentials)
    elif secure_mode == "tls":
        if not root_cert_path:
            raise HTTPException(status_code=500, detail="TLS mode requires a root certificate path.")
        with open(root_cert_path, 'rb') as f: root_cert = f.read()
        credentials = grpc.ssl_channel_credentials(root_certificates=root_cert)
        return grpc.aio.secure_channel(server_address, credentials)
    else:
        return grpc.aio.insecure_channel(server_address)

# --- Core Inference and Health Check Logic ---

async def process_a2f_inference(audio_bytes: bytes, config_bytes: bytes):
    grpc_url = os.getenv("A2F_NIM_URL", "34.67.4.120:52000")
    secure_mode, root_cert, client_cert, client_key = (os.getenv("SECURE_MODE", "disabled").lower(), os.getenv("ROOT_CERT_PATH"),
                                                       os.getenv("CLIENT_CERT_PATH"), os.getenv("CLIENT_KEY_PATH"))
    try:
        channel = create_grpc_channel(grpc_url, secure_mode, root_cert, client_cert, client_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create gRPC channel: {e}")

    async with channel:
        stub = A2FControllerServiceStub(channel)
        stream = stub.ProcessAudioStream()
        try:
            write_task = asyncio.create_task(write_to_stream_api(stream, config_bytes, audio_bytes))
            read_task = asyncio.create_task(read_from_stream_api(stream))
            await write_task
            result_data = await read_task

            status_code_map = {0: "SUCCESS", 1: "INFO", 2: "WARNING", 3: "ERROR"}
            status_info = {"code": "UNKNOWN", "message": ""}
            if result_data.get("final_status"):
                status_info["code"] = status_code_map.get(result_data["final_status"].code, "UNKNOWN")
                status_info["message"] = result_data["final_status"].message
            
            output_wav_bytes = create_wav_in_memory(result_data["audio_header"], result_data["audio_buffer"])
            output_audio_b64 = base64.b64encode(output_wav_bytes).decode('utf-8')
            
            return {"status": status_info, "animation_frames": result_data["animation_key_frames"],
                    "emotions": result_data["emotion_key_frames"], "output_audio_wav_base64": output_audio_b64}
        except (HTTPException, RuntimeError) as e:
            raise e
        except grpc.aio.AioRpcError as e:
            raise HTTPException(status_code=503, detail=f"gRPC service error: {e.details()}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

async def check_health(channel: grpc.aio.Channel):
    stub = HealthStub(channel)
    try:
        return (await stub.Check(HealthCheckRequest())).status == HealthCheckResponse.SERVING
    except grpc.RpcError:
        return False

# --- FastAPI Endpoints ---

@app.get("/", summary="Welcome Endpoint", include_in_schema=False)
async def root():
    return {"message": "Welcome to the A2F Conversational Client API. Use /docs for details."}

@app.post("/inference-from-prompt", summary="Run Full Conversational Turn from a Prompt", tags=["Inference"])
async def run_inference_from_prompt(
    prompt: str = Form(..., description="The user prompt to send to the Gemini model."),
    config_file: UploadFile = File(..., description="A YAML configuration file for A2F processing.")
):
    """
    Generates a response from the Gemini API based on a user prompt, converts that
    response to speech, and then processes it to generate facial animation. This is
    the all-in-one endpoint for a conversational turn.

    **Requires:** `GEMINI_API_KEY` environment variable and `FFmpeg` on the server.
    """
    warnings.filterwarnings("ignore", category=scipy.io.wavfile.WavFileWarning)

    # 1. Get polite, short response from Gemini
    gemini_response_text = await generate_response_from_gemini(prompt)

    # 2. Generate audio from the Gemini response text
    audio_bytes = generate_audio_from_text(gemini_response_text)
    
    # 3. Read the config file
    config_bytes = await config_file.read()

    # 4. Process with A2F pipeline and get result
    inference_result = await process_a2f_inference(audio_bytes, config_bytes)
    
    # 5. Add the spoken text to the response for client-side display
    inference_result["spoken_text"] = gemini_response_text
    
    return inference_result

@app.post("/inference-from-text", summary="Run A2F Inference from Text", tags=["Inference"])
async def run_inference_from_text(
    text: str = Form(..., description="The text to convert to speech and animate."),
    config_file: UploadFile = File(..., description="A YAML configuration file for A2F processing.")
):
    """
    Converts input text to speech and processes it with a config file to generate facial animation.
    **Note:** This endpoint requires `FFmpeg` to be installed on the server.
    """
    warnings.filterwarnings("ignore", category=scipy.io.wavfile.WavFileWarning)
    audio_bytes = generate_audio_from_text(text)
    config_bytes = await config_file.read()
    return await process_a2f_inference(audio_bytes, config_bytes)

@app.post("/inference-from-file", summary="Run A2F Inference from Audio File", tags=["Inference"])
async def run_inference_from_file(
    audio_file: UploadFile = File(..., description="A 16-bit mono PCM WAV audio file."),
    config_file: UploadFile = File(..., description="A YAML configuration file for A2F processing.")
):
    """
    Processes an uploaded audio file and config file to generate facial animation.
    """
    warnings.filterwarnings("ignore", category=scipy.io.wavfile.WavFileWarning)
    audio_bytes = await audio_file.read()
    config_bytes = await config_file.read()
    return await process_a2f_inference(audio_bytes, config_bytes)

@app.get("/health", summary="Check Backend Service Health", tags=["Health"])
async def health_check():
    grpc_url, secure_mode = os.getenv("A2F_NIM_URL", "34.67.4.120:52000"), os.getenv("SECURE_MODE", "disabled").lower()
    root_cert, client_cert, client_key = os.getenv("ROOT_CERT_PATH"), os.getenv("CLIENT_CERT_PATH"), os.getenv("CLIENT_KEY_PATH")
    try:
        channel = create_grpc_channel(grpc_url, secure_mode, root_cert, client_cert, client_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create gRPC channel: {e}")
    is_online = await check_health(channel)
    return {"service_url": grpc_url, "status": "ONLINE" if is_online else "OFFLINE"}

# --- Running the Server ---
if __name__ == "__main__":
    print("Starting FastAPI server for A2F Conversational Client...")
    print("Use 'uvicorn main:app --reload' to run for development.")
    print("API documentation available at http://34.67.4.120:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
