import asyncio
import os
import pyaudio
import sys
import warnings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import (
    AudioTranscriptionConfig,
    AutomaticActivityDetection,
    EndSensitivity,
    LiveConnectConfig,
    RealtimeInputConfig,
    StartSensitivity,
)

# Load environment variables
load_dotenv()

# Suppress DeprecationWarning for session.send
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("PROJECT_LOCATION", "us-central1")
MODEL_ID = "gemini-live-2.5-flash-preview-native-audio-09-2025"

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK = 512

class AudioLoop:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

    def start_streams(self):
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_RATE,
            output=True
        )

    def stop_streams(self):
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()

async def main():
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options={"api_version": "v1beta1"},
    )

    audio_loop = AudioLoop()
    audio_loop.start_streams()

    print(f"Connecting to model: {MODEL_ID}")
    print("Speak now! (Ctrl+C to exit)")

    with open("system_prompt.md", "r") as f:
        system_instruction = f.read()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(parts=[types.Part(text=system_instruction)]),
        proactivity=types.ProactivityConfig(proactive_audio=True),
        input_audio_transcription=types.AudioTranscriptionConfig(), 
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Fenrir",
                )
            ),
        ),
        realtime_input_config=RealtimeInputConfig(
            automatic_activity_detection=AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=200,
                silence_duration_ms=300,
            )
        ),
    )

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            
            async def send_audio():
                while True:
                    try:
                        data = await asyncio.to_thread(
                            audio_loop.input_stream.read, CHUNK, exception_on_overflow=False
                        )
                        # Using session.send as it is the currently working method for audio chunks
                        # despite the deprecation warning (which we suppressed).
                        await session.send(input={"data": data, "mime_type": "audio/pcm"}, end_of_turn=False)
                    except Exception as e:
                        print(f"Error sending audio: {e}")
                        break

            async def receive_audio():
                input_transcriptions = []
                output_transcriptions = []
                
                while True:
                    try:
                        async for response in session.receive():
                            if response.server_content:
                                # Collect input transcription chunks
                                if (
                                    response.server_content.input_transcription
                                    and response.server_content.input_transcription.text
                                ):
                                    input_transcriptions.append(response.server_content.input_transcription.text)
                                
                                # Collect output transcription chunks
                                if (
                                    response.server_content.output_transcription
                                    and response.server_content.output_transcription.text
                                ):
                                    output_transcriptions.append(response.server_content.output_transcription.text)

                                # Play audio output
                                if response.server_content.model_turn:
                                    for part in response.server_content.model_turn.parts:
                                        if part.inline_data:
                                            await asyncio.to_thread(
                                                audio_loop.output_stream.write, part.inline_data.data
                                            )
                                
                                # Print complete transcriptions when turn is complete
                                if response.server_content.turn_complete:
                                    if input_transcriptions:
                                        print(f"User: {''.join(input_transcriptions)}")
                                        input_transcriptions = []
                                    if output_transcriptions:
                                        print(f"Medforce: {''.join(output_transcriptions)}")
                                        output_transcriptions = []
                                        
                    except Exception as e:
                        print(f"Error receiving audio: {e}")
                        break

            await asyncio.gather(send_audio(), receive_audio())

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        audio_loop.stop_streams()
        print("\nExiting...")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
