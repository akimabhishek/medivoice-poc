import os
from google.cloud import speech
from google.cloud import texttospeech


def transcribe_audio(audio_file_path: str) -> str:
    """
    Converts a local audio file into text using Google Speech-to-Text.
    Includes defensive checks, Mono audio enforcement, and custom Speech Contexts.
    """
    client = speech.SpeechClient()

    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    # --- DEFENSIVE PROGRAMMING ---
    # If the user clicked too fast and the file is 0 bytes, abort gracefully.
    if not content:
        print("Warning: Received empty audio file. Aborting transcription.")
        return ""

    audio = speech.RecognitionAudio(content=content)

    # --- SPEECH CONTEXTS (CHEAT SHEET) ---
    # Boost highly specific medical and brand vocabulary
    insurance_jargon = speech.SpeechContext(
        phrases=[
            "HDFC Optima Restore",
            "Optima Restore",
            "Star Health",
            "co-payment",
            "copay",
            "co-payment percentage",
            "pre-authorization"
        ],
        boost=20.0
    )

    # --- CODEC & HARDWARE CONFIG ---
    # Lock to Mono (1 channel), Windows default 44100Hz, and Opus codec
    # --- CODEC & HARDWARE CONFIG ---
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        # sample_rate_hertz=44100,  <--- DELETE OR COMMENT OUT THIS LINE
        audio_channel_count=1,
        language_code="en-US",
        model="default",
        speech_contexts=[insurance_jargon]
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript


def synthesize_speech(text: str, output_path: str):
    """
    Converts text to speech using Google Cloud Text-to-Speech
    and saves the output as an MP3 file.
    """
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Using a standard English voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-F"
    )

    # Outputting to MP3
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_path, "wb") as out:
        out.write(response.audio_content)