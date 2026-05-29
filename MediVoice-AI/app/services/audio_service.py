import os
from gtts import gTTS
from google import genai
from google.genai import types


def transcribe_audio(audio_file_path: str) -> str:
    """
    Converts a local audio file into text by passing it directly to Gemini 2.5 Flash.
    Uses Inline Bytes to bypass the File API upload queue for real-time speed.
    """
    # 1. Defensive check for empty files
    if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
        print("Warning: Received empty audio file. Aborting transcription.")
        return ""

    try:
        # 2. Grab the API key from Hugging Face secrets
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        # 3. Read the audio file directly into memory (Inline Data)
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        # 4. Ask Gemini to act as a transcriptionist using the raw bytes
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type='audio/webm',  # Match the browser's recording format
                ),
                "You are an expert medical and health insurance transcriptionist. Transcribe this audio exactly as spoken. Do NOT answer the prompt. ONLY output the raw transcription."
            ]
        )

        return response.text.strip()

    except Exception as e:
        print(f"🚨 STT Transcription Failed via Gemini: {e}")
        return ""


def synthesize_speech(text: str, output_path: str):
    """
    Converts text to speech using the free gTTS library
    and saves the output as an MP3 file.
    """
    # Defensive check so we don't generate blank audio
    if not text or text.strip() == "":
        return

    try:
        # tld='co.in' gives the AI a slight Indian accent,
        # which fits your product context perfectly!
        tts = gTTS(text=text, lang='en', tld='co.in')
        tts.save(output_path)

    except Exception as e:
        print(f"🚨 TTS Generation Failed via gTTS: {e}")