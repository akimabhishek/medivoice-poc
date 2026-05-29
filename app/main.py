import os
import shutil
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request, Form, BackgroundTasks, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Import all our specialized services
from app.services.ai_service import process_medical_intent_with_metrics, rewrite_transcript, search_policy
from app.services.audio_service import transcribe_audio, synthesize_speech
from app.services.rag_service import ingest_policies, append_pdf_to_database


# Run this function every time the server boots up to sync the knowledge base
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Booting up RAG Engine...")
    ingest_policies()
    yield


# Simple in-memory storage for the conversation thread
CONVERSATION_HISTORY = []

# Initialize FastAPI with the lifespan manager
app = FastAPI(title="MediVoice AI API", lifespan=lifespan)

# Mount the static directory so the frontend can access the generated audio files
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Ensure the policy upload directory exists
os.makedirs("app/data/policies", exist_ok=True)

# Mount the templates directory for the HTML UI
templates = Jinja2Templates(directory="app/templates")


# ---------------------------------------------------------
# Application Routes
# ---------------------------------------------------------

@app.get("/")
async def get_ui(request: Request):
    """Serves the interactive Product Manager UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/status")
async def get_system_status():
    """Returns the current state of the vector database on UI boot."""
    try:
        # Check how many PDFs are actually in the folder
        files = [f for f in os.listdir("app/data/policies") if f.lower().endswith('.pdf')]

        if len(files) == 1:
            policy_text = files[0]
        elif len(files) > 1:
            policy_text = f"{len(files)} Policies Loaded Active"
        else:
            policy_text = "No Policy Loaded"

        return {"active_policy": policy_text}
    except Exception:
        return {"active_policy": "No Policy Loaded"}


@app.post("/api/query")
async def process_query(text_query: str = Form(...)):
    """Handles core conversational logic with conversational memory for TEXT inputs."""
    start_time = time.time()
    try:
        clean_transcript = text_query.strip()

        if not clean_transcript:
            raise HTTPException(status_code=400, detail="Query text cannot be empty.")

        # Search the Vector Database
        print(f"Searching Vector Database for: '{clean_transcript}'...")
        retrieved_context, rag_score = search_policy(clean_transcript)

        # Reason with Gemini passing the global conversation memory
        print("Reasoning with Gemini...")
        ai_result = process_medical_intent_with_metrics(
            clean_transcript=clean_transcript,
            retrieved_context=retrieved_context,
            rag_score=rag_score,
            history=CONVERSATION_HISTORY
        )

        # Extract the two distinct outputs
        reasoning = ai_result.get("reasoning_chain", "No reasoning provided.")
        script = ai_result.get("voice_script", "No script provided.")

        # Save ONLY the conversational script to memory (so it remembers what it "said")
        CONVERSATION_HISTORY.append({
            "user": clean_transcript,
            "bot": script
        })

        # --- FIX: Synthesize Audio for Text Queries Too! ---
        print("Synthesizing AI Voice for Text Input...")
        timestamp = int(time.time())
        filename = f"response_{timestamp}.mp3"
        output_filepath = f"app/static/{filename}"

        # Call the synthesizer with BOTH the script and the save path
        synthesize_speech(script, output_filepath)
        audio_url = f"/static/{filename}"
        # ---------------------------------------------------

        # Extract perfectly accurate tokens from Gemini
        input_tokens = ai_result.get("usage_metadata", {}).get("prompt_tokens", 0)
        output_tokens = ai_result.get("usage_metadata", {}).get("completion_tokens", 0)

        # Calculate cost using the real data
        api_cost = (input_tokens * 0.000000075) + (output_tokens * 0.0000003)
        latency = round(time.time() - start_time, 2)

        return {
            "matched_intent": "query_policy_details",
            "pre_auth_flag": "NO",
            "ai_reasoning_chain": reasoning,
            "generated_voice_script": script,
            "audio_url": audio_url, # Send the audio URL to the frontend
            "metrics": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "api_cost": f"${api_cost:.6f}",
                "rag_confidence": round(rag_score, 2),
                "latency": f"{latency}s"
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error in /api/query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/api/voice")
async def process_voice_query(audio_file: UploadFile = File(...)):
    """Handles end-to-end VOICE interactions (STT -> RAG -> Gemini -> TTS)."""
    start_time = time.time()
    temp_path = f"app/data/{audio_file.filename}"

    try:
        # 1. Save the incoming audio blob from the microphone temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # 2. Transcribe the audio to text
        print("Transcribing microphone audio...")
        raw_transcript = transcribe_audio(temp_path)

        # Clean up the temporary file immediately
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # 3. Clean up the transcript for medical accuracy
        print("Scrubbing transcript for medical terms...")
        clean_transcript = rewrite_transcript(raw_transcript)
        if not clean_transcript or clean_transcript.strip() == "":
            raise HTTPException(status_code=400, detail="No speech detected. Please try again.")

        # 4. Search the Vector Database
        print(f"Searching Vector Database for: '{clean_transcript}'...")
        retrieved_context, rag_score = search_policy(clean_transcript)

        # 5. Reason with Gemini passing the global conversation memory
        print("Reasoning with Gemini...")
        ai_result = process_medical_intent_with_metrics(
            clean_transcript=clean_transcript,
            retrieved_context=retrieved_context,
            rag_score=rag_score,
            history=CONVERSATION_HISTORY
        )

        # Extract the two distinct outputs
        reasoning = ai_result.get("reasoning_chain", "No reasoning provided.")
        script = ai_result.get("voice_script", "No script provided.")

        # Save ONLY the conversational script to memory
        CONVERSATION_HISTORY.append({
            "user": clean_transcript,
            "bot": script
        })

        # 6. Generate the AI Voice Response
        print("Synthesizing AI Voice...")

        # Create a unique filename using a timestamp to prevent browser caching
        timestamp = int(time.time())
        filename = f"response_{timestamp}.mp3"
        output_filepath = f"app/static/{filename}"

        # Call the synthesizer with BOTH the script and the save path
        synthesize_speech(script, output_filepath)

        # Tell the frontend exactly where to find the new audio file
        audio_url = f"/static/{filename}"

        # Extract perfectly accurate tokens from Gemini
        input_tokens = ai_result.get("usage_metadata", {}).get("prompt_tokens", 0)
        output_tokens = ai_result.get("usage_metadata", {}).get("completion_tokens", 0)

        # Calculate cost using the real data
        api_cost = (input_tokens * 0.000000075) + (output_tokens * 0.0000003)
        latency = round(time.time() - start_time, 2)

        return {
            "transcript": clean_transcript,
            "matched_intent": "query_policy_details",
            "pre_auth_flag": "NO",
            "ai_reasoning_chain": reasoning,
            "generated_voice_script": script,
            "audio_url": audio_url,
            "metrics": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "api_cost": f"${api_cost:.6f}",
                "rag_confidence": round(rag_score, 2),
                "latency": f"{latency}s"
            }
        }

    except Exception as e:
        print(f"❌ Error in /api/voice: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a PDF from the UI and processes it in the background."""
    save_path = f"app/data/policies/{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Tell FastAPI to run the heavy Markdown extraction in the background
    background_tasks.add_task(append_pdf_to_database, save_path, file.filename)

    return {"message": "File uploaded successfully. AI is learning the document in the background!"}