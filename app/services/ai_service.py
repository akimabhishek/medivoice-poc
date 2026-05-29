import json
import time
import chromadb
from google import genai
from google.genai import types

# Initialize client using your validated Project Number and location
client = genai.Client(
    vertexai=True,
    project="277169065369",
    location="us-central1"
)


def search_policy(query_text: str):
    """
    Searches the local ChromaDB vector store for relevant policy chunks.
    Pulls the Top 20 chunks to ensure dense tables and legal texts are caught.
    """
    chroma_client = chromadb.PersistentClient(path="app/data/chroma_db")

    # --- FIX 1: Use get_or_create to prevent crash on an empty database ---
    collection = chroma_client.get_or_create_collection(name="policy_docs")

    # If the database is completely empty, short-circuit gracefully
    if collection.count() == 0:
        return "No policy documents have been uploaded to the system yet.", 0.0

    # Ensure we don't ask for more results than chunks actually exist
    num_results = min(20, collection.count())

    results = collection.query(
        query_texts=[query_text],
        n_results=num_results
    )

    if results['documents'] and len(results['documents'][0]) > 0:
        # Combine the chunks into a single large context block for Gemini
        combined_context = "\n\n--- NEXT CHUNK ---\n\n".join(results['documents'][0])

        # Convert ChromaDB's raw distance metric into a 0-1 Confidence Score
        raw_distance = results['distances'][0][0] if 'distances' in results and results['distances'][0] else 0.5
        import math
        # Convert L2 distance to a clean 0-1 confidence score
        similarity_score = 1.0 / (1.0 + raw_distance)

        return combined_context, similarity_score
    else:
        return "No relevant policy documents found.", 0.0


def rewrite_transcript(raw_transcript: str) -> str:
    """Uses Gemini to clean up STT errors before searching the database."""
    print(f"Original Transcript: {raw_transcript}")

    prompt = f"""
    You are an AI transcription editor specializing in Indian health insurance (HDFC Ergo, Star Health, etc.).
    A user spoke into a microphone and the Speech-to-Text engine made errors. 
    Fix the obvious medical and insurance typos in this transcript. 
    Do NOT answer the question. ONLY output the corrected transcript.

    Raw Transcript: "{raw_transcript}"
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
        )
    )

    corrected_transcript = response.text.strip()
    print(f"Corrected Transcript: {corrected_transcript}")

    return corrected_transcript


def process_medical_intent_with_metrics(clean_transcript: str, retrieved_context: str, rag_score: float, history: list = None):
    """Processes the user query using Gemini and outputs Structured JSON."""
    if history is None:
        history = []

    # 1. Force Gemini to return JSON with two distinct personalities
    system_instruction = f"""
    You are MediVoice AI, an expert health insurance assistant. 
    Analyze the user's query using the provided Policy Document Context.

    CRITICAL POLICY CONTEXT:
    {retrieved_context}

    Strict Rules:
    1. Answer based ONLY on the context. If not found, say so clearly.
    2. You MUST return your response as a valid JSON object. Do NOT wrap it in markdown block quotes.

    Required JSON Schema:
    {{
        "reasoning_chain": "Your internal technical monologue. Quote specific section numbers, clauses, and explain exactly why the policy applies or doesn't.",
        "voice_script": "A single, conversational, human-sounding sentence that directly answers the user. This will be read out loud via Text-to-Speech."
    }}
    """

    formatted_contents = []
    for turn in history:
        formatted_contents.append({"role": "user", "parts": [{"text": turn["user"]}]})
        formatted_contents.append({"role": "model", "parts": [{"text": turn["bot"]}]})

    formatted_contents.append({"role": "user", "parts": [{"text": clean_transcript}]})

    # 2. Call the Gemini model asking for JSON
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=formatted_contents,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.2,
            "response_mime_type": "application/json", # Forces strict JSON output
        }
    )

    # 3. Parse the JSON and attach the exact Gemini token metrics
    try:
        parsed_json = json.loads(response.text)

        # Pull the exact token counts straight from Google's servers
        parsed_json["usage_metadata"] = {
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count
        }
        return parsed_json

    except Exception as e:
        print(f"Failed to parse Gemini JSON: {e}")
        return {
            "reasoning_chain": response.text,
            "voice_script": "I experienced an error formatting my response.",
            "usage_metadata": {"prompt_tokens": 0, "completion_tokens": 0}
        }