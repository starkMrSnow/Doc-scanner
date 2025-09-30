import time
import tempfile
import json
import re
import os
from unstract.llmwhisperer import LLMWhispererClientV2
from google import genai

# init whisperer client (use env vars for production)
client = LLMWhispererClientV2(
    base_url="https://llmwhisperer-api.us-central.unstract.com/api/v2",
    api_key=os.getenv("LLMWHISPERER_API_KEY", "RnkEXMIG6sjFDv-5suZKdPHQzMozYH-IpINhYpv773M")
)

client_genai = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDI02MGCVXW3d1iLPwuw_S_2B4BBoTBOow")
)


def whisper_extract(pdf_bytes: bytes) -> str:
    """
    Send PDF to LLM Whisperer, wait until it's processed,
    and return the extracted text.
    
    Optimizations:
    - Adaptive polling (starts fast, then slows down)
    - Better error handling
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        result = client.whisper(file_path=tmp_path)
        whisper_hash = result['whisper_hash']
        
        # Adaptive polling: start with 1s, increase to 3s max
        poll_interval = 1
        max_attempts = 60  # 60 attempts max (about 2 minutes)
        
        for attempt in range(max_attempts):
            status = client.whisper_status(whisper_hash=whisper_hash)
            
            if status['status'] == 'processed':
                resultx = client.whisper_retrieve(whisper_hash=whisper_hash)
                return resultx['extraction']['result_text']
            
            elif status['status'] == 'failed':
                raise RuntimeError(f"Whisperer failed: {status.get('message', 'Unknown error')}")
            
            # Adaptive wait: 1s -> 2s -> 3s (max)
            time.sleep(poll_interval)
            poll_interval = min(poll_interval + 0.5, 3)
        
        raise TimeoutError("PDF processing timed out after 2 minutes")
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def clean_json_response(text: str) -> str:
    """
    Extract JSON from Gemini's response, removing markdown code blocks.
    
    Handles:
    - ```json ... ```
    - ``` ... ```
    - Raw JSON
    """
    # Remove markdown code blocks
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'^```\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    
    return text.strip()


def extract_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Step 1: Extract raw text with Whisperer
    Step 2: Use Gemini to structure it as JSON
    
    Optimizations:
    - Better prompt for JSON output
    - Robust JSON parsing
    - Structured error responses
    """
    try:
        # Extract text
        text = whisper_extract(pdf_bytes)
        
        # Improved prompt for cleaner JSON output
        prompt = f"""Extract information from this document and return ONLY a JSON object (no markdown, no explanation).

Required fields:
- invoice_number: string (or null if not found)
- date: string in format "DD MMM YYYY" (or null)
- vendor: string (or null)
- total_amount: string (or null)
- currency: string (3-letter code like USD, EUR, or null)

Document text:
{text}

Return ONLY the JSON object, nothing else:"""

        # Call Gemini
        response = client_genai.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        # Clean and parse response
        cleaned_text = clean_json_response(response.text)
        
        try:
            extracted = json.loads(cleaned_text)
            return {
                "success": True,
                "data": extracted,
                "raw_text_preview": text[:200]  # First 200 chars for reference
            }
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return structured error
            return {
                "success": False,
                "error": "Failed to parse JSON from Gemini",
                "gemini_response": cleaned_text,
                "parse_error": str(e),
                "raw_text_preview": text[:200]
            }
    
    except TimeoutError as e:
        return {
            "success": False,
            "error": "PDF processing timeout",
            "details": str(e)
        }
    
    except RuntimeError as e:
        return {
            "success": False,
            "error": "PDF processing failed",
            "details": str(e)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": "Unexpected error",
            "details": str(e),
            "type": type(e).__name__
        }