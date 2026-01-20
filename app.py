# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
import time
import logging
import inspect

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Environment variables loaded from .env")
except ImportError:
    logger.warning("python-dotenv not installed. Using system environment variables only.")

# Import prompts
try:
    from prompts import GEMINI_SYSTEM_PROMPT, TYPHOON_SYSTEM_PROMPT
    logger.info("System prompts loaded")
except ImportError:
    logger.warning("prompts.py not found. Using default prompts.")
    GEMINI_SYSTEM_PROMPT = "You are a helpful AI assistant with access to Google Search. Provide accurate, well-researched answers."
    TYPHOON_SYSTEM_PROMPT = "You are a helpful AI assistant. Process and present information clearly. Support both English and Thai."

# Import Google GenAI SDK
try:
    from google import genai
    GENAI_AVAILABLE = True
    logger.info("Google GenAI SDK loaded")
except ImportError:
    GENAI_AVAILABLE = False
    logger.error("Google GenAI SDK not installed. Run: pip install google-genai")

app = Flask(__name__)
CORS(app)

# ===== API CONFIG =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API keys
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables!")

if not TYPHOON_API_KEY:
    logger.error("TYPHOON_API_KEY not found in environment variables!")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found - GROQ backup will not be available")

# Initialize Google GenAI client
gemini_client = None
if GENAI_AVAILABLE and GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Google GenAI client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GenAI client: {e}")

# API URLs
TYPHOON_URL = "https://api.opentyphoon.ai/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ===== FALLBACK RESPONSES =====

def generate_fallback_response(user_message):
    """Generate intelligent fallback response when API is unavailable"""
    message_lower = user_message.lower()

    # Check for greetings
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return "Hello! I'm experiencing some technical difficulties right now. Please try again in a moment."

    if any(word in message_lower for word in ['สวัสดี', 'หวัดดี', 'ดี']):
        return "สวัสดีครับ/ค่ะ! ขณะนี้ระบบกำลังประสบปัญหาชั่วคราว กรุณาลองใหม่อีกครั้งในอีกสักครู่"

    # Check for thanks
    if any(word in message_lower for word in ['thanks', 'thank you']):
        return "You're welcome! Feel free to ask me anything else."

    if any(word in message_lower for word in ['ขอบคุณ', 'ขอบใจ']):
        return "ยินดีครับ/ค่ะ! หากมีคำถามเพิ่มเติม สามารถถามได้เลย"

    # Default fallback
    return """I apologize, but I'm currently experiencing technical difficulties.

This could be due to:
- API rate limits being reached
- Network connectivity issues
- Temporary service unavailability

Please try again in a few moments. If the issue persists, please check back later."""


# ===== API FUNCTIONS =====

def call_gemini(user_message, history=None, max_retries=2):
    """
    Call Google Gemini API using the genai SDK

    Args:
        user_message: User's question
        history: List of previous conversation messages (last 5 conversations)
        max_retries: Number of retry attempts

    Returns:
        dict or None: Response with answer, or None if failed
    """
    if not gemini_client:
        logger.error("Gemini client not initialized")
        return None

    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Gemini API (attempt {attempt + 1}/{max_retries})...")

            # Build conversation context from history
            history_context = ""
            if history and len(history) > 0:
                history_context = "\n\nPrevious conversation:\n"
                for msg in history:
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    history_context += f"{role}: {msg.get('content', '')}\n"
                history_context += "\n"

            # Build the prompt with system instruction and history
            full_prompt = f"{GEMINI_SYSTEM_PROMPT}{history_context}\n\nUser Question: {user_message}"

            # Call Gemini using the SDK
            response = gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=full_prompt
            )

            answer = response.text

            # Handle empty/None response
            if not answer:
                logger.warning("Gemini returned empty response")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

            logger.info(f"Gemini Response: {len(answer)} characters")

            return {
                "answer": answer
            }

        except Exception as e:
            error_str = str(e)
            logger.error(f"Gemini Error: {error_str}")

            # Check for rate limit
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limited. Retrying in 30s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(30)
                    continue

            if attempt == max_retries - 1:
                return None
            time.sleep(5)

    return None


def call_groq(user_message, history=None, max_retries=2):
    """
    Call GROQ API as backup for Gemini

    Args:
        user_message: User's question
        history: List of previous conversation messages (last 5 conversations)
        max_retries: Number of retry attempts

    Returns:
        dict or None: Response with answer, or None if failed
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not configured")
        return None

    for attempt in range(max_retries):
        try:
            logger.info(f"Calling GROQ API (attempt {attempt + 1}/{max_retries})...")

            # Build messages array with history
            messages = [
                {
                    "role": "system",
                    "content": GEMINI_SYSTEM_PROMPT
                }
            ]

            # Add conversation history
            if history and len(history) > 0:
                for msg in history:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })

            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            }

            response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

            # Handle rate limit
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    logger.warning("GROQ rate limited. Retrying in 10s...")
                    time.sleep(10)
                    continue
                else:
                    logger.error("GROQ API rate limited after all retries")
                    return None

            if response.status_code != 200:
                logger.error(f"GROQ API error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return None

            result = response.json()
            answer = result["choices"][0]["message"]["content"]

            logger.info(f"GROQ Response: {len(answer)} characters")

            return {
                "answer": answer
            }

        except requests.exceptions.Timeout:
            logger.error(f"GROQ timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            return None

        except Exception as e:
            logger.error(f"GROQ Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            return None

    return None


def call_typhoon(gemini_response, user_message, max_retries=2):
    """
    Process Gemini response through Typhoon for final answer generation

    Args:
        gemini_response: Response from Gemini API (dict or string)
        user_message: Original user message
        max_retries: Number of retry attempts

    Returns:
        str: Processed response from Typhoon, or original Gemini response if failed
    """
    if not TYPHOON_API_KEY:
        logger.error("TYPHOON_API_KEY not configured")
        # Return Gemini response as fallback
        if isinstance(gemini_response, dict):
            return gemini_response.get("answer", str(gemini_response))
        return str(gemini_response)

    # Extract context from Gemini response
    if isinstance(gemini_response, dict):
        context = gemini_response.get("answer", "")
    else:
        context = str(gemini_response)

    for attempt in range(max_retries):
        try:
            payload = {
                "model": "typhoon-v2.5-30b-a3b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": TYPHOON_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Based on the following information from Google Gemini:

{context}

User's original question: {user_message}

Please process, refine, and enhance this answer. Provide a comprehensive, well-structured response. If the user writes in Thai, respond in Thai. If in English, respond in English."""
                    }
                ],
                "temperature": 0.5,
                "max_tokens": 2048
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TYPHOON_API_KEY}"
            }

            logger.info(f"Calling Typhoon API (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(TYPHOON_URL, json=payload, headers=headers, timeout=30)

            # Handle rate limit
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    logger.warning("Typhoon rate limited. Retrying in 5s...")
                    time.sleep(5)
                    continue
                else:
                    logger.error("Typhoon API rate limited. Returning Gemini response.")
                    return context

            if response.status_code != 200:
                logger.error(f"Typhoon API error {response.status_code}: {response.text}")
                return context

            result = response.json()
            typhoon_answer = result["choices"][0]["message"]["content"]

            logger.info(f"Typhoon Response: {len(typhoon_answer)} characters")
            return typhoon_answer

        except requests.exceptions.Timeout:
            logger.error(f"Typhoon timeout (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return context
            time.sleep(3)

        except Exception as e:
            logger.error(f"Typhoon Error: {str(e)}")
            return context

    return context


# ===== ROUTES =====

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint with two-step API flow:
    1. Google Gemini for search/information gathering
    2. Typhoon for processing and final answer generation
    """
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        history = data.get('history', [])  # Get conversation history (last 5 conversations)

        if not user_message:
            return jsonify({
                'error': 'Message is required',
                'status': 'error'
            }), 400

        logger.info(f"\n{'='*60}")
        logger.info(f"New Message: {user_message}")
        logger.info(f"History: {len(history)} messages")
        logger.info(f"{'='*60}")

        # Step 1: Call Google Gemini API for search/information gathering
        gemini_response = call_gemini(user_message, history=history)

        # Use GROQ as backup if Gemini fails
        used_groq_backup = False
        if gemini_response is None:
            logger.warning("Gemini failed, trying GROQ backup...")
            gemini_response = call_groq(user_message, history=history)
            used_groq_backup = True

        # Use fallback if both Gemini and GROQ fail
        if gemini_response is None:
            logger.warning("Both Gemini and GROQ failed. Using fallback response...")
            return jsonify({
                'reply': generate_fallback_response(user_message),
                'gemini_raw': None,
                'status': 'fallback',
                'is_fallback': True
            })

        # Step 2: Process with Typhoon
        final_response = call_typhoon(gemini_response, user_message)

        logger.info(f"{'='*60}")
        logger.info("Response complete!")
        logger.info(f"{'='*60}\n")

        return jsonify({
            'reply': final_response,
            'gemini_raw': gemini_response.get("answer", "") if isinstance(gemini_response, dict) else str(gemini_response),
            'status': 'success',
            'is_fallback': False,
            'used_groq_backup': used_groq_backup
        })

    except Exception as e:
        logger.error(f"Error in /api/chat: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/health')
def health():
    """Health check endpoint with detailed API status"""
    return jsonify({
        'status': 'healthy',
        'service': 'MyAI Assist',
        'gemini_configured': bool(gemini_client),
        'typhoon_configured': bool(TYPHOON_API_KEY),
        'groq_configured': bool(GROQ_API_KEY),
        'apis': {
            'gemini': {
                'configured': bool(gemini_client),
                'name': 'Gemini',
                'status': 'online' if gemini_client else 'offline'
            },
            'typhoon': {
                'configured': bool(TYPHOON_API_KEY),
                'name': 'Typhoon',
                'status': 'online' if TYPHOON_API_KEY else 'offline'
            },
            'groq': {
                'configured': bool(GROQ_API_KEY),
                'name': 'GROQ',
                'status': 'online' if GROQ_API_KEY else 'offline'
            }
        }
    })


@app.route('/api/source/<func_name>')
def get_source(func_name):
    """
    Get source code of a function using inspect.getsource()

    Usage: /api/source/call_gemini
    """
    # Map of available functions to inspect
    available_functions = {
        'call_gemini': call_gemini,
        'call_groq': call_groq,
        'call_typhoon': call_typhoon,
        'generate_fallback_response': generate_fallback_response,
        'generate_suggestions_with_groq': generate_suggestions_with_groq,
        'chat': chat,
        'health': health,
        'index': index
    }

    if func_name not in available_functions:
        return jsonify({
            'error': f'Function "{func_name}" not found',
            'available': list(available_functions.keys())
        }), 404

    try:
        func = available_functions[func_name]
        source_code = inspect.getsource(func)
        return jsonify({
            'function': func_name,
            'source': source_code,
            'file': inspect.getfile(func),
            'lines': len(source_code.splitlines())
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/sources')
def list_sources():
    """List all available functions with their source code"""
    functions = {
        'call_gemini': call_gemini,
        'call_groq': call_groq,
        'call_typhoon': call_typhoon,
        'generate_fallback_response': generate_fallback_response,
        'generate_suggestions_with_groq': generate_suggestions_with_groq
    }

    sources = {}
    for name, func in functions.items():
        try:
            sources[name] = {
                'source': inspect.getsource(func),
                'doc': inspect.getdoc(func),
                'lines': len(inspect.getsource(func).splitlines())
            }
        except Exception as e:
            sources[name] = {'error': str(e)}

    return jsonify(sources)


def get_default_suggestions():
    """Return default suggestions when AI generation fails"""
    return [
        {"question": "What are the latest breakthroughs in AI?", "description": "Recent AI developments", "category": "Technology"},
        {"question": "อธิบายเรื่อง Quantum Computing แบบง่ายๆ", "description": "เรียนรู้เทคโนโลยีใหม่", "category": "Science"},
        {"question": "What's trending in the world today?", "description": "Current global events", "category": "News"},
        {"question": "สอนทำอาหารไทยง่ายๆ สักจาน", "description": "สูตรอาหารไทย", "category": "Culture"}
    ]


def generate_suggestions_with_groq():
    """Generate suggestions using GROQ API"""
    if not GROQ_API_KEY:
        return None

    try:
        prompt = """Generate 4 interesting and diverse questions or topics that would be fun to ask an AI assistant today.

Include a mix of:
- Current events or trending topics
- Science or technology
- Thai culture, food, or language (write in Thai)
- Fun facts, trivia, or creative topics

Return ONLY a valid JSON array with this exact format (no markdown, no explanation):
[
  {"question": "your question here", "description": "5 word description", "category": "Category"},
  {"question": "คำถามภาษาไทย", "description": "คำอธิบายสั้นๆ", "category": "Culture"}
]

Make questions engaging, varied, and relevant to today. Include at least 1-2 Thai language questions."""

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.9,
            "max_tokens": 1024
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=15)

        if response.status_code != 200:
            logger.error(f"GROQ suggestions error {response.status_code}: {response.text}")
            return None

        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()

        # Parse JSON from response
        import json
        import re

        # Extract JSON from response (handle markdown code blocks)
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\n?', '', text)
            text = re.sub(r'\n?```$', '', text)

        suggestions = json.loads(text)

        if isinstance(suggestions, list) and len(suggestions) > 0:
            return suggestions[:4]

        return None

    except Exception as e:
        logger.error(f"GROQ suggestions error: {str(e)}")
        return None


@app.route('/api/suggestions')
def get_suggestions():
    """Generate random interesting questions/topics of the day using GROQ (primary) or Gemini (backup)"""

    # Try GROQ first (faster and dedicated for suggestions)
    suggestions = generate_suggestions_with_groq()
    if suggestions:
        return jsonify({'suggestions': suggestions, 'status': 'success', 'source': 'groq'})

    # Fallback to Gemini if GROQ fails
    if gemini_client:
        try:
            prompt = """Generate 4 interesting and diverse questions or topics that would be fun to ask an AI assistant today.

Include a mix of:
- Current events or trending topics
- Science or technology
- Thai culture, food, or language (write in Thai)
- Fun facts, trivia, or creative topics

Return ONLY a valid JSON array with this exact format (no markdown, no explanation):
[
  {"question": "your question here", "description": "5 word description", "category": "Category"},
  {"question": "คำถามภาษาไทย", "description": "คำอธิบายสั้นๆ", "category": "Culture"}
]

Make questions engaging, varied, and relevant to today. Include at least 1-2 Thai language questions."""

            response = gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

            # Parse JSON from response
            import json
            import re

            # Extract JSON from response (handle markdown code blocks)
            text = response.text.strip()
            if text.startswith('```'):
                text = re.sub(r'^```(?:json)?\n?', '', text)
                text = re.sub(r'\n?```$', '', text)

            suggestions = json.loads(text)

            # Validate structure
            if isinstance(suggestions, list) and len(suggestions) > 0:
                return jsonify({'suggestions': suggestions[:4], 'status': 'success', 'source': 'gemini'})

        except Exception as e:
            logger.error(f"Error generating suggestions with Gemini: {str(e)}")

    # Final fallback to default suggestions
    return jsonify({'suggestions': get_default_suggestions(), 'status': 'fallback', 'source': 'default'})


if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("Starting MyAI Assist Server")
    logger.info("="*60)
    logger.info(f"Gemini: {'Configured' if gemini_client else 'Not configured'}")
    logger.info(f"Typhoon: {'Configured' if TYPHOON_API_KEY else 'Not configured'}")
    logger.info(f"GROQ: {'Configured' if GROQ_API_KEY else 'Not configured'}")
    logger.info("="*60)
    logger.info("Server: http://localhost:5000")
    logger.info("API: http://localhost:5000/api/chat")
    logger.info("Health: http://localhost:5000/health")
    logger.info("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
