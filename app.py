# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests
import os
import time
import logging
import inspect
import json
import uuid
import random
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(_env_path)
    logger.info("Environment variables loaded from .env")
except ImportError:
    logger.warning("python-dotenv not installed. Using system environment variables only.")

# Import prompts
try:
    from prompts import (
        GEMINI_SYSTEM_PROMPT,
        TYPHOON_SYSTEM_PROMPT,
        VOCABULARY_EXERCISE_PROMPT,
        GRAMMAR_EXERCISE_PROMPT,
        TRANSLATION_EXERCISE_PROMPT
    )
    logger.info("System prompts loaded")
except ImportError:
    logger.warning("prompts.py not found. Using default prompts.")
    GEMINI_SYSTEM_PROMPT = "You are a helpful AI assistant with access to Google Search. Provide accurate, well-researched answers."
    TYPHOON_SYSTEM_PROMPT = "You are a helpful AI assistant. Process and present information clearly. Support both English and Thai."
    VOCABULARY_EXERCISE_PROMPT = ""
    GRAMMAR_EXERCISE_PROMPT = ""
    TRANSLATION_EXERCISE_PROMPT = ""

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


def stream_gemini_with_thinking(user_message, history=None):
    """
    Stream Gemini response with thinking mode (Chain of Thought)

    Args:
        user_message: User's question
        history: List of previous conversation messages

    Yields:
        tuples of (event_type, content) where event_type is:
        - 'status': Status updates
        - 'thinking': AI reasoning/thinking content
        - 'answer': Final answer content
        - 'complete': Completion signal
        - 'error': Error message
    """
    if not gemini_client:
        yield ('error', {'message': 'Gemini client not initialized'})
        return

    # Build conversation context from history
    history_context = ""
    if history and len(history) > 0:
        history_context = "\n\nPrevious conversation:\n"
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_context += f"{role}: {msg.get('content', '')}\n"
        history_context += "\n"

    full_prompt = f"{GEMINI_SYSTEM_PROMPT}{history_context}\n\nUser Question: {user_message}"

    try:
        # Signal start
        yield ('status', {'phase': 'thinking', 'message': 'กำลังวิเคราะห์คำถาม...'})

        accumulated_thought = ""
        accumulated_answer = ""

        # Try streaming with thinking mode
        try:
            from google.genai import types

            # Configure thinking mode
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=8192
                )
            )

            for chunk in gemini_client.models.generate_content_stream(
                model="gemini-3-flash-preview",
                contents=full_prompt,
                config=config
            ):
                if chunk.candidates and chunk.candidates[0].content:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            # This is thinking/reasoning content
                            if part.text:
                                accumulated_thought += part.text
                                yield ('thinking', {
                                    'text': part.text,
                                    'accumulated': accumulated_thought
                                })
                        elif part.text:
                            # This is the actual answer
                            accumulated_answer += part.text
                            yield ('answer', {
                                'text': part.text,
                                'accumulated': accumulated_answer
                            })

        except Exception as thinking_error:
            # Fallback: Stream without thinking mode if thinking mode fails
            logger.warning(f"Thinking mode failed, using standard streaming: {thinking_error}")

            yield ('status', {'phase': 'answering', 'message': 'กำลังตอบคำถาม...'})

            for chunk in gemini_client.models.generate_content_stream(
                model="gemini-3-flash-preview",
                contents=full_prompt
            ):
                if chunk.text:
                    accumulated_answer += chunk.text
                    yield ('answer', {
                        'text': chunk.text,
                        'accumulated': accumulated_answer
                    })

        # Signal completion
        yield ('complete', {
            'thinking': accumulated_thought,
            'answer': accumulated_answer
        })

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield ('error', {'message': str(e)})


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


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming chat endpoint with Chain of Thought visualization
    Uses Server-Sent Events (SSE) to stream responses in real-time
    """
    data = request.json
    user_message = data.get('message', '').strip()
    history = data.get('history', [])

    if not user_message:
        return jsonify({'error': 'Message is required', 'status': 'error'}), 400

    logger.info(f"\n{'='*60}")
    logger.info(f"[STREAM] New Message: {user_message}")
    logger.info(f"[STREAM] History: {len(history)} messages")
    logger.info(f"{'='*60}")

    def generate():
        try:
            for event_type, content in stream_gemini_with_thinking(user_message, history):
                # SSE format: "event: type\ndata: content\n\n"
                yield f"event: {event_type}\ndata: {json.dumps(content, ensure_ascii=False)}\n\n"

            # Signal completion
            yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )


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


# ===== INVESTOR INSIGHTS =====

# Cache for investor insights (simple in-memory cache)
investor_insights_cache = {
    'content': None,
    'generated_at': None,
    'date': None,
    'twitter_data': None
}

# List of Twitter accounts to follow for investor insights
INVESTOR_TWITTER_ACCOUNTS = [
    "@setthailand",      # SET Thailand official
    "@stockthailand",    # Thai stock news
    "@efinancethai",     # E-Finance Thai
    "@MoneyChannelTV",   # Money Channel
    "@taborrowสามัคคี",   # Thai investor community
    "@Bloomberg",        # Bloomberg
    "@Reuters",          # Reuters
    "@WSJ",              # Wall Street Journal
    "@ABORROWCNBC",            # CNBC
    "@FinancialTimes",   # Financial Times
]

def fetch_twitter_data_via_gemini():
    """
    Use Gemini with Google Search to fetch latest tweets from investor accounts
    """
    if not gemini_client:
        return None

    try:
        from google.genai import types

        # Create search prompt to get Twitter data
        search_prompt = f"""ค้นหาข้อมูลล่าสุดจาก Twitter/X เกี่ยวกับการลงทุนและตลาดหุ้นวันนี้

ค้นหาจากบัญชี Twitter ที่สำคัญ เช่น:
- SET Thailand, ตลาดหลักทรัพย์แห่งประเทศไทย
- นักวิเคราะห์หุ้นไทย, กูรูหุ้น
- Bloomberg, Reuters, CNBC
- ข่าวเศรษฐกิจและการเงิน

สิ่งที่ต้องการ:
1. ข่าวสำคัญที่ถูกพูดถึงใน Twitter วันนี้
2. หุ้นที่ถูก mention บ่อย
3. ความเห็นของนักวิเคราะห์
4. Sentiment โดยรวมของตลาด (bullish/bearish)
5. ข่าวที่อาจกระทบตลาดหุ้นไทย

ค้นหาข้อมูลล่าสุดจาก Twitter และ X.com"""

        # Use Gemini with Google Search tool
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=search_prompt,
            config=config
        )

        return response.text

    except Exception as e:
        logger.error(f"Error fetching Twitter data via Gemini: {str(e)}")
        return None


def generate_investor_insights():
    """Generate daily investor insights using Gemini API with Twitter data"""
    from datetime import datetime

    today = datetime.now().strftime('%Y-%m-%d')

    # Check cache - return cached if same day
    if investor_insights_cache['date'] == today and investor_insights_cache['content']:
        return investor_insights_cache['content'], investor_insights_cache['generated_at']

    # Step 1: Fetch Twitter data via Gemini Search
    logger.info("Fetching Twitter data for investor insights...")
    twitter_data = fetch_twitter_data_via_gemini()

    if twitter_data:
        logger.info(f"Twitter data fetched: {len(twitter_data)} characters")
        investor_insights_cache['twitter_data'] = twitter_data
    else:
        logger.warning("Could not fetch Twitter data, using general analysis")
        twitter_data = "ไม่สามารถดึงข้อมูลจาก Twitter ได้ กรุณาวิเคราะห์จากข้อมูลทั่วไป"

    # Step 2: Generate analysis based on Twitter data
    prompt = f"""คุณเป็นนักวิเคราะห์การเงินมืออาชีพ กรุณาเขียนบทความวิเคราะห์สำหรับนักลงทุนในวันนี้

## ข้อมูลจาก Twitter/X ที่รวบรวมได้:
{twitter_data}

---

จากข้อมูลข้างต้น กรุณาเขียนบทความวิเคราะห์ตามโครงสร้างนี้:

## 📊 สรุปภาพรวมตลาดวันนี้
- วิเคราะห์สถานการณ์ตลาดหุ้นไทยและตลาดโลก
- ปัจจัยสำคัญที่ส่งผลต่อตลาด
- Sentiment จาก Twitter (bullish/bearish/neutral)

## 🐦 Trending จาก Twitter/X
- หัวข้อที่ถูกพูดถึงมากที่สุด
- หุ้นที่ถูก mention บ่อย
- ความเห็นจากนักวิเคราะห์และ influencer

## 📰 สิ่งที่นักลงทุนต้องรู้วันนี้
- ข่าวสำคัญที่ส่งผลต่อการลงทุน
- ตัวเลขเศรษฐกิจที่ประกาศ
- เหตุการณ์สำคัญที่ต้องติดตาม

## 🎯 กลุ่มหุ้นน่าจับตา
- กลุ่มอุตสาหกรรมที่มีโอกาส
- หุ้นที่มีข่าวหรือปัจจัยพิเศษ

## 💡 คำแนะนำการลงทุน
- กลยุทธ์สำหรับนักลงทุนระยะสั้น
- แนวทางสำหรับนักลงทุนระยะยาว

## ⚠️ ความเสี่ยงที่ต้องระวัง
- ปัจจัยเสี่ยงที่อาจกระทบพอร์ต
- สัญญาณเตือนจาก Social Media

[WARNING]การลงทุนมีความเสี่ยง ข้อมูลจาก Twitter เป็นเพียงความเห็นส่วนตัว ควรศึกษาข้อมูลเพิ่มเติมก่อนตัดสินใจลงทุน[/WARNING]

เขียนเนื้อหาให้กระชับ อ่านง่าย อ้างอิงข้อมูลจาก Twitter ที่ให้มา"""

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )

            content = response.text
            generated_at = datetime.now().isoformat()

            # Update cache
            investor_insights_cache['content'] = content
            investor_insights_cache['generated_at'] = generated_at
            investor_insights_cache['date'] = today

            return content, generated_at

        except Exception as e:
            logger.error(f"Error generating investor insights: {str(e)}")
            return None, None

    return None, None


@app.route('/api/investor-insights')
def get_investor_insights():
    """Get daily investor insights analysis with Twitter data"""
    from datetime import datetime

    refresh = request.args.get('refresh', 'false').lower() == 'true'

    # Clear cache if refresh requested
    if refresh:
        investor_insights_cache['content'] = None
        investor_insights_cache['date'] = None
        investor_insights_cache['twitter_data'] = None

    content, generated_at = generate_investor_insights()

    if content:
        return jsonify({
            'status': 'success',
            'content': content,
            'generated_at': generated_at,
            'cached': investor_insights_cache['date'] == datetime.now().strftime('%Y-%m-%d') and not refresh,
            'source': 'twitter_gemini_search',
            'has_twitter_data': bool(investor_insights_cache.get('twitter_data'))
        })
    else:
        # Return fallback content
        fallback_content = """## สรุปภาพรวมตลาดวันนี้

ขณะนี้ไม่สามารถดึงข้อมูลวิเคราะห์ได้ กรุณาลองใหม่อีกครั้ง

## คำแนะนำทั่วไป

- **ติดตามข่าวสาร**: อ่านข่าวจากแหล่งข้อมูลที่น่าเชื่อถือ
- **วิเคราะห์ก่อนลงทุน**: ศึกษาพื้นฐานบริษัทก่อนตัดสินใจ
- **กระจายความเสี่ยง**: ไม่ลงทุนหุ้นตัวเดียว
- **ตั้ง Stop Loss**: กำหนดจุดตัดขาดทุนเสมอ

## ความเสี่ยงที่ต้องระวัง

[WARNING]การลงทุนมีความเสี่ยง ผู้ลงทุนควรศึกษาข้อมูลก่อนตัดสินใจลงทุน[/WARNING]"""

        return jsonify({
            'status': 'success',
            'content': fallback_content,
            'generated_at': datetime.now().isoformat(),
            'cached': False,
            'is_fallback': True
        })


# ===== ENGLISH LEARNING =====

# Cache for English learning exercises
english_exercise_cache = {}


def generate_english_exercise(exercise_type, difficulty, topic=None):
    """Generate an English learning exercise using Gemini or Typhoon API"""

    # Add uniqueness instruction
    unique_instruction = f"\n\nIMPORTANT: Generate a UNIQUE and CREATIVE question. Use randomness. Current timestamp: {datetime.now().isoformat()}"

    # Select appropriate prompt based on exercise type
    if exercise_type == 'vocabulary':
        prompt = VOCABULARY_EXERCISE_PROMPT.format(
            difficulty=difficulty.upper(),
            topic=topic or 'general everyday vocabulary'
        ) + unique_instruction
    elif exercise_type == 'grammar':
        prompt = GRAMMAR_EXERCISE_PROMPT.format(
            difficulty=difficulty.upper()
        ) + unique_instruction
    elif exercise_type == 'translation':
        direction = random.choice(['thai_to_english', 'english_to_thai'])
        prompt = TRANSLATION_EXERCISE_PROMPT.format(
            difficulty=difficulty.upper(),
            direction=direction
        ) + unique_instruction
    else:
        logger.error(f"Unknown exercise type: {exercise_type}")
        return None

    text = None

    # Try Gemini first
    if gemini_client:
        try:
            logger.info(f"Generating {exercise_type} exercise with Gemini at {difficulty} level, topic: {topic}")

            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )

            text = response.text.strip()
            logger.info(f"Gemini response received: {len(text)} chars")

        except Exception as e:
            logger.error(f"Gemini error for English Learning: {str(e)}")
            text = None

    # Fallback to Typhoon if Gemini fails
    if not text and TYPHOON_API_KEY:
        try:
            logger.info(f"Falling back to Typhoon for {exercise_type} exercise")

            payload = {
                "model": "typhoon-v2.5-30b-a3b-instruct",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an English language tutor. Generate exercises in valid JSON format only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.8
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TYPHOON_API_KEY}"
            }

            response = requests.post(TYPHOON_URL, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"].strip()
                logger.info(f"Typhoon response received: {len(text)} chars")
            else:
                logger.error(f"Typhoon API error: {response.status_code}")
                text = None

        except Exception as e:
            logger.error(f"Typhoon error for English Learning: {str(e)}")
            text = None

    if not text:
        logger.error("Both Gemini and Typhoon failed to generate exercise")
        return None

    try:
        # Remove markdown code blocks if present
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\n?', '', text)
            text = re.sub(r'\n?```$', '', text)

        exercise_data = json.loads(text)

        # Generate unique ID
        exercise_id = str(uuid.uuid4())

        # Store exercise with correct answer for validation
        english_exercise_cache[exercise_id] = {
            'correct_answer': exercise_data.get('correct_answer'),
            'explanation': exercise_data.get('explanation', ''),
            'thai_explanation': exercise_data.get('thai_explanation', ''),
            'grammar_rule': exercise_data.get('grammar_rule', ''),
            'created_at': datetime.now()
        }

        logger.info(f"Exercise generated successfully: {exercise_id}")

        return {
            'id': exercise_id,
            'type': exercise_type,
            'difficulty': difficulty,
            'question': exercise_data.get('question'),
            'options': exercise_data.get('options', []),
            'context': exercise_data.get('context', ''),
            'hint_available': True
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse exercise JSON: {e}")
        logger.error(f"Raw response: {text[:500] if text else 'None'}")
        return None
    except Exception as e:
        logger.error(f"Error processing exercise: {str(e)}")
        return None


def check_exercise_answer(exercise_id, user_answer):
    """Check user's answer against correct answer"""
    if exercise_id not in english_exercise_cache:
        logger.warning(f"Exercise not found: {exercise_id}")
        return None

    cached = english_exercise_cache[exercise_id]
    correct_answer = cached['correct_answer']

    # Normalize answers for comparison (case-insensitive, strip whitespace)
    user_normalized = user_answer.strip().lower()
    correct_normalized = correct_answer.strip().lower()

    is_correct = user_normalized == correct_normalized

    # Calculate XP
    base_xp = 10 if is_correct else 2  # Participation XP even if wrong

    return {
        'correct': is_correct,
        'correct_answer': correct_answer,
        'explanation': cached.get('explanation', ''),
        'thai_explanation': cached.get('thai_explanation', ''),
        'grammar_rule': cached.get('grammar_rule', ''),
        'xp_earned': base_xp
    }


@app.route('/api/english-learning/exercise', methods=['POST'])
def get_english_exercise():
    """Generate a new English learning exercise"""
    try:
        data = request.json
        exercise_type = data.get('exercise_type', 'vocabulary')
        difficulty = data.get('difficulty', 'beginner')
        topic = data.get('topic')

        # Validate inputs
        valid_types = ['vocabulary', 'grammar', 'translation']
        valid_difficulties = ['beginner', 'intermediate', 'advanced']

        if exercise_type not in valid_types:
            return jsonify({
                'status': 'error',
                'error': f'Invalid exercise type. Must be one of: {valid_types}'
            }), 400

        if difficulty not in valid_difficulties:
            return jsonify({
                'status': 'error',
                'error': f'Invalid difficulty. Must be one of: {valid_difficulties}'
            }), 400

        exercise = generate_english_exercise(exercise_type, difficulty, topic)

        if exercise:
            return jsonify({
                'status': 'success',
                'exercise': exercise
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Failed to generate exercise. Please try again.'
            }), 500

    except Exception as e:
        logger.error(f"Error in /api/english-learning/exercise: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/english-learning/check', methods=['POST'])
def check_english_answer():
    """Check user's answer for an exercise"""
    try:
        data = request.json
        exercise_id = data.get('exercise_id')
        user_answer = data.get('user_answer', '').strip()

        if not exercise_id:
            return jsonify({
                'status': 'error',
                'error': 'Exercise ID is required'
            }), 400

        if not user_answer:
            return jsonify({
                'status': 'error',
                'error': 'Answer is required'
            }), 400

        result = check_exercise_answer(exercise_id, user_answer)

        if result:
            # Generate encouragement message
            if result['correct']:
                encouragements = [
                    "Excellent work! 🎉",
                    "You're on fire! 🔥",
                    "Keep it up! 💪",
                    "Amazing! ⭐",
                    "Perfect! ✨",
                    "Great job! 👏"
                ]
            else:
                encouragements = [
                    "Don't give up! 💪",
                    "You'll get it next time! 🌟",
                    "Keep practicing! 📚",
                    "Learning is a journey! 🚀",
                    "Almost there! Keep going! 💫"
                ]

            result['encouragement'] = random.choice(encouragements)
            result['status'] = 'success'

            return jsonify(result)
        else:
            return jsonify({
                'status': 'error',
                'error': 'Exercise not found or expired. Please start a new exercise.'
            }), 404

    except Exception as e:
        logger.error(f"Error in /api/english-learning/check: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/english-learning/hint', methods=['POST'])
def get_english_hint():
    """Get a hint for an exercise"""
    try:
        data = request.json
        exercise_id = data.get('exercise_id')

        if not exercise_id:
            return jsonify({
                'status': 'error',
                'error': 'Exercise ID is required'
            }), 400

        if exercise_id not in english_exercise_cache:
            return jsonify({
                'status': 'error',
                'error': 'Exercise not found'
            }), 404

        cached = english_exercise_cache[exercise_id]
        correct_answer = cached['correct_answer']

        # Generate hint (first letter + length)
        first_letter = correct_answer[0].upper() if correct_answer else '?'
        answer_length = len(correct_answer)

        hint = f"💡 The answer starts with '{first_letter}' and has {answer_length} characters."

        # Add grammar rule hint if available
        if cached.get('grammar_rule'):
            hint += f"\n📝 Grammar hint: {cached['grammar_rule']}"

        return jsonify({
            'status': 'success',
            'hint': hint
        })

    except Exception as e:
        logger.error(f"Error in /api/english-learning/hint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# ── Daily investor-insights email ────────────────────────────────────────────

@app.route('/api/send-investor-email', methods=['POST'])
def trigger_investor_email():
    """Manually trigger the investor insights email (for testing / on-demand)."""
    try:
        from email_service import send_investor_email

        refresh = request.json.get('refresh', False) if request.is_json else False

        if refresh:
            investor_insights_cache['content'] = None
            investor_insights_cache['date']    = None
            investor_insights_cache['twitter_data'] = None

        content, generated_at = generate_investor_insights()

        if not content:
            return jsonify({'status': 'error', 'error': 'Failed to generate insights'}), 500

        success, message = send_investor_email(content, generated_at)

        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'error': message}), 500

    except Exception as e:
        logger.error(f"Error in /api/send-investor-email: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

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

    # Start the daily email scheduler
    try:
        from scheduler import start_scheduler
        start_scheduler()
    except Exception as e:
        logger.error(f"Could not start scheduler: {e}")

    app.run(debug=True, host='0.0.0.0', port=5000)
