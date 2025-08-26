import logging
import aiohttp
from typing import Dict

REFACTOR_PROMPT = (
    "Please refactor the following Python code to be more readable, efficient, and adhere to PEP 8 standards.\n"
    "Add docstrings to all functions and classes. Add comments explaining complex logic.\n"
    "Ensure the refactored code maintains the exact original functionality.\n"
    "Only return the raw refactored Python code, without any markdown formatting or explanations.\n\n"
    "Original Code:\n"
    "```python\n"
    "{file_content}\n"
    "```\n\n"
    "Refactored Code:"
)

async def _make_request(session, url, headers, json_payload):
    """A helper function to make async POST requests with error handling."""
    try:
        async with session.post(url, headers=headers, json=json_payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logging.error(f"API error: {response.status} - {error_text}")
                return None
    except Exception as e:
        logging.error(f"Error calling API: {e}")
        return None

def _clean_response(refactored_code: str) -> str:
    """Removes markdown formatting from the model's response."""
    if refactored_code.strip().startswith("```python"):
        refactored_code = refactored_code.strip()[9:]
    if refactored_code.strip().endswith("```"):
        refactored_code = refactored_code.strip()[:-3]
    return refactored_code.strip()

# --- Model Implementations ---

async def refactor_with_gemini(session: aiohttp.ClientSession, file_content: str, api_key: str) -> str:
    """Refactors code using a Google Gemini model."""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": REFACTOR_PROMPT.format(file_content=file_content)}]}]
    }
    result = await _make_request(session, api_url, headers, payload)
    if result and result.get('candidates'):
        return _clean_response(result['candidates'][0]['content']['parts'][0]['text'])
    return file_content

async def refactor_with_openai(session: aiohttp.ClientSession, file_content: str, api_key: str) -> str:
    """Refactors code using an OpenAI model (e.g., GPT-4o)."""
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": REFACTOR_PROMPT.format(file_content=file_content)}]
    }
    result = await _make_request(session, api_url, headers, payload)
    if result and result.get('choices'):
        return _clean_response(result['choices'][0]['message']['content'])
    return file_content

async def refactor_with_anthropic(session: aiohttp.ClientSession, file_content: str, api_key: str) -> str:
    """Refactors code using an Anthropic Claude model."""
    api_url = "https://api.anthropic.com/v1/messages"
    headers = {'x-api-key': api_key, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'}
    payload = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": REFACTOR_PROMPT.format(file_content=file_content)}]
    }
    result = await _make_request(session, api_url, headers, payload)
    if result and result.get('content'):
        return _clean_response(result['content'][0]['text'])
    return file_content
    
async def refactor_with_deepseek(session: aiohttp.ClientSession, file_content: str, api_key: str) -> str:
    """Refactors code using a DeepSeek Coder model."""
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {
        "model": "deepseek-coder",
        "messages": [{"role": "user", "content": REFACTOR_PROMPT.format(file_content=file_content)}]
    }
    result = await _make_request(session, api_url, headers, payload)
    if result and result.get('choices'):
        return _clean_response(result['choices'][0]['message']['content'])
    return file_content

# --- Provider Dictionary ---
MODEL_PROVIDERS = {
    "gemini": {"function": refactor_with_gemini, "api_key_name": "GEMINI_API_KEY"},
    "openai": {"function": refactor_with_openai, "api_key_name": "OPENAI_API_KEY"},
    "claude": {"function": refactor_with_anthropic, "api_key_name": "ANTHROPIC_API_KEY"},
    "deepseek": {"function": refactor_with_deepseek, "api_key_name": "DEEPSEEK_API_KEY"},
}
