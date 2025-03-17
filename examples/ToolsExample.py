import os
import sys
import requests
from flask import Flask, request, jsonify
import re
import logging

sys.path.append(os.path.abspath('..'))

from aipaze import resource, connect, server, tool, get_tool_registry

# Flask app
app = Flask(__name__)

# Start the server
server.start_local()

# Register tools with specific descriptions
@tool("get_weather", "Get current weather information for a specific city. ONLY use this for weather queries.")
def get_weather(city: str) -> dict:
    """Get current weather information for the specified city. Only use for weather-related questions."""
    api_key = "98981aba6b80e87797a5b950cee0eccc"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    return {
        "city": city,
        "temperature": data["main"]["temp"],
        "description": data["weather"][0]["description"],
        "humidity": data["main"]["humidity"]
    }

@tool("calculate", "Perform a mathematical calculation. ONLY use this for math problems.")
def calculate(expression: str) -> dict:
    """Calculate the result of a mathematical expression. Only use for calculation questions."""
    # Basic safety check
    if re.match(r'^[0-9+\-*/() .]+$', expression):
        result = eval(expression)
        return {
            "expression": expression,
            "result": result
        }
    else:
        raise ValueError("Invalid expression. Only numbers and basic operators (+,-,*,/) are allowed.")

# Connect to LLM
client = connect(
    model="gemma2-9b-it",
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request"}), 400
    
    prompt = data["prompt"]
    logging.info(f"Received prompt: {prompt}")
    
    try:
        # Let the LLM decide whether to use tools or answer directly
        response = client.query_with_tools(prompt)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error processing prompt: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    tool_registry = get_tool_registry()
    tools = [name for name in tool_registry.tools.keys()]
    return jsonify({
        "status": "healthy",
        "server_endpoint": server.endpoint,
        "registered_tools": tools
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)