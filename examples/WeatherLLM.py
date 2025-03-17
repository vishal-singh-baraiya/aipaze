import os
import sys
import requests
from flask import Flask, request, jsonify
import re
import logging
import asyncio

sys.path.append(os.path.abspath('..'))

from aipaze import resource, connect, server

# Flask app
app = Flask(__name__)

# Weather resource using OpenWeatherMap API
@resource("get_weather")
def get_weather(city: str) -> dict:
    # Extract city name if the input is a question
    if "weather" in city.lower() or "temperature" in city.lower():
        # Try to extract city name from the question
        match = re.search(r'in\s+([A-Za-z\s]+)(?:\s+today|\s+now|\s+right now|\?|$)', city)
        if match:
            city = match.group(1).strip()
    
    # For "New York today" -> extract just "New York"
    if "today" in city:
        city = city.replace("today", "").strip()
    
    logging.info(f"Extracted city name: {city}")
    
    api_key = "98981aba6b80e87797a5b950cee0eccc"
    if not api_key:
        return {"error": "Weather API key not set"}

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        logging.info(f"Fetching weather for: {city}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "city": city,
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"]
        }
    except requests.RequestException as e:
        logging.error(f"Weather API error: {str(e)}")
        return {"error": f"Failed to fetch weather: {str(e)}"}

# Global variable to store the server endpoint
SERVER_ENDPOINT = None

# Start the MCP server explicitly before Flask
def start_server():
    global SERVER_ENDPOINT
    if not server.endpoint:
        # Create and use a dedicated event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server.start_local()
        SERVER_ENDPOINT = server.endpoint
        logging.info(f"Server started at: {SERVER_ENDPOINT}")

# Start server once at module level
start_server()

# Connect to Groq model using aipaze
client = connect(
    endpoint=SERVER_ENDPOINT,  # Use the stored endpoint
    model="gemma2-9b-it",
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

@app.route("/weather", methods=["POST"])
def weather_prompt():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request"}), 400

    prompt = data["prompt"]
    logging.info(f"Received prompt: {prompt}")
    
    try:
        # Extract city from prompt for direct API call
        city = prompt
        if "weather" in prompt.lower() or "temperature" in prompt.lower():
            match = re.search(r'in\s+([A-Za-z\s]+)(?:\s+today|\s+now|\s+right now|\?|$)', prompt)
            if match:
                city = match.group(1).strip()
        
        # Remove "today" if present
        if "today" in city:
            city = city.replace("today", "").strip()
            
        logging.info(f"Extracted city for direct call: {city}")
        
        # Try direct API call
        weather_data = get_weather(city)
        
        if isinstance(weather_data, dict) and "error" in weather_data:
            return jsonify({"error": weather_data["error"]}), 500
            
        response = f"The weather in {weather_data['city']} is {weather_data['temperature']}°C with {weather_data['description']} and {weather_data['humidity']}% humidity."
        return jsonify({"response": response})
            
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "server_endpoint": SERVER_ENDPOINT})

if __name__ == "__main__":
    # Disable debug mode to prevent server restarting
    app.run(debug=False, host="0.0.0.0", port=5000)