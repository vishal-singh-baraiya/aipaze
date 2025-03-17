import asyncio
import logging
import os
import re
import requests
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.append(os.path.abspath('..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure AIPaze is installed
try:
    from aipaze import (
        connect, tool, get_tool_registry, resource, server, 
        Memory, VectorStore, PromptTemplate, Metrics, Workflow, validate
    )
    from pydantic import BaseModel, Field
except ImportError:
    logger.error("AIPaze library not found. Please install it with: pip install aipaze")
    sys.exit(1)

# Configuration
API_KEY = "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"
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "98981aba6b80e87797a5b950cee0eccc")
MODEL = os.environ.get("LLM_MODEL", "gemma2-9b-it")

# Start the AIPaze server
server.start_local()
logger.info(f"AIPaze server started at {server.endpoint}")

# Set up metrics tracking
metrics = Metrics()
metrics.start_tracking()
logger.info("Metrics tracking started")

# Set up memory system
memory = Memory(max_entries=20)
logger.info("Memory system initialized")

# Set up vector store for knowledge
knowledge_store = VectorStore("demo_knowledge")
logger.info("Vector store initialized")

# Add some knowledge to the vector store
knowledge_documents = [
    "AIPaze is a framework for connecting LLMs to external tools and data sources.",
    "Python is a programming language known for its readability and versatility.",
    "The Eiffel Tower is 330 meters tall and located in Paris, France.",
    "Machine learning is a subset of artificial intelligence focused on building systems that learn from data.",
    "The first computer programmer was Ada Lovelace, who wrote an algorithm for Charles Babbage's Analytical Engine.",
    "Tokyo is the capital of Japan and the most populous metropolitan area in the world.",
    "The Great Barrier Reef is the world's largest coral reef system and is located off the coast of Australia.",
    "Renewable energy sources include solar, wind, hydro, and geothermal power.",
    "Quantum computing uses quantum-mechanical phenomena to perform calculations.",
    "The human genome contains approximately 3 billion base pairs."
]

knowledge_store.add_documents(knowledge_documents)
logger.info(f"Added {len(knowledge_documents)} documents to knowledge store")

# Define input validation models
class WeatherInput(BaseModel):
    city: str = Field(..., description="The city to get weather for")

class CalculationInput(BaseModel):
    expression: str = Field(..., description="The mathematical expression to calculate")

class SearchInput(BaseModel):
    query: str = Field(..., description="The search query")
    top_k: int = Field(3, description="Number of results to return")

class ReminderInput(BaseModel):
    message: str = Field(..., description="The reminder message")
    time_minutes: int = Field(..., description="Time in minutes to set the reminder for")

# Define tools with validation
@tool("get_weather", "Get current weather information for a Country/State/City. ONLY use this for weather queries.")
@validate(input_schema=WeatherInput)
def get_weather(city: str) -> dict:
    """Get current weather information for the specified city."""
    start_time = time.time()
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Record successful API call
        metrics.record_request(
            tokens=100,  # Approximate token count
            latency_ms=(time.time() - start_time) * 1000,
            model="weather_api",
            resource="get_weather"
        )
        
        return {
            "city": city,
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
    except Exception as e:
        metrics.record_error()
        logger.error(f"Error fetching weather: {str(e)}")
        raise ValueError(f"Error fetching weather for {city}: {str(e)}")

@tool("calculate", "Perform a mathematical calculation. ONLY use this for math problems.")
@validate(input_schema=CalculationInput)
def calculate(expression: str) -> dict:
    """Calculate the result of a mathematical expression."""
    start_time = time.time()
    
    # Basic safety check to prevent code execution
    if not re.match(r'^[0-9+\-*/() .]+$', expression):
        metrics.record_error()
        raise ValueError("Invalid expression. Only numbers and basic operators (+,-,*,/) are allowed.")
    
    try:
        result = eval(expression)
        
        # Record calculation in metrics
        metrics.record_request(
            tokens=20,  # Approximate token count
            latency_ms=(time.time() - start_time) * 1000,
            model="calculator",
            resource="calculate"
        )
        
        return {
            "expression": expression,
            "result": result
        }
    except Exception as e:
        metrics.record_error()
        logger.error(f"Error calculating expression: {str(e)}")
        raise ValueError(f"Error calculating expression: {str(e)}")

@tool("search_knowledge", "Search for information in the knowledge base. Use for general knowledge questions.")
@validate(input_schema=SearchInput)
def search_knowledge(query: str, top_k: int = 3) -> dict:
    """Search the knowledge base for information related to the query."""
    start_time = time.time()
    
    try:
        results = knowledge_store.similarity_search(query, top_k=top_k)
        
        # Record search in metrics
        metrics.record_request(
            tokens=50,  # Approximate token count
            latency_ms=(time.time() - start_time) * 1000,
            model="vector_db",
            resource="search_knowledge"
        )
        
        return {
            "query": query,
            "results": [r["document"] for r in results],
            "scores": [r["score"] for r in results]
        }
    except Exception as e:
        metrics.record_error()
        logger.error(f"Error searching knowledge: {str(e)}")
        raise ValueError(f"Error searching knowledge: {str(e)}")

@tool("set_reminder", "Set a reminder for a future time. Use for reminder requests.")
@validate(input_schema=ReminderInput)
def set_reminder(message: str, time_minutes: int) -> dict:
    """Set a reminder for a specified number of minutes in the future."""
    start_time = time.time()
    
    try:
        current_time = datetime.now()
        reminder_time = current_time.timestamp() + (time_minutes * 60)
        formatted_reminder_time = datetime.fromtimestamp(reminder_time).strftime("%Y-%m-%d %H:%M:%S")
        
        # In a real application, you would actually set up a reminder system
        # Here we just pretend to do it
        
        # Record reminder in metrics
        metrics.record_request(
            tokens=30,  # Approximate token count
            latency_ms=(time.time() - start_time) * 1000,
            model="reminder_system",
            resource="set_reminder"
        )
        
        return {
            "message": message,
            "reminder_time": formatted_reminder_time,
            "minutes_from_now": time_minutes,
            "status": "Reminder set successfully"
        }
    except Exception as e:
        metrics.record_error()
        logger.error(f"Error setting reminder: {str(e)}")
        raise ValueError(f"Error setting reminder: {str(e)}")

# Define a resource for getting the current time
@resource("get_current_time")
def get_current_time() -> dict:
    """Get the current time in different formats."""
    now = datetime.now()
    return {
        "iso_format": now.isoformat(),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
        "unix_timestamp": now.timestamp(),
        "timezone": "UTC" if time.daylight else "Local"
    }

# Create a prompt template for greeting users
greeting_template = PromptTemplate(
    "Hello {name}! Welcome to the AIPaze demo. Today is {date} and the time is {time}. How can I help you?"
)

# Connect to LLM
client = connect(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"  # Change as needed for your provider
)

# Enable metrics on the client
client.enable_metrics(metrics)
logger.info(f"Connected to LLM model: {MODEL}")

# Define a function to process user queries
async def process_query(query: str, user_name: str = "User") -> str:
    """Process a user query, potentially using tools."""
    
    # Special case for greetings
    if any(greeting in query.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
        # Get current time
        time_data = client.sync_query("get_current_time")
        # Use the prompt template
        greeting = greeting_template.format(
            name=user_name,
            date=time_data["formatted"].split()[0],
            time=time_data["formatted"].split()[1]
        )
        return greeting
    
    # Add to memory
    memory.add("user", query)
    
    try:
        # Let the LLM decide whether to use tools
        start_time = time.time()
        response = client.query_with_tools(query, memory=memory)
        
        # Add response to memory
        memory.add("assistant", response)
        
        # Record query in metrics
        metrics.record_request(
            tokens=len(query) // 4,  # Rough estimate
            latency_ms=(time.time() - start_time) * 1000,
            model=MODEL,
            resource="query_with_tools"
        )
        
        return response
    except Exception as e:
        metrics.record_error()
        logger.error(f"Error processing query: {str(e)}")
        return f"I encountered an error: {str(e)}"

# Define a function to execute a workflow
async def run_workflow(query: str) -> Dict[str, Any]:
    """Run a workflow based on the query."""
    
    workflow = Workflow("query_workflow")
    
    # Always start with a knowledge search
    workflow.add_step("search", "search_knowledge", {"query": query})
    
    # If query mentions weather, add weather step
    if any(term in query.lower() for term in ["weather", "temperature", "forecast"]):
        # Extract city from query
        match = re.search(r'in\s+([A-Za-z\s]+)(?:\s+today|\s+now|\s+right now|\?|$)', query)
        city = match.group(1).strip() if match else "New York"
        workflow.add_step("weather", "get_weather", {"city": city})
    
    # If query mentions calculation, add calculation step
    if any(term in query.lower() for term in ["calculate", "compute", "what is", "solve"]):
        # For demo purposes, extract simple expressions
        expression = "123 + 456"  # Default
        if re.search(r'\d+\s*[\+\-\*/]\s*\d+', query):
            expression_match = re.search(r'(\d+\s*[\+\-\*/]\s*\d+)', query)
            if expression_match:
                expression = expression_match.group(1)
        
        workflow.add_step("calculate", "calculate", {"expression": expression})
    
    # If query mentions reminder, add reminder step
    if any(term in query.lower() for term in ["remind", "reminder", "remember"]):
        message = query
        time_minutes = 30  # Default
        
        # Try to extract time
        time_match = re.search(r'(\d+)\s*(?:minute|minutes|min)', query)
        if time_match:
            time_minutes = int(time_match.group(1))
        
        workflow.add_step("reminder", "set_reminder", {
            "message": message,
            "time_minutes": time_minutes
        })
    
    # Set client for the workflow
    workflow.set_client(client)
    
    # Execute workflow
    try:
        results = workflow.sync_execute()
        return results
    except Exception as e:
        metrics.record_error()
        logger.error(f"Error executing workflow: {str(e)}")
        return {"error": str(e)}

# Define the main interactive CLI
async def main():
    """Main interactive CLI function."""
    print("\n" + "="*50)
    print("🤖 AIPaze Demo - Interactive CLI")
    print("="*50)
    print("Type 'exit' to quit, 'stats' to see metrics, 'memory' to see conversation history.")
    print("Try asking about the weather, doing calculations, or searching for knowledge.")
    print("You can also try 'workflow:' followed by a query to run a workflow.")
    print("="*50 + "\n")
    
    user_name = input("What's your name? ").strip() or "User"
    print(f"Hello, {user_name}! How can I help you today?\n")
    
    while True:
        query = input(f"{user_name}> ").strip()
        
        if query.lower() == "exit":
            print("\nThank you for using AIPaze Demo. Goodbye!")
            break
        
        elif query.lower() == "stats":
            stats = metrics.get_summary()
            print("\n" + "="*50)
            print("📊 AIPaze Metrics")
            print("="*50)
            print(f"Total Requests: {stats['requests']}")
            print(f"Error Rate: {stats['error_rate']:.2f}%")
            print(f"Average Latency: {stats['avg_latency_ms']:.2f}ms")
            print(f"Total Tokens: {stats['total_tokens']}")
            print(f"Uptime: {stats['uptime_seconds']:.2f} seconds")
            print("\nResource Usage:")
            for resource, count in stats['resource_counts'].items():
                print(f"  - {resource}: {count} calls")
            print("="*50 + "\n")
            
        elif query.lower() == "memory":
            print("\n" + "="*50)
            print("🧠 Conversation Memory")
            print("="*50)
            if not memory.messages:
                print("No conversation history yet.")
            else:
                for i, msg in enumerate(memory.messages):
                    print(f"{msg['role'].capitalize()}: {msg['content']}")
            print("="*50 + "\n")
            
        elif query.lower().startswith("workflow:"):
            workflow_query = query[9:].strip()
            print("Running workflow...")
            results = await run_workflow(workflow_query)
            
            print("\n" + "="*50)
            print("⚙️ Workflow Results")
            print("="*50)
            for step, result in results.items():
                if step == "initial":
                    continue
                print(f"Step '{step}':")
                if isinstance(result, dict):
                    for k, v in result.items():
                        print(f"  - {k}: {v}")
                else:
                    print(f"  - {result}")
            print("="*50 + "\n")
            
        else:
            print("Processing your query...")
            response = await process_query(query, user_name)
            print(f"\nAIPaze: {response}\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
    finally:
        # Print final metrics
        print("\nFinal Metrics Summary:")
        print(metrics.get_summary())
        print("\nShutting down AIPaze server...")
        