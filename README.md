# AIPaze

## Introduction

AIPaze is a comprehensive framework for connecting large language models (LLMs) to external tools, data sources, and resources. It provides a standardized way to build AI applications that can leverage real-time data, execute external functions, and maintain context across interactions.

This library enables developers to create AI applications that can:

- Connect to various LLM providers (OpenAI, local models, etc.)
- Access external tools and data sources
- Maintain conversation memory
- Validate inputs and outputs
- Build complex workflows
- Implement retrieval-augmented generation (RAG)
- Monitor performance and usage

## Installation

```bash
pip install aipaze
```

## Quick Start

### Basic Usage

```python
from aipaze import connect, resource, tool

# Define a resource
@resource("get_weather")
def get_weather(city):
    # Implementation to fetch weather data
    return {"temperature": 72, "condition": "sunny", "city": city}

# Define a tool
@tool("calculate", "Perform mathematical calculations")
def calculate(expression: str):
    return {"result": eval(expression)}

# Connect to an LLM
client = connect(
    model="gpt-3.5-turbo",
    api_key="your-api-key"
)

# Query a resource directly
weather = client.sync_query("get_weather", "New York")
print(weather)

# Let the LLM decide which tool to use
response = client.query_with_tools("What's the weather in San Francisco and what's 25 * 34?")
print(response)
```

## Core Concepts

### Resources

Resources are functions that the LLM can access to retrieve information or perform actions.

```python
@resource("search_database")
async def search_database(query):
    # Implementation to search a database
    results = await db.search(query)
    return results
```

### Tools

Tools are functions that the LLM can choose to use based on user queries.

```python
@tool("send_email", "Send an email to a recipient")
def send_email(to: str, subject: str, body: str):
    # Implementation to send an email
    return {"status": "sent", "to": to}
```

### Memory

Memory allows conversations to maintain context across interactions.

```python
from aipaze import Memory

memory = Memory()
memory.add("user", "What's the weather in Paris?")
memory.add("assistant", "It's currently 22°C and sunny in Paris.")

# Use memory in a query
response = client.query_with_tools("What about tomorrow?", memory=memory)
```

### Workflows

Workflows allow you to chain multiple tools or resources together.

```python
from aipaze import Workflow

workflow = Workflow("email_workflow")
workflow.add_step("search", "search_database", "customer complaint")
workflow.add_step("draft", "generate_response", {"query_results": "search"})
workflow.add_step("send", "send_email", {"draft": "draft"})

results = workflow.sync_execute()
```

## Detailed API Reference

### Client

#### `connect()`

Creates a client to connect to LLMs and resources.

```python
client = connect(
    model="gpt-4",              # LLM model name
    api_key="your-api-key",     # API key for the LLM provider
    base_url=None,              # Optional base URL for API
    model_path=None,            # Path to local model (if using local LLM)
    device="cpu"                # Device for local model
)
```

#### `client.sync_query()`

Synchronously query a resource.

```python
result = client.sync_query(
    "resource_name",  # Name of the resource
    *args,            # Positional arguments for the resource
    **kwargs          # Keyword arguments for the resource
)
```

#### `client.query_with_tools()`

Let the LLM decide which tools to use based on the query.

```python
response = client.query_with_tools(
    prompt,           # User query
    tools=None,       # Optional list of tool names to restrict to
    memory=None       # Optional memory object for context
)
```

### Resource Management

#### `@resource()`

Decorator to register a resource.

```python
@resource("resource_name")
def my_resource(arg1, arg2):
    # Implementation
    return result
```

### Tool Management

#### `@tool()`

Decorator to register a tool.

```python
@tool("tool_name", "Description of when to use this tool")
def my_tool(param1: str, param2: int):
    # Implementation
    return {"result": value}
```

### Memory

#### `Memory`

Class for managing conversation history.

```python
memory = Memory(max_entries=10)

# Add messages
memory.add("user", "Hello")
memory.add("assistant", "Hi there!")

# Get all messages
context = memory.get_context()

# Clear memory
memory.clear()
```

### Prompt Templates

#### `PromptTemplate`

Class for creating reusable prompt templates.

```python
from aipaze import PromptTemplate

template = PromptTemplate("Hello, {name}! The weather in {city} is {weather}.")
prompt = template.format(name="Alice", city="Paris", weather="sunny")
```

#### `ChatTemplate`

Class for creating multi-turn chat templates.

```python
from aipaze import ChatTemplate

chat = ChatTemplate("You are a helpful assistant.")
chat.add_message("user", "Hello, I need help with {topic}")
chat.add_message("assistant", "I'd be happy to help with {topic}.")

messages = chat.format({"topic": "programming"})
```

### Validation

#### `@validate()`

Decorator for input/output validation.

```python
from aipaze import validate
from pydantic import BaseModel

class InputSchema(BaseModel):
    query: str
    max_results: int = 10

@validate(input_schema=InputSchema)
def search(query, max_results=10):
    # Implementation
    return results
```

### Vector Store

#### `VectorStore`

Class for semantic search and RAG applications.

```python
from aipaze import VectorStore

# Create and populate store
vector_store = VectorStore("my_store")
vector_store.add_documents(["Document 1", "Document 2"])

# Search
results = vector_store.similarity_search("query", top_k=3)
```

### Middleware

#### `@middleware`

Decorator to register middleware.

```python
from aipaze import middleware

@middleware
async def logging_middleware(resource, args, result):
    print(f"Called {resource} with {args}, got {result}")
    return result
```

### Metrics

#### `Metrics`

Class for tracking usage and performance.

```python
from aipaze import Metrics

metrics = client.enable_metrics()

# Later, get metrics
summary = metrics.get_summary()
print(f"Total requests: {summary['requests']}")
print(f"Average latency: {summary['avg_latency_ms']}ms")
```

### Workflow

#### `Workflow`

Class for orchestrating multi-step processes.

```python
from aipaze import Workflow

workflow = Workflow("my_workflow")
workflow.add_step("step1", "tool1", input_data)
workflow.add_step("step2", "tool2", "step1", depends_on=["step1"])

results = workflow.sync_execute()
```

## Advanced Usage

### Local LLM Integration

```python
client = connect(
    model_path="path/to/local/model",
    device="cuda"  # or "cpu"
)
```

### Multimodal Support

```python
client = connect(
    model="gpt-4-vision",
    api_key="your-api-key"
)

response = client.llm({
    "messages": [
        {"role": "user", "content": "What's in this image?", "image_url": "https://example.com/image.jpg"}
    ]
})
```

### Parallel Queries

```python
results = client.parallel_query([
    ("resource1", ["arg1", "arg2"]),
    ("resource2", ["arg3"], {"kwarg1": "value1"})
])
```

### Middleware Usage

```python
# Apply middleware to all resources
client.use_middleware(caching_middleware)

# Apply middleware to specific resources
client.use_middleware(logging_middleware, resources=["search", "get_weather"])
```

## Best Practices

### Tool Description Guidelines

- Be specific about when to use a tool
- Clearly describe what the tool does
- Include parameter descriptions

```python
@tool(
    "search_flights", 
    "Search for available flights between cities. ONLY use for flight-related queries."
)
def search_flights(origin: str, destination: str, date: str):
    """
    Search for flights between cities on a specific date.
    
    Args:
        origin: Origin city or airport code
        destination: Destination city or airport code
        date: Travel date in YYYY-MM-DD format
    """
    # Implementation
    return results
```

### Memory Management

- Clear memory when starting new conversations
- Consider summarizing long conversations
- Use metadata to store user preferences

### Error Handling

```python
try:
    response = client.query_with_tools("What's the weather?")
except Exception as e:
    print(f"Error: {str(e)}")
    # Provide fallback response
```

### Resource Usage Optimization

- Monitor token usage with metrics
- Implement caching for frequent queries

## Examples

### RAG Application

```python
# Set up vector store
store = VectorStore("documents")
store.add_documents(["Document about AI", "Document about ML"])

# Define search tool
@tool("search_docs", "Search through documents for information")
def search_docs(query: str):
    results = store.similarity_search(query)
    return {"results": [r["document"] for r in results]}

# Query with context
response = client.query_with_tools("What do the documents say about AI?")
```

### Chatbot with Memory

```python
memory = Memory()

def chat():
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
            
        memory.add("user", user_input)
        response = client.query_with_tools(user_input, memory=memory)
        memory.add("assistant", response)
        
        print(response)

chat()
```

### Workflow Example

```python
workflow = Workflow("customer_support")
workflow.add_step("classify", "classify_issue", "Customer can't log in")
workflow.add_step("search", "search_knowledge_base", "classify")
workflow.add_step("draft", "draft_response", {
    "issue": "classify",
    "knowledge": "search"
})
workflow.add_step("review", "review_response", "draft")

results = workflow.sync_execute()
print(results["review"])
```

## Troubleshooting

### Common Issues

#### Connection Problems

```javascript
Error: WebSocket connection failed
```

**Solution**: Check that the server is running and the endpoint is correct.

#### Tool Execution Errors

```javascript
Error executing tool: tool_name not found
```

**Solution**: Ensure the tool is registered before querying.

#### LLM API Errors

```javascript
OpenAI API error: 401 Unauthorized
```

**Solution**: Verify your API key and permissions.

### Debugging Tips

- Enable detailed logging: `logging.basicConfig(level=logging.DEBUG)`
- Use `client.direct_call()` to bypass the WebSocket for troubleshooting
- Check the server logs for errors

## Future Features

In future versions, we plan to add:

1. **Streaming Support**: For real-time response generation
2. **Advanced Caching**: To improve performance and reduce API costs
3. **More Adapters**: For additional LLM providers
4. **Web Interface**: For easier debugging and testing

## Contributing

We welcome contributions to AIPaze! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

AIPaze is released under the MIT License. See [LICENSE](LICENSE) for details.




## Made with ❤️ by Vishal Singh Baraiya