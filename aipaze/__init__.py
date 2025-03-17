from .server import MCPServer
from .client import MCPClient
from .resources import ResourceManager
from .adapters import adapt_llm, connect_llm_openai
from .tools import tool, get_tool_registry

# Global instances
_server = MCPServer()
_resources = ResourceManager()

def resource(name: str):
    """Register a resource with optional dependencies."""
    def decorator(func):
        _resources.register(name, func)
        _server.register_resource(name, _resources.get_callable(name))
        return func
    return decorator

def connect(llm=None, endpoint: str = "auto", mode: str = "local", 
           model: str = None, api_key: str = None, base_url: str = None) -> MCPClient:
    """
    Connect an LLM to the MCP server.
    - llm: Prebuilt LLM callable (optional).
    - endpoint: "auto" (local), "cloud" (hosted), or custom URL.
    - mode: "local" (in-memory), "standalone" (external), "cloud" (hosted).
    - model: LLM model name (e.g., "gpt-3.5-turbo", "llama3-8b-8192").
    - api_key: API key for the LLM provider.
    - base_url: Base URL for the LLM API (e.g., "https://api.groq.com/openai/v1").
    """
    if llm is None and model and api_key:
        llm = connect_llm_openai(model, api_key, base_url)
    elif llm is None:
        raise ValueError("Must provide either 'llm' or 'model' and 'api_key'")

    llm = adapt_llm(llm)
    if endpoint == "auto":
        if mode == "local":
            _server.start_local()
        elif mode == "standalone":
            _server.start_standalone()
        elif mode == "cloud":
            _server.start_cloud()
        endpoint = _server.endpoint
    client = MCPClient(llm, endpoint, _resources)  # Pass ResourceManager to client
    return client

server = _server