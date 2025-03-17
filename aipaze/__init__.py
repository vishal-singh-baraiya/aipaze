from .server import MCPServer
from .client import MCPClient
from .resources import ResourceManager
from .adapters import adapt_llm, connect_llm_openai, connect_llm_local, connect_multimodal_llm
from .tools import tool, get_tool_registry
from .memory import Memory
from .prompt_templates import PromptTemplate
from .validation import validate
from .middleware import middleware, use_middleware
from .vector_store import VectorStore
from .metrics import Metrics
from .workflow import Workflow

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
           model: str = None, api_key: str = None, base_url: str = None,
           model_path: str = None, device: str = "cpu") -> MCPClient:
    """
    Connect an LLM to the MCP server.
    - llm: Prebuilt LLM callable (optional).
    - endpoint: "auto" (local), "cloud" (hosted), or custom URL.
    - mode: "local" (in-memory), "standalone" (external), "cloud" (hosted).
    - model: LLM model name (e.g., "gpt-3.5-turbo", "llama3-8b-8192").
    - api_key: API key for the LLM provider.
    - base_url: Base URL for the LLM API (e.g., "https://api.groq.com/openai/v1").
    - model_path: Path to local model for local LLM support.
    - device: Device to use for local model ("cpu", "cuda", etc.).
    """
    if llm is None:
        if model and api_key:
            llm = connect_llm_openai(model, api_key, base_url)
        elif model_path:
            llm = connect_llm_local(model_path, device)
        else:
            raise ValueError("Must provide either 'llm', 'model' and 'api_key', or 'model_path'")

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