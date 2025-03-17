import websockets
import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, List, Dict
from .utils import retry
from .resources import ResourceManager
from .tools import get_tool_registry

class MCPClient:
    def __init__(self, llm, endpoint: str, resources: ResourceManager):
        self.llm = llm
        self.endpoint = endpoint
        self.resources = resources  # Properly assign ResourceManager
        self.executor = ThreadPoolExecutor(max_workers=4)
        logging.info(f"Client initialized with endpoint: {endpoint}")

    @retry(max_attempts=3, delay=1)
    async def query(self, resource: str, *args) -> Any:
        try:
            # Use the correct parameters for websockets.connect
            # Remove timeout parameter and use only ping settings
            async with websockets.connect(
                self.endpoint,
                ping_interval=5,
                ping_timeout=10
            ) as ws:
                payload = {"resource": resource, "args": args}
                logging.info(f"Sending query to resource: {resource}")
                await ws.send(json.dumps(payload))
                response = await ws.recv()
                data = json.loads(response)
                if "error" in data:
                    logging.error(f"Error from server: {data['error']}")
                    raise Exception(data["error"])
                return self.llm(data["result"])
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
            logging.error(f"WebSocket connection error: {str(e)}")
            raise

    def sync_query(self, resource: str, *args) -> Any:
        # Create a new event loop for each query to avoid conflicts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.query(resource, *args))
        finally:
            loop.close()

    def parallel_query(self, tasks: list[tuple[str, list]]) -> list:
        futures = [self.executor.submit(self.sync_query, task[0], *task[1]) for task in tasks]
        return [f.result() for f in futures]

    def direct_call(self, resource_name: str, *args):
        """Directly call a resource, bypassing the websocket for troubleshooting"""
        if resource_name in self.resources.resources:
            result = self.resources.resources[resource_name](*args)
            # Handle coroutines if necessary
            if asyncio.iscoroutine(result):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(result)
                finally:
                    loop.close()
            return self.llm(result)
        else:
            raise ValueError(f"Resource {resource_name} not found")
            
    def query_with_tools(self, prompt: str, tools: List[str] = None) -> Any:
            """
            Query the LLM with tools that it can choose to use based on the prompt.
            
            Args:
                prompt: The user's input prompt
                tools: Optional list of tool names to make available. If None, all tools are available.
            
            Returns:
                The LLM's response after potentially using tools
            """
            registry = get_tool_registry()
            
            # Filter tools if specified
            available_tools = registry.get_tool_specs()
            if tools:
                available_tools = [t for t in available_tools if t["name"] in tools]
            
            if not available_tools:
                logging.warning("No tools available for query_with_tools")
                return self.llm(prompt)
            
            # Create the system message with tool specifications
            system_message = {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to external tools. "
                    "ONLY use a tool when the user's question matches the specific purpose of that tool. "
                    "For general knowledge questions or any request that doesn't match a tool's description, "
                    "answer directly using your knowledge without calling a tool.\n\n"
                    "Available tools:\n"
                    f"{json.dumps(available_tools, indent=2)}\n\n"
                    "To use a tool, respond with a JSON object in this format:\n"
                    '{"tool": "tool_name", "parameters": {"param1": "value1"}}\n\n'
                    "Only respond with JSON when using a tool. For all other queries, respond normally."
                )
            }
            
            # Send to LLM with tools description
            llm_input = {
                "messages": [
                    system_message,
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Get initial response from LLM
            initial_response = self.llm(llm_input)
            logging.info(f"Initial LLM response type: {type(initial_response)}")
            
            # Parse the response to find tool call
            tool_call = None
            
            if isinstance(initial_response, dict) and "tool" in initial_response and "parameters" in initial_response:
                tool_call = initial_response
            elif isinstance(initial_response, str):
                # Clean up the response - remove markdown code blocks
                cleaned_response = initial_response
                # Remove markdown code blocks if present
                if "```" in cleaned_response:
                    cleaned_response = re.sub(r'```(?:json)?\n(.*?)\n```', r'\1', cleaned_response, flags=re.DOTALL)
                    logging.info(f"Cleaned response from markdown: {cleaned_response}")
                
                # Check if it looks like a JSON object with tool fields
                if "{" in cleaned_response and "tool" in cleaned_response and "parameters" in cleaned_response:
                    # Try to extract JSON from the response
                    try:
                        # First try to parse the entire cleaned response
                        tool_call = json.loads(cleaned_response)
                        if not isinstance(tool_call, dict) or "tool" not in tool_call or "parameters" not in tool_call:
                            tool_call = None
                    except json.JSONDecodeError:
                        # If that fails, try to extract JSON using regex
                        json_match = re.search(r'({.*?})', cleaned_response.replace('\n', ' '))
                        if json_match:
                            try:
                                extracted_json = json_match.group(1)
                                tool_call = json.loads(extracted_json)
                                if not isinstance(tool_call, dict) or "tool" not in tool_call or "parameters" not in tool_call:
                                    tool_call = None
                            except json.JSONDecodeError:
                                tool_call = None
            
            # If LLM didn't return a tool call, just return the response
            if not tool_call:
                logging.info("LLM did not call a tool, returning as normal response")
                return initial_response
            
            # Execute the tool
            try:
                tool_name = tool_call["tool"]
                parameters = tool_call["parameters"]
                
                if tool_name not in registry.tools:
                    logging.warning(f"Tool '{tool_name}' not found in registry")
                    return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(registry.tools.keys())}"
                
                logging.info(f"Executing tool: {tool_name} with parameters: {parameters}")
                tool_result = registry.execute_tool(tool_name, **parameters)
                
                # Send the tool result back to the LLM for final response
                llm_input["messages"].extend([
                    {"role": "assistant", "content": json.dumps(tool_call)},
                    {"role": "user", "content": f"Tool result: {json.dumps(tool_result)}. Please provide a human-friendly response based on this result."}
                ])
                
                final_response = self.llm(llm_input)
                return final_response
            except Exception as e:
                logging.error(f"Error executing tool: {str(e)}")
                return f"Error executing tool: {str(e)}"