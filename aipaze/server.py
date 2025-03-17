import asyncio
from aiohttp import web
import logging
import json
from .utils import get_free_port, setup_logging

class MCPServer:
    def __init__(self):
        self.resources = {}
        self.endpoint = None
        self.app = web.Application()
        self.app.router.add_get("/mcp", self.handle_websocket)
        self.runner = None
        self.site = None
        setup_logging()

    def register_resource(self, name: str, func):
        self.resources[name] = func
        logging.info(f"Registered resource: {name}")

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=30)  # Added heartbeat
        await ws.prepare(request)
        logging.info("WebSocket connection opened")
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        resource = data.get("resource")
                        args = data.get("args", [])
                        kwargs = data.get("kwargs", {})
                        stream = data.get("stream", False)
                        
                        logging.info(f"Received request for resource: {resource} with args: {args}")
                        
                        if resource in self.resources:
                            if stream:
                                # Handle streaming response
                                async for chunk in self.resources[resource](*args, **kwargs):
                                    await ws.send_json({"chunk": chunk})
                                await ws.send_json({"done": True})
                            else:
                                result = await self.resources[resource](*args, **kwargs)
                                await ws.send_json({"result": result})
                        else:
                            logging.warning(f"Resource {resource} not found")
                            await ws.send_json({"error": f"Resource {resource} not found"})
                    except Exception as e:
                        logging.error(f"WebSocket error: {e}")
                        await ws.send_json({"error": str(e)})
                elif msg.type == web.WSMsgType.ERROR:
                    logging.error(f"WebSocket connection closed with exception {ws.exception()}")
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
        
        logging.info("WebSocket connection closed")
        return ws

    def start_local(self):
        if not self.endpoint:
            port = get_free_port()
            self.endpoint = f"ws://localhost:{port}"
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If running in an existing loop (e.g., Flask debug mode), use create_task
                asyncio.ensure_future(self._start_server())
            else:
                # Otherwise, run a new loop
                loop.run_until_complete(self._start_server())
            logging.info(f"Server started at {self.endpoint}")

    def start_standalone(self):
        port = get_free_port()
        self.endpoint = f"ws://localhost:{port}"
        # Use a dedicated loop for standalone mode
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._start_server())
        logging.info(f"Server started in standalone mode at {self.endpoint}")
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            loop.run_until_complete(self._shutdown())
            loop.close()

    def start_cloud(self):
        self.endpoint = "wss://aipaze-cloud.example.com"
        logging.info(f"Using cloud endpoint: {self.endpoint}")

    async def _start_server(self):
        if self.runner is None:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            host = "localhost"
            port = int(self.endpoint.split(":")[-1])
            self.site = web.TCPSite(self.runner, host, port)
            await self.site.start()
            logging.info(f"Server started at {self.endpoint}")

    async def _shutdown(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()