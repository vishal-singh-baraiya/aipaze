import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Set
import logging

class WorkflowStep:
    """
    Represents a step in a workflow.
    """
    def __init__(self, name: str, tool: Optional[str] = None, input_spec: Any = None, depends_on: Optional[List[str]] = None):
        """
        Initialize workflow step.
        
        Args:
            name: Step name
            tool: Tool or resource to use
            input_spec: Input specification (can be a reference to previous step output)
            depends_on: List of step names this step depends on
        """
        self.name = name
        self.tool = tool
        self.input_spec = input_spec
        self.depends_on = depends_on or []
    
    def __repr__(self):
        return f"WorkflowStep(name={self.name}, tool={self.tool})"

class Workflow:
    """
    Workflow for chaining multiple tools or resources.
    """
    def __init__(self, name: str = "workflow"):
        """
        Initialize workflow.
        
        Args:
            name: Workflow name
        """
        self.name = name
        self.steps: Dict[str, WorkflowStep] = {}
        self.results: Dict[str, Any] = {}
        self.registry = None
        self.client = None
    
    def add_step(self, name: str, tool: Optional[str] = None, input_spec: Any = None, depends_on: Optional[List[str]] = None) -> 'Workflow':
        """
        Add a step to the workflow.
        
        Args:
            name: Step name
            tool: Tool or resource to use
            input_spec: Input specification (can be a reference to previous step output)
            depends_on: List of step names this step depends on
            
        Returns:
            Self for chaining
        """
        self.steps[name] = WorkflowStep(name, tool, input_spec, depends_on)
        return self
    
    def set_client(self, client: Any) -> 'Workflow':
        """
        Set the client to use for executing tools.
        
        Args:
            client: MCPClient instance
            
        Returns:
            Self for chaining
        """
        self.client = client
        return self
    
    async def execute(self, initial_input: Any = None) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Args:
            initial_input: Initial input to the workflow
            
        Returns:
            Dictionary mapping step names to results
        """
        if not self.steps:
            logging.warning("Workflow has no steps to execute")
            return {}
        
        # Reset results
        self.results = {"initial": initial_input}
        
        # Get dependency graph
        graph = self._build_dependency_graph()
        
        # Execute steps in dependency order
        executed: Set[str] = set()
        while len(executed) < len(self.steps):
            # Find steps that can be executed (all dependencies satisfied)
            ready_steps = []
            for name, step in self.steps.items():
                if name in executed:
                    continue
                
                if all(dep in executed for dep in step.depends_on):
                    ready_steps.append(name)
            
            if not ready_steps:
                raise ValueError("Circular dependency detected in workflow")
            
            # Execute ready steps in parallel
            tasks = []
            for name in ready_steps:
                tasks.append(self._execute_step(name))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            # Store results
            for name, result in zip(ready_steps, results):
                self.results[name] = result
                executed.add(name)
        
        return self.results
    
    def sync_execute(self, initial_input: Any = None) -> Dict[str, Any]:
        """
        Synchronous version of execute.
        
        Args:
            initial_input: Initial input to the workflow
            
        Returns:
            Dictionary mapping step names to results
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.execute(initial_input))
        finally:
            loop.close()
    
    async def _execute_step(self, step_name: str) -> Any:
        """
        Execute a single step.
        
        Args:
            step_name: Name of step to execute
            
        Returns:
            Step result
        """
        step = self.steps[step_name]
        
        # Resolve input
        input_value = self._resolve_input(step.input_spec)
        
        if not step.tool:
            # If no tool specified, just pass through the input
            return input_value
        
        # Execute tool
        if self.client:
            # Use client to execute tool or resource
            try:
                if isinstance(input_value, dict):
                    result = await self.client.query(step.tool, **input_value)
                else:
                    result = await self.client.query(step.tool, input_value)
                return result
            except Exception as e:
                logging.error(f"Error executing step {step_name}: {str(e)}")
                raise
        else:
            # Try to use tool registry directly
            from .tools import get_tool_registry
            registry = get_tool_registry()
            
            try:
                if isinstance(input_value, dict):
                    return registry.execute_tool(step.tool, **input_value)
                else:
                    return registry.execute_tool(step.tool, input=input_value)
            except Exception as e:
                logging.error(f"Error executing step {step_name}: {str(e)}")
                raise
    
    def _resolve_input(self, input_spec: Any) -> Any:
        """
        Resolve input specification to actual value.
        
        Args:
            input_spec: Input specification
            
        Returns:
            Resolved input value
        """
        if isinstance(input_spec, str) and input_spec in self.results:
            # Reference to previous step result
            return self.results[input_spec]
        elif isinstance(input_spec, dict):
            # Dictionary with references to previous results
            resolved = {}
            for key, value in input_spec.items():
                if isinstance(value, str) and value in self.results:
                    resolved[key] = self.results[value]
                else:
                    resolved[key] = value
            return resolved
        elif isinstance(input_spec, list):
            # List with possible references
            resolved = []
            for item in input_spec:
                if isinstance(item, str) and item in self.results:
                    resolved.append(self.results[item])
                else:
                    resolved.append(item)
            return resolved
        else:
            # Direct value
            return input_spec
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Build dependency graph.
        
        Returns:
            Dictionary mapping step names to lists of dependent steps
        """
        graph = {name: [] for name in self.steps}
        
        for name, step in self.steps.items():
            for dep in step.depends_on:
                if dep not in self.steps:
                    raise ValueError(f"Step {name} depends on non-existent step {dep}")
                graph[dep].append(name)
        
        return graph
    
    def visualize(self) -> str:
        """
        Generate a simple text visualization of the workflow.
        
        Returns:
            Text representation of workflow
        """
        result = [f"Workflow: {self.name}"]
        result.append("Steps:")
        
        for name, step in self.steps.items():
            deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
            tool = f" using {step.tool}" if step.tool else ""
            result.append(f"  - {name}{tool}{deps}")
        
        return "\n".join(result)