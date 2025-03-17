from typing import List, Dict, Any, Optional

class Memory:
    """
    Memory system for storing conversation history and context.
    """
    def __init__(self, max_entries: int = 10):
        """
        Initialize memory system.
        
        Args:
            max_entries: Maximum number of message entries to store
        """
        self.messages: List[Dict[str, str]] = []
        self.max_entries = max_entries
        self.metadata: Dict[str, Any] = {}
    
    def add(self, role: str, content: str):
        """
        Add a message to memory.
        
        Args:
            role: The role of the message sender (e.g., "user", "assistant", "system")
            content: The content of the message
        """
        self.messages.append({"role": role, "content": content})
        # Remove oldest messages if we exceed max entries
        if len(self.messages) > self.max_entries:
            self.messages.pop(0)
    
    def get_context(self) -> List[Dict[str, str]]:
        """
        Get the current conversation context.
        
        Returns:
            List of message dictionaries with role and content
        """
        return self.messages
    
    def clear(self):
        """Clear all messages from memory."""
        self.messages = []
    
    def set_metadata(self, key: str, value: Any):
        """
        Store metadata associated with this conversation.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Retrieve metadata associated with this conversation.
        
        Args:
            key: Metadata key
            default: Default value to return if key not found
            
        Returns:
            The stored metadata value or default
        """
        return self.metadata.get(key, default)
    
    def summarize(self, llm_callable=None) -> str:
        """
        Get a summary of the conversation.
        
        Args:
            llm_callable: Optional callable to generate a summary
            
        Returns:
            Summary string
        """
        if not self.messages:
            return "No conversation history."
        
        if llm_callable:
            prompt = "Summarize the following conversation:\n\n"
            for msg in self.messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            return llm_callable(prompt)
        
        # Simple summary if no LLM provided
        user_msgs = sum(1 for msg in self.messages if msg["role"] == "user")
        assistant_msgs = sum(1 for msg in self.messages if msg["role"] == "assistant")
        return f"Conversation with {user_msgs} user messages and {assistant_msgs} assistant responses."