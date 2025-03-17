import os
import re
from typing import Dict, Any, List, Optional
import logging

class PromptTemplate:
    """
    Template for generating prompts with variable substitution.
    """
    def __init__(self, template_string: str):
        """
        Initialize with a template string.
        
        Args:
            template_string: Template string with {variable} placeholders
        """
        self.template = template_string
        self._validate_template()
    
    def _validate_template(self):
        """Validate that the template string has valid placeholders."""
        placeholders = re.findall(r'{(\w+)}', self.template)
        if not placeholders:
            # Warning rather than error to allow static templates
            logging.warning("Template has no variable placeholders.")
    
    def format(self, **kwargs) -> str:
        """
        Format the template by replacing variables with values.
        
        Args:
            **kwargs: Keyword arguments for template variables
            
        Returns:
            Formatted string with variables replaced
            
        Raises:
            ValueError: If a required variable is missing
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
    
    def get_variables(self) -> List[str]:
        """
        Get list of variable names in the template.
        
        Returns:
            List of variable names
        """
        return re.findall(r'{(\w+)}', self.template)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'PromptTemplate':
        """
        Create a template from a file.
        
        Args:
            filepath: Path to template file
            
        Returns:
            PromptTemplate instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Template file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            template_string = f.read()
        return cls(template_string)

class ChatTemplate:
    """
    Template for generating multi-turn chat prompts.
    """
    def __init__(self, system_template: Optional[str] = None):
        """
        Initialize chat template.
        
        Args:
            system_template: Optional system message template
        """
        self.system_template = system_template
        self.message_templates: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content_template: str, variables: Optional[Dict[str, Any]] = None):
        """
        Add a message template to the chat.
        
        Args:
            role: Message role (e.g., "user", "assistant")
            content_template: Content template string
            variables: Optional default variables
        """
        self.message_templates.append({
            "role": role,
            "template": content_template,
            "variables": variables or {}
        })
    
    def format(self, variables: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Format all messages in the chat template.
        
        Args:
            variables: Variables to substitute in templates
            
        Returns:
            List of formatted messages
        """
        variables = variables or {}
        messages = []
        
        # Add system message if present
        if self.system_template:
            try:
                system_content = self.system_template.format(**variables)
                messages.append({"role": "system", "content": system_content})
            except KeyError as e:
                raise ValueError(f"Missing variable for system message: {e}")
        
        # Add other messages
        for msg_template in self.message_templates:
            # Combine default variables with provided variables
            msg_vars = {**msg_template.get("variables", {}), **variables}
            try:
                content = msg_template["template"].format(**msg_vars)
                messages.append({
                    "role": msg_template["role"],
                    "content": content
                })
            except KeyError as e:
                raise ValueError(f"Missing variable for {msg_template['role']} message: {e}")
        
        return messages