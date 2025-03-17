import time
import logging
from typing import Dict, List, Any, Optional
import threading

class Metrics:
    """
    Metrics collection system for monitoring AIPaze usage.
    """
    def __init__(self):
        """Initialize metrics system."""
        self.start_time = None
        self.requests = 0
        self.token_usage = 0
        self.latencies = []
        self.errors = 0
        self.lock = threading.Lock()
        self.resource_counts = {}
        self.token_by_model = {}
        
    def start_tracking(self):
        """Start tracking metrics."""
        self.start_time = time.time()
        logging.info("Metrics tracking started")
        
    def record_request(self, tokens: int, latency_ms: float, model: str = "default", resource: str = None):
        """
        Record a request.
        
        Args:
            tokens: Number of tokens used
            latency_ms: Latency in milliseconds
            model: Model name
            resource: Resource name if applicable
        """
        with self.lock:
            self.requests += 1
            self.token_usage += tokens
            self.latencies.append(latency_ms)
            
            # Track by model
            if model not in self.token_by_model:
                self.token_by_model[model] = 0
            self.token_by_model[model] += tokens
            
            # Track by resource
            if resource:
                if resource not in self.resource_counts:
                    self.resource_counts[resource] = 0
                self.resource_counts[resource] += 1
    
    def record_error(self, error_type: str = "general"):
        """
        Record an error.
        
        Args:
            error_type: Type of error
        """
        with self.lock:
            self.errors += 1
    
    def get_avg_latency(self) -> float:
        """
        Get average latency.
        
        Returns:
            Average latency in milliseconds
        """
        with self.lock:
            if not self.latencies:
                return 0
            return sum(self.latencies) / len(self.latencies)
    
    def get_total_tokens(self) -> int:
        """
        Get total tokens used.
        
        Returns:
            Total tokens used
        """
        with self.lock:
            return self.token_usage
    
    def get_tokens_by_model(self) -> Dict[str, int]:
        """
        Get tokens used by model.
        
        Returns:
            Dictionary mapping model names to token counts
        """
        with self.lock:
            return self.token_by_model.copy()
    
    def get_request_count(self) -> int:
        """
        Get total request count.
        
        Returns:
            Total number of requests
        """
        with self.lock:
            return self.requests
    
    def get_error_rate(self) -> float:
        """
        Get error rate.
        
        Returns:
            Error rate as a percentage
        """
        with self.lock:
            if not self.requests:
                return 0
            return (self.errors / self.requests) * 100
    
    def get_uptime(self) -> float:
        """
        Get uptime.
        
        Returns:
            Uptime in seconds
        """
        if not self.start_time:
            return 0
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        with self.lock:
            return {
                "requests": self.requests,
                "errors": self.errors,
                "error_rate": self.get_error_rate(),
                "avg_latency_ms": self.get_avg_latency(),
                "total_tokens": self.token_usage,
                "tokens_by_model": self.token_by_model,
                "uptime_seconds": self.get_uptime(),
                "resource_counts": self.resource_counts
            }
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.start_time = time.time()
            self.requests = 0
            self.token_usage = 0
            self.latencies = []
            self.errors = 0
            self.resource_counts = {}
            self.token_by_model = {}