"""
API Client for vLLM server communication.
Handles OpenAI-compatible chat completions API.
"""

import requests
import time
from typing import List, Dict, Optional, Any
import json


class VLLMClient:
    """Client for interacting with vLLM server."""
    
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "openai/gpt-oss-20b"):
        self.base_url = base_url
        self.model = model
        self.chat_endpoint = f"{base_url}/v1/chat/completions"
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 32000,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a completion from the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            API response dict or None if failed
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.chat_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=300  # 5 minute timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    print(f"Request failed after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    def extract_content(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Extract ONLY the reasoning/chain-of-thought from an API response.
        We only want to track reasoning for counterfactual probing, not the final content.
        """
        if not response:
            return None
        
        choices = response.get("choices", [])
        if not choices:
            return None
        
        message = choices[0].get("message", {})
        # Extract reasoning - this is what we want to track sentence-by-sentence
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        return reasoning
    
    def test_connection(self) -> bool:
        """Test if the vLLM server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            # Try models endpoint as fallback
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=5)
                return response.status_code == 200
            except requests.exceptions.RequestException:
                return False
    
    def get_models(self) -> Optional[List[Dict]]:
        """Get list of available models."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Failed to get models: {e}")
            return None