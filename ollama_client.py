import requests
import json
import logging
import subprocess
import time
from typing import Dict, Any, Optional, Generator, Callable
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with the Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", max_retries: int = 3,
                 timeout: int = 30, context_window: int = 8192):
        """Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API. Defaults to http://localhost:11434.
            max_retries: Maximum number of retry attempts for failed requests.
            timeout: Request timeout in seconds.
            context_window: Maximum context window size in tokens.
        """
        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.timeout = timeout
        self.context_window = context_window
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None,
                stream: bool = False, callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Generate a response from the model.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt for the model
            options: Optional dictionary of generation options (temperature, max_tokens, etc.)
            stream: Whether to stream the response
            callback: Optional callback function for streaming responses
            
        Returns:
            Dictionary containing the model's response or a generator if streaming
        """
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if options:
            # Ensure max_tokens doesn't exceed context window
            if "max_tokens" in options:
                options["max_tokens"] = min(options["max_tokens"], self.context_window)
            data.update(options)
            
        try:
            response = requests.post(url, json=data, stream=stream, timeout=self.timeout)
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response, callback)
            
            # Handle non-streaming response
            try:
                result = response.json()
                if isinstance(result, str):
                    return {
                        "response": result,
                        "model": model,
                        "created_at": "",
                        "done": True
                    }
                elif isinstance(result, dict):
                    return {
                        "response": result.get("response", ""),
                        "model": result.get("model", model),
                        "created_at": result.get("created_at", ""),
                        "done": result.get("done", True)
                    }
                else:
                    return {
                        "response": str(result),
                        "model": model,
                        "created_at": "",
                        "done": True
                    }
            except json.JSONDecodeError:
                # Handle plain text response
                return {
                    "response": response.text,
                    "model": model,
                    "created_at": "",
                    "done": True
                }
                    
        except requests.exceptions.Timeout:
            logger.error("Request to Ollama API timed out")
            return {"response": "", "error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Ollama API failed: {e}")
            return {"response": "", "error": str(e)}
            
    def _handle_streaming_response(self, response: requests.Response,
                                 callback: Optional[Callable[[str], None]] = None) -> Generator[Dict[str, Any], None, None]:
        """Handle streaming response from the Ollama API."""
        buffer = []
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    # Try to parse as JSON first
                    result = json.loads(line)
                    if isinstance(result, dict):
                        # Extract response text from dictionary
                        response_text = result.get("response", "")
                        if response_text:
                            if callback:
                                callback(response_text)
                            buffer.append(response_text)
                            
                        yield {
                            "response": response_text,
                            "model": result.get("model", ""),
                            "created_at": result.get("created_at", ""),
                            "done": result.get("done", False)
                        }
                    else:
                        # Handle non-dict JSON response
                        text = str(result)
                        if callback:
                            callback(text)
                        buffer.append(text)
                        yield {
                            "response": text,
                            "model": "",
                            "created_at": "",
                            "done": False
                        }
                except json.JSONDecodeError:
                    # Handle plain text response
                    text = line.decode('utf-8')
                    if callback:
                        callback(text)
                    buffer.append(text)
                    yield {
                        "response": text,
                        "model": "",
                        "created_at": "",
                        "done": False
                    }
                    
            # Yield final chunk with complete response
            final_text = "".join(buffer)
            yield {
                "response": final_text,
                "model": "",
                "created_at": "",
                "done": True
            }
                
        except Exception as e:
            logger.error(f"Error processing streaming response: {e}")
            yield {
                "response": "",
                "error": str(e),
                "done": True
            }
            
    def list_models(self) -> Dict[str, Any]:
        """Get list of available models.
        
        Returns:
            Dictionary containing list of models
        """
        url = f"{self.base_url}/api/tags"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            try:
                result = response.json()
                # Handle both array and object responses
                if isinstance(result, list):
                    return {"models": result}
                elif isinstance(result, dict):
                    return {"models": result.get("models", [])}
                else:
                    return {"models": []}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"models": [], "error": "Failed to parse response"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Ollama API failed: {e}")
            return {"models": [], "error": str(e)}
            
    def pull_model(self, model: str) -> Dict[str, Any]:
        """Pull a model from Ollama.
        
        Args:
            model: Name of the model to pull
            
        Returns:
            Dictionary containing pull status
        """
        url = f"{self.base_url}/api/pull"
        
        try:
            response = requests.post(url, json={"name": model})
            response.raise_for_status()
            
            try:
                return response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"status": "error", "error": "Failed to parse response"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Ollama API failed: {e}")
            return {"status": "error", "error": str(e)}

    def create_model(self, model_name: str, modelfile_path: str) -> Dict[str, Any]:
        """Create a new model using a modelfile.
        
        Args:
            model_name: Name for the new model
            modelfile_path: Path to the modelfile
            
        Returns:
            Dictionary containing creation status
        """
        try:
            # Use subprocess to run ollama create command
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", modelfile_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Handle both string and JSON responses
            try:
                output = json.loads(result.stdout)
                return {"status": "success", "message": output}
            except json.JSONDecodeError:
                return {"status": "success", "message": result.stdout}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create model: {e.stderr}")
            return {"status": "error", "error": e.stderr}
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            return {"status": "error", "error": str(e)}

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
        """
        try:
            # Use subprocess to run ollama show command
            result = subprocess.run(
                ["ollama", "show", model_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output to extract model info
            info = {}
            for line in result.stdout.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    info[key.strip()] = value.strip()
                    
            return {"status": "success", "info": info}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get model info: {e.stderr}")
            return {"status": "error", "error": e.stderr}
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"status": "error", "error": str(e)}

# Create a global instance
ollama = OllamaClient()
