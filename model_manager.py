import os
import json
import tempfile
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import streamlit as st
from ollama_client import ollama

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages Ollama model operations including listing, creating, and configuring models."""
    
    DEFAULT_CONTEXT_LENGTH = 38000
    RECOMMENDED_MODELS = {
        "mistral-nemo": {
            "base": "mistral-nemo",
            "description": "Mistral Nemo with 128k context (recommended)",
            "default_ctx": 38000
        },
        "mistral": {
            "base": "mistral",
            "description": "Mistral 7B with enhanced capabilities",
            "default_ctx": 32000
        },
        "llama2": {
            "base": "llama2",
            "description": "Meta's Llama 2 model",
            "default_ctx": 32000
        }
    }
    
    @staticmethod
    def get_available_models() -> List[Dict[str, str]]:
        """Get list of available models from Ollama."""
        try:
            response = ollama.list_models()
            if "models" in response:
                logger.info(f"Found {len(response['models'])} available models")
                return response["models"]
            logger.warning("No models found in Ollama response")
            return []
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            return []
    
    @staticmethod
    def validate_model(model_name: str) -> bool:
        """Validate that a model exists and is ready for use.
        
        Args:
            model_name: Name of model to validate
            
        Returns:
            bool: True if model is valid and ready
        """
        try:
            models = ModelManager.get_available_models()
            return any(m["name"] == model_name for m in models)
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {str(e)}")
            return False
            
    @staticmethod
    def create_research_model(base_model: str, ctx_length: int = DEFAULT_CONTEXT_LENGTH) -> Tuple[bool, str]:
        """Create a custom research model with specified context length.
        
        Args:
            base_model: Base model name to use
            ctx_length: Context length for the model
            
        Returns:
            Tuple of (success, message)
        """
        model_name = f"research-{base_model.replace(':', '-')}"
        
        # Validate base model exists
        if not ModelManager.validate_model(base_model):
            return False, f"Base model {base_model} not found"
        
        # Create modelfile
        modelfile_content = f"""
FROM {base_model}

PARAMETER num_ctx {ctx_length}
"""
        
        # Write modelfile to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
            f.write(modelfile_content)
            modelfile_path = f.name
            
        try:
            # Create the model using Ollama CLI
            result = ollama.create_model(model_name, modelfile_path)
            
            if result.get("status") == "success":
                logger.info(f"Successfully created model {model_name}")
                return True, f"Successfully created model {model_name}"
            else:
                error_msg = f"Failed to create model: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Error creating model: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(modelfile_path)
            except Exception as e:
                logger.warning(f"Failed to clean up modelfile: {str(e)}")
                
    @staticmethod
    def render_model_creator():
        """Render the model creation UI in Streamlit."""
        st.subheader("Create Custom Research Model")
        
        # Get available base models
        available_models = ModelManager.get_available_models()
        base_model_names = [model["name"] for model in available_models]
        
        # Add recommended models if not already available
        for model_info in ModelManager.RECOMMENDED_MODELS.values():
            if model_info["base"] not in base_model_names:
                base_model_names.append(model_info["base"])
                
        # Sort and remove duplicates
        base_model_names = sorted(list(set(base_model_names)))
        
        # Model selection
        selected_model = st.selectbox(
            "Select Base Model",
            base_model_names,
            help="Choose a base model to create your research variant"
        )
        
        # Show model description if available
        for model_key, model_info in ModelManager.RECOMMENDED_MODELS.items():
            if model_info["base"] == selected_model:
                st.info(model_info["description"])
                break
        
        # Context length input
        default_ctx = next(
            (info["default_ctx"] for info in ModelManager.RECOMMENDED_MODELS.values() 
             if info["base"] == selected_model),
            ModelManager.DEFAULT_CONTEXT_LENGTH
        )
        
        ctx_length = st.number_input(
            "Context Length",
            min_value=1000,
            max_value=128000,
            value=default_ctx,
            step=1000,
            help="Specify the context length for your model"
        )
        
        # Create model button
        if st.button("Create Research Model"):
            with st.spinner("Creating model..."):
                success, message = ModelManager.create_research_model(selected_model, ctx_length)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
                    
    @staticmethod
    def render_model_selector() -> Optional[str]:
        """Render the model selection UI in Streamlit.
        
        Returns:
            Selected model name or None if no selection
        """
        available_models = ModelManager.get_available_models()
        
        if not available_models:
            st.warning("No models available. Please ensure Ollama is running and has models installed.")
            return None
            
        # Group models into categories
        research_models = []
        recommended_models = []
        other_models = []
        
        for model in available_models:
            model_name = model["name"]
            if model_name.startswith("research-"):
                research_models.append(model)
            elif any(info["base"] == model_name for info in ModelManager.RECOMMENDED_MODELS.values()):
                recommended_models.append(model)
            else:
                other_models.append(model)
                
        # Create tabs for different model categories
        tab1, tab2, tab3 = st.tabs(["Research Models", "Recommended Models", "All Models"])
        
        with tab1:
            if research_models:
                selected = st.selectbox(
                    "Select Research Model",
                    [m["name"] for m in research_models],
                    help="These models are optimized for research tasks"
                )
                if selected:
                    return selected
            else:
                st.info("No research models available. Create one in the Model Creator tab.")
                
        with tab2:
            if recommended_models:
                selected = st.selectbox(
                    "Select Recommended Model",
                    [m["name"] for m in recommended_models],
                    help="Models recommended for research tasks"
                )
                if selected:
                    # Show model description
                    for model_info in ModelManager.RECOMMENDED_MODELS.values():
                        if model_info["base"] == selected:
                            st.info(model_info["description"])
                            break
                    return selected
            else:
                st.info("No recommended models available. Install them using Ollama.")
                
        with tab3:
            if other_models:
                selected = st.selectbox(
                    "Select Model",
                    [m["name"] for m in other_models],
                    help="All available Ollama models"
                )
                if selected:
                    return selected
            else:
                st.info("No other models available.")
                
        return None
