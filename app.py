import streamlit as st
import os
from pathlib import Path
from model_manager import ModelManager
from research_manager import ResearchManager
from dotenv import load_dotenv
import logging
import requests
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

def initialize_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = {
            "model_name": "researcher",  # Default to our custom phi model
            "temperature": 0.7,
            "max_tokens": 4000,
            "search_depth": 3,
            "max_sources": 10,
        }
    if 'research_results' not in st.session_state:
        st.session_state.research_results = []

def create_model():
    """Create a new model using Ollama."""
    try:
        # First pull the base model
        with st.spinner("Pulling base Phi model..."):
            response = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": "phi"})
            if response.status_code != 200:
                st.error("Failed to pull base model")
                return False

        # Create Modelfile content
        modelfile = '''
FROM phi

# Set a custom system message
SYSTEM """You are a highly capable research assistant with expertise in gathering, analyzing, and synthesizing information from various sources. Your goal is to help users conduct thorough research by:
1. Breaking down complex topics into key research areas
2. Generating targeted search queries
3. Analyzing and summarizing findings
4. Identifying patterns and insights
5. Providing actionable conclusions

Always strive to be thorough, accurate, and objective in your research."""

# Configure model parameters for research tasks
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 128000
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
'''

        # Create the model
        with st.spinner("Creating researcher model..."):
            response = requests.post(
                f"{OLLAMA_URL}/api/create",
                json={"name": "researcher", "modelfile": modelfile}
            )
            if response.status_code != 200:
                st.error("Failed to create model")
                return False

        st.success("Model created successfully!")
        return True
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return False

def delete_model(model_name):
    """Delete a model from Ollama."""
    try:
        response = requests.delete(f"{OLLAMA_URL}/api/delete", json={"name": model_name})
        if response.status_code == 200:
            st.success(f"Model {model_name} deleted successfully!")
            return True
        else:
            st.error(f"Failed to delete model {model_name}")
            return False
    except Exception as e:
        st.error(f"Error deleting model: {str(e)}")
        return False

def verify_model():
    """Verify that the required model is available."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            if not any(model.startswith("researcher") for model in models):
                col1, col2 = st.columns(2)
                with col1:
                    st.error("Required model 'researcher' not found.")
                with col2:
                    if st.button("Create Model"):
                        create_model()
                return False
            return True
        else:
            st.error("Failed to connect to Ollama API. Please ensure Ollama is running.")
            return False
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return False

def display_model_management():
    """Display model management section in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🤖 Model Management")
    
    # Get available models
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            
            # Create model button
            if st.sidebar.button("Create Researcher Model"):
                create_model()
            
            # Delete model section
            if models:
                st.sidebar.markdown("### Delete Model")
                model_to_delete = st.sidebar.selectbox(
                    "Select model to delete",
                    options=models
                )
                if st.sidebar.button("Delete Selected Model"):
                    delete_model(model_to_delete)
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")

def main():
    st.set_page_config(
        page_title="AI Web Researcher",
        page_icon="🔍",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Verify model before proceeding
    model_available = verify_model()
    
    # Display model management in sidebar
    display_model_management()
    
    if not model_available:
        return
        
    # Sidebar for configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Model selection (but default to researcher)
        available_models = ["researcher"]  # We'll stick with our custom model
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=0,
            help="Using the custom researcher model based on phi"
        )
        st.session_state.config["model_name"] = selected_model
        
        # Other configuration options
        st.session_state.config["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config["temperature"],
            step=0.1,
            help="Higher values make output more creative, lower values more focused"
        )
        
        st.session_state.config["max_tokens"] = st.slider(
            "Max Tokens",
            min_value=1000,
            max_value=8000,
            value=st.session_state.config["max_tokens"],
            step=1000,
            help="Maximum number of tokens in the response"
        )
        
        st.session_state.config["search_depth"] = st.slider(
            "Search Depth",
            min_value=1,
            max_value=5,
            value=st.session_state.config["search_depth"],
            help="Number of search iterations per area"
        )
        
        st.session_state.config["max_sources"] = st.slider(
            "Max Sources",
            min_value=5,
            max_value=20,
            value=st.session_state.config["max_sources"],
            help="Maximum number of sources to consider per search"
        )
    
    # Main content area
    st.title("🔍 AI Web Researcher")
    st.markdown("""
    Enter a research topic and I'll help you:
    1. Break it down into key research areas
    2. Generate targeted search queries
    3. Analyze and summarize findings
    4. Identify patterns and insights
    5. Provide actionable conclusions
    """)
    
    # Research input
    query = st.text_area("Enter your research topic:", height=100, placeholder="e.g., What are the latest developments in quantum computing?")
    
    # Create containers for different sections
    progress_container = st.empty()
    research_plan_container = st.container()
    research_areas_container = st.container()
    final_report_container = st.container()
    
    if st.button("Start Research", type="primary", disabled=not query):
        if not query:
            st.warning("Please enter a research topic")
            return
            
        research_manager = None
        try:
            # Create research manager with current settings
            research_manager = ResearchManager(
                model_name=st.session_state.config["model_name"],
                temperature=st.session_state.config["temperature"],
                max_tokens=st.session_state.config["max_tokens"]
            )
            
            # Run research with progress tracking
            with st.spinner("Researching..."):
                def update_progress(message: str, progress: float):
                    progress_container.progress(progress)
                    
                    # Show research plan
                    if "Research Plan" in message:
                        with research_plan_container:
                            st.markdown("## Research Plan")
                            st.markdown(message)
                    # Show area progress
                    elif "Researching area" in message:
                        with research_areas_container:
                            st.markdown(message)
                    # Show final summary
                    elif "Research complete" in message:
                        with final_report_container:
                            st.success(message)
                    else:
                        st.markdown(message)
                
                # Run the research
                for result in research_manager.research(
                    query=query,
                    search_depth=st.session_state.config["search_depth"],
                    max_sources=st.session_state.config["max_sources"],
                    progress_callback=update_progress
                ):
                    if isinstance(result, dict):
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.markdown(result.get("response", ""))
                    else:
                        st.markdown(result)
                
        except Exception as e:
            st.error(f"Error during research: {str(e)}")
            logger.exception("Research error")
        finally:
            if research_manager:
                research_manager.cleanup()

if __name__ == "__main__":
    main()
