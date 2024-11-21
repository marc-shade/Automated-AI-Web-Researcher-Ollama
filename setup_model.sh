#!/bin/bash

echo "Setting up Phi model for AI Web Researcher..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it first from https://ollama.ai"
    exit 1
fi

# First, pull the base phi model
echo "Pulling base Phi model..."
ollama pull phi

# Create custom Modelfile for phi
cat << 'EOF' > Modelfile
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

# Add example research format
TEMPLATE """{{ .System }}

USER: Research this topic: {{ .Prompt }}

A: I'll help you research this topic thoroughly. Let me break it down into key areas:

1. [First research area]
Priority: [1-5]
Search queries:
- [Specific search query 1]
- [Specific search query 2]

2. [Second research area]
Priority: [1-5]
Search queries:
- [Specific search query 1]
- [Specific search query 2]

[Continue with research findings and analysis...]"""
EOF

# Create the model using Ollama
echo "Creating custom researcher model..."
ollama create researcher -f Modelfile

# Verify the model was created successfully
if ollama list | grep -q "researcher"; then
    echo "✅ Researcher model created successfully!"
    echo "You can now run the application with: streamlit run app.py"
else
    echo "❌ Failed to create the researcher model"
    exit 1
fi
