#!/bin/bash

# Start Ollama server in background
ollama serve &

# Give server time to start
sleep 5

# Pull models
ollama pull llama3.1
ollama pull dolphin3
ollama pull llava-llama3

# Wait until all models are available
echo "Waiting for models to become available..."

function models_ready {
  ollama list | grep -q "llama3.1" && \
  ollama list | grep -q "dolphin3" && \
  ollama list | grep -q "llava-llama3"
}

until models_ready; do
  echo "Models not ready yet. Retrying in 3 seconds..."
  sleep 3
done

echo "✅ All models are now available and ready for use."

# Keep the container alive
tail -f /dev/null
