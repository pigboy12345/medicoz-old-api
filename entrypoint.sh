#!/bin/bash
# entrypoint.sh

# Start ngrok in the background, logging to /app/ngrok.log
ngrok http 8000 --authtoken "$NGROK_AUTH_TOKEN" --log=stdout > /app/ngrok.log 2>&1 &

# Wait for ngrok to initialize
sleep 5

# Extract the ngrok public URL
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o "https://[a-z0-9-]*\.ngrok-free\.app" || echo "Failed to get URL")
echo "Ngrok public URL: $NGROK_URL"

# Start the FastAPI application
uvicorn app:app --host 0.0.0.0 --port 8000

# curl -X POST "https://b87c-3-226-232-170.ngrok-free.app/query" \
#   -H "Content-Type: application/json" \
#   -d '{"question": "How to Prevention of cerebrovascular disease?"}'

