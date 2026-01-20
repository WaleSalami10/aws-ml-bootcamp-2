#!/bin/bash
# Alternative simple server script

cd "$(dirname "$0")"
PORT=${1:-8080}
echo "Starting server on port $PORT..."
python3 -m http.server $PORT
