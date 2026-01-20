#!/bin/bash

# AWS ML Bootcamp Documentation Deployment Script

echo "🚀 AWS ML Bootcamp Documentation Server"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -f "index.html" ]; then
    echo "❌ Error: Please run this script from the docs-site directory"
    echo "   Usage: cd docs-site && ./deploy.sh"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    echo "   Please install Python 3: https://www.python.org/downloads/"
    exit 1
fi

# Check if port 8080 is already in use
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8080 is already in use"
    echo "   Attempting to use port 8000 instead..."
    PORT=8000
    # Check if 8000 is also in use
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "⚠️  Port 8000 is also in use"
        echo "   Please stop the process using these ports or specify a different port:"
        echo "   PORT=3000 ./deploy.sh"
        exit 1
    fi
else
    PORT=8080
fi

# Display information
echo "✅ Documentation site found"
echo "📁 Serving directory: $(pwd)"
echo "🐍 Python version: $(python3 --version)"
echo ""
echo "📡 Starting local development server..."
echo "🌐 Your documentation site will be available at:"
echo "   👉 http://localhost:${PORT}"
echo ""
echo "💡 Tips:"
echo "   - Press Ctrl+C to stop the server"
echo "   - Open http://localhost:${PORT} in your browser"
echo "   - Make sure all files are saved before viewing"
echo ""
echo "🚀 Starting server on port ${PORT}..."
echo ""

# Start Python HTTP server
python3 -m http.server ${PORT}
