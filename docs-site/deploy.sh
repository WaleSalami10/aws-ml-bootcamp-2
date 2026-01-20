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
    exit 1
fi

# Display information
echo "✅ Documentation site found"
echo "📁 Serving directory: $(pwd)"
echo ""
echo "📡 Starting local development server..."
echo "🌐 Your documentation site will be available at:"
echo "   👉 http://localhost:8080"
echo ""
echo "💡 Tips:"
echo "   - Press Ctrl+C to stop the server"
echo "   - Open http://localhost:8080 in your browser"
echo "   - The site will auto-reload on file changes"
echo ""
echo "🚀 Starting server..."
echo ""

# Start Python HTTP server
python3 -m http.server 8080