#!/bin/bash

# Update package lists
echo "Updating package lists..."
apt-get update

# Install system packages if packages.txt exists
if [ -f "packages.txt" ]; then
    echo "Installing system packages from packages.txt..."
    xargs apt-get install -y < packages.txt
else
    echo "packages.txt not found, skipping system package installation."
fi

# Install Python packages if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing Python packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping Python package installation."
fi

# Install cloudflared if not already installed
if ! command -v cloudflared &> /dev/null; then
    echo "Installing cloudflared..."
    wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
    mv cloudflared /usr/local/bin/
fi

# Run the Python application in the background
echo "Starting the application..."
nohup python app.py --port 7860 > app.log 2>&1 &

# Start cloudflared tunnel in the background
echo "Starting cloudflared tunnel..."
nohup cloudflared tunnel --url http://localhost:7860 > cloudflared.log 2>&1 &

echo "Application and cloudflared tunnel are running in the background."
echo "Check app.log and cloudflared.log for logs."
echo "To stop the application, run: pkill -f 'python app.py' && pkill -f 'cloudflared'"
