#!/bin/bash
# Eye Tracker Setup Script
# Installs all dependencies and initializes the application

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Eye Tracker - Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}[1/5] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✓ Python $python_version (OK)${NC}"
else
    echo -e "${RED}✗ Python $python_version (required 3.9+)${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}[2/5] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}[3/5] Installing core dependencies...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    echo -e "${GREEN}✓ Core dependencies installed${NC}"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

# Optional: Install YOLO and MediaPipe
echo -e "${YELLOW}[4/5] Optional: Install enhanced detection?${NC}"
echo -e "This requires downloading large packages (~2GB) and may take 10-20 minutes."
read -p "Install ultralytics (YOLO) and mediapipe for better detection? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Installing ultralytics and mediapipe...${NC}"
    pip install ultralytics mediapipe --no-cache-dir
    echo -e "${GREEN}✓ Enhanced detection packages installed${NC}"
else
    echo -e "${YELLOW}⊘ Skipping enhanced detection (cascade fallback will be used)${NC}"
fi

# Run tests
echo -e "${YELLOW}[5/5] Running component tests...${NC}"
python test_components.py
test_result=$?

if [ $test_result -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Setup Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "To start the Eye Tracker:"
    echo -e "  ${BLUE}source venv/bin/activate${NC}"
    echo -e "  ${BLUE}python main.py${NC}"
    echo ""
    echo -e "For minimized mode:"
    echo -e "  ${BLUE}python main.py --minimized${NC}"
    echo ""
    echo -e "For help:"
    echo -e "  ${BLUE}python main.py --help${NC}"
    echo ""
    echo -e "See ${BLUE}README.md${NC} for detailed usage and configuration."
    echo ""
else
    echo -e "${RED}✗ Some tests failed. Check output above.${NC}"
    exit 1
fi
