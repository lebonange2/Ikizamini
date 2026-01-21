#!/bin/bash
# IKIZAMINI Setup Script
# This script sets up the application with zstd library and Ollama with qwen:30b model

set -e  # Exit on error

echo "=========================================="
echo "IKIZAMINI Application Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check if running as root (for package installation)
check_root() {
    if [ "$EUID" -eq 0 ]; then 
        print_info "Running as root - will install system packages"
    else
        print_info "Not running as root - may need sudo for package installation"
    fi
}

# Detect package manager
detect_package_manager() {
    if command -v apt-get &> /dev/null; then
        PKG_MANAGER="apt-get"
        INSTALL_CMD="sudo apt-get install -y"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
        INSTALL_CMD="sudo yum install -y"
    elif command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
        INSTALL_CMD="sudo dnf install -y"
    elif command -v pacman &> /dev/null; then
        PKG_MANAGER="pacman"
        INSTALL_CMD="sudo pacman -S --noconfirm"
    elif command -v brew &> /dev/null; then
        PKG_MANAGER="brew"
        INSTALL_CMD="brew install"
    else
        print_error "Could not detect package manager. Please install zstd manually."
        exit 1
    fi
    print_success "Detected package manager: $PKG_MANAGER"
}

# Install zstd library
install_zstd() {
    echo ""
    echo "Step 1: Installing zstd library..."
    
    if command -v zstd &> /dev/null; then
        print_success "zstd is already installed: $(zstd --version | head -n1)"
    else
        print_info "Installing zstd..."
        detect_package_manager
        
        if [ "$PKG_MANAGER" = "brew" ]; then
            $INSTALL_CMD zstd
        else
            $INSTALL_CMD zstd libzstd-dev 2>/dev/null || $INSTALL_CMD zstd zstd-devel 2>/dev/null || $INSTALL_CMD zstd
        fi
        
        if command -v zstd &> /dev/null; then
            print_success "zstd installed successfully: $(zstd --version | head -n1)"
        else
            print_error "Failed to install zstd. Please install it manually."
            exit 1
        fi
    fi
}

# Check Python installation
check_python() {
    echo ""
    echo "Step 2: Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python found: $PYTHON_VERSION"
    else
        print_error "Python3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check if pip is available
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found: $(pip3 --version)"
    else
        print_info "pip3 not found. Installing pip..."
        if [ "$PKG_MANAGER" = "brew" ]; then
            $INSTALL_CMD python3-pip || python3 -m ensurepip --upgrade
        else
            $INSTALL_CMD python3-pip
        fi
    fi
}

# Install Python dependencies
install_python_dependencies() {
    echo ""
    echo "Step 3: Installing Python dependencies..."
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        print_info "Virtual environment found. Activating..."
        source venv/bin/activate
    else
        print_info "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        print_success "Virtual environment created"
    fi
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install required packages
    print_info "Installing Python packages..."
    pip install flask requests jsonschema openai --quiet
    
    print_success "Python dependencies installed"
}

# Install Ollama
install_ollama() {
    echo ""
    echo "Step 4: Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "installed")
        print_success "Ollama is already installed: $OLLAMA_VERSION"
    else
        print_info "Installing Ollama..."
        
        # Download and install Ollama
        curl -fsSL https://ollama.com/install.sh | sh
        
        if command -v ollama &> /dev/null; then
            print_success "Ollama installed successfully"
        else
            print_error "Failed to install Ollama. Please install it manually from https://ollama.com"
            exit 1
        fi
    fi
    
    # Start Ollama service if not running
    print_info "Checking Ollama service..."
    if ! pgrep -x "ollama" > /dev/null; then
        print_info "Starting Ollama service..."
        ollama serve > /dev/null 2>&1 &
        sleep 3
        
        # Wait for Ollama to be ready
        for i in {1..10}; do
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                print_success "Ollama service is running"
                break
            fi
            sleep 1
        done
        
        if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_error "Ollama service failed to start. Please start it manually: ollama serve"
        fi
    else
        print_success "Ollama service is already running"
    fi
}

# Pull qwen:30b model
pull_qwen_model() {
    echo ""
    echo "Step 5: Pulling qwen:30b model..."
    
    # Check if model is already pulled
    if ollama list 2>/dev/null | grep -q "qwen:30b"; then
        print_success "qwen:30b model is already available"
    else
        print_info "Pulling qwen:30b model (this may take a while and requires significant disk space)..."
        ollama pull qwen:30b
        
        if ollama list 2>/dev/null | grep -q "qwen:30b"; then
            print_success "qwen:30b model pulled successfully"
        else
            print_error "Failed to pull qwen:30b model. Please check your internet connection and disk space."
            exit 1
        fi
    fi
}

# Verify installation
verify_installation() {
    echo ""
    echo "Step 6: Verifying installation..."
    
    # Check zstd
    if command -v zstd &> /dev/null; then
        print_success "zstd: OK"
    else
        print_error "zstd: NOT FOUND"
    fi
    
    # Check Python packages
    if python3 -c "import flask, requests, jsonschema, openai" 2>/dev/null; then
        print_success "Python packages: OK"
    else
        print_error "Python packages: MISSING"
    fi
    
    # Check Ollama
    if command -v ollama &> /dev/null; then
        print_success "Ollama: OK"
    else
        print_error "Ollama: NOT FOUND"
    fi
    
    # Check qwen:30b model
    if ollama list 2>/dev/null | grep -q "qwen:30b"; then
        print_success "qwen:30b model: OK"
    else
        print_error "qwen:30b model: NOT FOUND"
    fi
    
    # Check Ollama service
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama service: RUNNING"
    else
        print_error "Ollama service: NOT RUNNING (start with: ollama serve)"
    fi
}

# Main execution
main() {
    check_root
    install_zstd
    check_python
    install_python_dependencies
    install_ollama
    pull_qwen_model
    verify_installation
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo "=========================================="
    echo ""
    echo "To use the application:"
    echo "  1. Activate the virtual environment:"
    echo "     source venv/bin/activate"
    echo ""
    echo "  2. Run the local UI:"
    echo "     python3 ikizamini_local.py"
    echo ""
    echo "  3. Or run the CLI version:"
    echo "     python3 ikizamini_app.py --input Uru.txt --output ikizamini.txt --output-dir \"output/MATHEMATICS/1.1 Algebra and Trigonometry\""
    echo ""
    echo "Note: Make sure Ollama is running (ollama serve) before using the application."
    echo ""
}

# Run main function
main
