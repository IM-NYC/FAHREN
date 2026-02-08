#!/bin/bash
set -euo pipefail

# FAHREN Neural Network Trainer - Installation Script
# Usage: curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/your-repo/FAHREN/main/install.sh | bash

BOLD='\033[1m'
ACCENT='\033[38;2;76;175;80m'
ACCENT_BRIGHT='\033[38;2;102;205;170m'
ACCENT_DIM='\033[38;2;56;142;60m'
INFO='\033[38;2;76;175;80m'
SUCCESS='\033[38;2;76;175;80m'
WARN='\033[38;2;255;176;32m'
ERROR='\033[38;2;226;61;45m'
MUTED='\033[38;2;139;127;119m'
NC='\033[0m'

DEFAULT_TAGLINE="Train neural networks without writing code."

ORIGINAL_PATH="${PATH:-}"
TMPFILES=()
DRY_RUN=0
NO_PROMPT=0
VERBOSE=0
HELP=0
INSTALL_METHOD=""

cleanup_tmpfiles() {
    local f
    for f in "${TMPFILES[@]:-}"; do
        rm -f "$f" 2>/dev/null || true
    done
}
trap cleanup_tmpfiles EXIT

mktempfile() {
    local f
    f="$(mktemp)"
    TMPFILES+=("$f")
    echo "$f"
}

DOWNLOADER=""
detect_downloader() {
    if command -v curl &> /dev/null; then
        DOWNLOADER="curl"
        return 0
    fi
    if command -v wget &> /dev/null; then
        DOWNLOADER="wget"
        return 0
    fi
    echo -e "${ERROR}Error: Missing downloader (curl or wget required)${NC}"
    exit 1
}

download_file() {
    local url="$1"
    local output="$2"
    if [[ -z "$DOWNLOADER" ]]; then
        detect_downloader
    fi
    if [[ "$DOWNLOADER" == "curl" ]]; then
        curl -fsSL --proto '=https' --tlsv1.2 --retry 3 --retry-delay 1 --retry-connrefused -o "$output" "$url"
        return
    fi
    wget -q --https-only --secure-protocol=TLSv1_2 --tries=3 --timeout=20 -O "$output" "$url"
}

TAGLINES=()
TAGLINES+=("Train networks without writing code.")
TAGLINES+=("Deep learning. Simplified.")
TAGLINES+=("255 layers. 65K neurons. No PhD required.")
TAGLINES+=("Your GPU is about to get a workout.")
TAGLINES+=("Less typing. More training.")
TAGLINES+=("Neural networks made simple.")

pick_tagline() {
    local count=${#TAGLINES[@]}
    if [[ "$count" -eq 0 ]]; then
        echo "$DEFAULT_TAGLINE"
        return
    fi
    local idx=$((RANDOM % count))
    echo "${TAGLINES[$idx]}"
}

TAGLINE=$(pick_tagline)

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) DRY_RUN=1; shift ;;
            --verbose) VERBOSE=1; shift ;;
            --no-prompt) NO_PROMPT=1; shift ;;
            --help|-h) HELP=1; shift ;;
            --install-method|--method) INSTALL_METHOD="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
}

print_usage() {
    cat <<EOF
FAHREN installer (macOS + Linux)

Usage:
  curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/your-repo/FAHREN/main/install.sh | bash

Options:
  --install-method, --method git|system   Install from git checkout or system defaults
  --git                                   Shortcut for --install-method git
  --no-prompt                              Disable prompts (required in CI/automation)
  --dry-run                                Print what would happen (no changes)
  --verbose                                Print debug output (set -x)
  --help, -h                               Show this help

Environment variables:
  FAHREN_INSTALL_METHOD=git|system
  FAHREN_NO_PROMPT=1
  FAHREN_DRY_RUN=1
  FAHREN_VERBOSE=1

Examples:
  curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/your-repo/FAHREN/main/install.sh | bash
  curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/your-repo/FAHREN/main/install.sh | bash -s -- --no-prompt
EOF
}

configure_verbose() {
    if [[ "$VERBOSE" != "1" ]]; then
        return 0
    fi
    set -x
}

is_promptable() {
    if [[ "$NO_PROMPT" == "1" ]]; then
        return 1
    fi
    if [[ -r /dev/tty && -w /dev/tty ]]; then
        return 0
    fi
    return 1
}

is_root() {
    [[ "$(id -u)" -eq 0 ]]
}

maybe_sudo() {
    if is_root; then
        if [[ "${1:-}" == "-E" ]]; then
            shift
        fi
        "$@"
    else
        sudo "$@"
    fi
}

require_sudo() {
    if [[ "$VERBOSE" != "linux" ]]; then
        return 0
    fi
    if is_root; then
        return 0
    fi
    if command -v sudo &> /dev/null; then
        return 0
    fi
    echo -e "${ERROR}Error: sudo is required for system installs on Linux${NC}"
    exit 1
}

echo -e "${ACCENT}${BOLD}"
echo """
█████████████████████████████████████
█▄─▄▄─██▀▄─██─█─█▄─▄▄▀█▄─▄▄─█▄─▀█▄─▄█
██─▄████─▀─██─▄─██─▄─▄██─▄█▀██─█▄▀─██
▀▄▄▄▀▀▀▄▄▀▄▄▀▄▀▄▀▄▄▀▄▄▀▄▄▄▄▄▀▄▄▄▀▀▄▄▀
"""
echo -e "${NC}${ACCENT_DIM}  ${TAGLINE}${NC}"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]] || [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    OS="linux"
fi

if [[ "$OS" == "unknown" ]]; then
    echo -e "${ERROR}Error: Unsupported operating system${NC}"
    echo "This installer supports macOS and Linux (including WSL)."
    exit 1
fi

echo -e "${SUCCESS}✓${NC} Detected: $OS"

# Install Homebrew on macOS
install_homebrew() {
    if [[ "$OS" == "macos" ]]; then
        if ! command -v brew &> /dev/null; then
            echo -e "${WARN}→${NC} Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            if [[ -f "/opt/homebrew/bin/brew" ]]; then
                eval "$(/opt/homebrew/bin/brew shellenv)"
            elif [[ -f "/usr/local/bin/brew" ]]; then
                eval "$(/usr/local/bin/brew shellenv)"
            fi
            echo -e "${SUCCESS}✓${NC} Homebrew installed"
        else
            echo -e "${SUCCESS}✓${NC} Homebrew already installed"
        fi
    fi
}

# Check CMake
check_cmake() {
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
        echo -e "${SUCCESS}✓${NC} CMake v${CMAKE_VERSION} found"
        return 0
    else
        echo -e "${WARN}→${NC} CMake not found"
        return 1
    fi
}

# Install CMake  
install_cmake() {
    echo -e "${WARN}→${NC} Installing CMake..."
    if [[ "$OS" == "macos" ]]; then
        brew install cmake
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            maybe_sudo apt-get update -y
            maybe_sudo apt-get install -y cmake
        elif command -v dnf &> /dev/null; then
            maybe_sudo dnf install -y cmake
        elif command -v yum &> /dev/null; then
            maybe_sudo yum install -y cmake
        else
            echo -e "${ERROR}Error: Could not detect package manager${NC}"
            exit 1
        fi
    fi
    echo -e "${SUCCESS}✓${NC} CMake installed"
}

# Check Git
check_git() {
    if command -v git &> /dev/null; then
        echo -e "${SUCCESS}✓${NC} Git already installed"
        return 0
    else
        echo -e "${WARN}→${NC} Git not found"
        return 1
    fi
}

# Install Git
install_git() {
    echo -e "${WARN}→${NC} Installing Git..."
    if [[ "$OS" == "macos" ]]; then
        brew install git
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            maybe_sudo apt-get update -y
            maybe_sudo apt-get install -y git
        elif command -v dnf &> /dev/null; then
            maybe_sudo dnf install -y git
        elif command -v yum &> /dev/null; then
            maybe_sudo yum install -y git
        else
            echo -e "${ERROR}Error: Could not detect package manager${NC}"
            exit 1
        fi
    fi
    echo -e "${SUCCESS}✓${NC} Git installed"
}

# Check GCC/Clang
check_compiler() {
    if command -v cc &> /dev/null || command -v gcc &> /dev/null || command -v clang &> /dev/null; then
        echo -e "${SUCCESS}✓${NC} C compiler found"
        return 0
    else
        echo -e "${WARN}→${NC} C compiler not found"
        return 1
    fi
}

# Install compiler
install_compiler() {
    echo -e "${WARN}→${NC} Installing C compiler..."
    if [[ "$OS" == "macos" ]]; then
        echo -e "${INFO}i${NC} Installing Xcode Command Line Tools..."
        xcode-select --install 2>/dev/null || true
        echo -e "${SUCCESS}✓${NC} Xcode Command Line Tools installed"
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            maybe_sudo apt-get update -y
            maybe_sudo apt-get install -y build-essential
        elif command -v dnf &> /dev/null; then
            maybe_sudo dnf install -y gcc gcc-c++ make
        elif command -v yum &> /dev/null; then
            maybe_sudo yum install -y gcc gcc-c++ make
        fi
        echo -e "${SUCCESS}✓${NC} C compiler installed"
    fi
}

# Check for existing FAHREN
check_existing_fahren() {
    if [[ -x "$HOME/.local/bin/fahren" || -x "$HOME/.fahren/train.sh" ]]; then
        echo -e "${WARN}→${NC} Existing FAHREN installation detected"
        return 0
    fi
    return 1
}

# Ensure user local bin directory
ensure_user_local_bin_on_path() {
    local target="$HOME/.local/bin"
    mkdir -p "$target"
    export PATH="$target:$PATH"
    
    # shellcheck disable=SC2016
    local path_line='export PATH="$HOME/.local/bin:$PATH"'
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [[ -f "$rc" ]] && ! grep -q ".local/bin" "$rc"; then
            echo "$path_line" >> "$rc"
        fi
    done
}

warn_shell_path_missing_dir() {
    local dir="${1%/}"
    local label="$2"
    if [[ -z "$dir" ]]; then
        return 0
    fi
    case ":${ORIGINAL_PATH}:" in
        *":${dir}:"*) return 0 ;;
    esac
    
    echo ""
    echo -e "${WARN}→${NC} PATH warning: missing ${label}: ${INFO}${dir}${NC}"
    echo -e "This can make ${INFO}fahren${NC} show as \"command not found\" in new terminals."
    echo -e "Fix (zsh: ~/.zshrc, bash: ~/.bashrc):"
    echo -e "  export PATH=\"${dir}:\\$PATH\""
}

refresh_shell_command_cache() {
    hash -r 2>/dev/null || true
}

# Main installation
main() {
    if [[ "$HELP" == "1" ]]; then
        print_usage
        return 0
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo -e "${SUCCESS}✓${NC} Dry run mode"
        return 0
    fi

    local is_upgrade=false
    if check_existing_fahren; then
        is_upgrade=true
    fi

    # Step 1: Homebrew (macOS)
    install_homebrew

    # Step 2: Dependencies
    echo ""
    echo -e "${INFO}i${NC} Checking dependencies..."
    
    if ! check_compiler; then
        install_compiler
    fi

    if ! check_cmake; then
        install_cmake
    fi

    if ! check_git; then
        install_git
    fi

    # Step 3: Clone/update repository
    local INSTALL_DIR="${FAHREN_HOME:-$HOME/.fahren}"
    local REPO_URL="https://github.com/IM-NYC/FAHREN.git"

    echo ""
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        echo -e "${WARN}→${NC} Updating FAHREN from git..."
        cd "$INSTALL_DIR"
        if ! git diff-index --quiet HEAD 2>/dev/null; then
            echo -e "${WARN}→${NC} Repo has uncommitted changes; skipping git pull"
        else
            git pull --rebase 2>/dev/null || git fetch --all
        fi
    else
        echo -e "${WARN}→${NC} Cloning FAHREN from GitHub..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    # Make train.sh executable
    chmod +x train.sh 2>/dev/null || true
    # Build project
    echo -e "${WARN}→${NC} Building FAHREN..."
    mkdir -p build
    cd build
    cmake .. >/dev/null 2>&1 || {
        echo -e "${ERROR}✗ CMake configuration failed${NC}"
        exit 1
    }
    make >/dev/null 2>&1 || {
        echo -e "${ERROR}✗ Build failed${NC}"
        exit 1
    }

    echo -e "${SUCCESS}✓${NC} Build successful"
    cd "$INSTALL_DIR"

    # Setup symlink and PATH
    ensure_user_local_bin_on_path
    
    if [[ ! -L "$HOME/.local/bin/fahren" ]]; then
        mkdir -p "$HOME/.local/bin"
        ln -sf "$INSTALL_DIR/train.sh" "$HOME/.local/bin/fahren"
        echo -e "${SUCCESS}✓${NC} Symlink created: \$HOME/.local/bin/fahren"
    fi

    refresh_shell_command_cache
    warn_shell_path_missing_dir "$HOME/.local/bin" "user-local bin dir (~/.local/bin)"

    # Success message
    echo ""
    if [[ "$is_upgrade" == "true" ]]; then
        local upgrade_messages=(
            "Upgraded! New neurons, same attitude."
            "Back and better. The backprop has returned."
            "Updated! I went to compute school."
            "Freshly trained (myself). Let's go."
            "Version bump! I aged well."
            "Reimplemented. Fewer bugs, more sass."
        )
        local upgrade_message
        upgrade_message="${upgrade_messages[RANDOM % ${#upgrade_messages[@]}]}"
        echo -e "${SUCCESS}${BOLD}🧠 FAHREN upgraded successfully!${NC}"
        echo -e "${MUTED}${upgrade_message}${NC}"
    else
        local completion_messages=(
            "Welcome to neural networks without the suffering."
            "Installation complete. Your GPU is warming up."
            "Ready to train. Let's make some predictions."
            "All set. The neurons are eager."
            "System online. Accuracy levels: optimal."
            "Installed. Your data is about to get smart."
        )
        local completion_message
        completion_message="${completion_messages[RANDOM % ${#completion_messages[@]}]}"
        echo -e "${SUCCESS}${BOLD}🧠 FAHREN installed successfully!${NC}"
        echo -e "${MUTED}${completion_message}${NC}"
    fi

    echo ""
    echo -e "Installation directory: ${INFO}${INSTALL_DIR}${NC}"
    echo -e "Symlink: ${INFO}\$HOME/.local/bin/fahren${NC}"
    echo ""
    
    if ! command -v fahren &>/dev/null; then
        echo -e "${WARN}→${NC} ${INFO}fahren${NC} command not found in current shell"
        warn_shell_path_missing_dir "$HOME/.local/bin" "user-local bin"
        echo -e "To use immediately: ${INFO}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
        echo -e "To use in new terminals: Edit ${INFO}~/.bashrc${NC} or ${INFO}~/.zshrc${NC}"
    fi

    echo ""
    echo -e "Quick start:"
    echo -e "  ${INFO}fahren${NC}              # Run the TUI"
    echo -e "  ${INFO}cd $INSTALL_DIR${NC}"
    echo -e "  ${INFO}./train.sh${NC}          # Or run directly"
    echo ""
    echo -e "Documentation: ${INFO}https://docs.fahren.ai${NC}"
    echo -e "GitHub: ${INFO}${REPO_URL}${NC}"
    echo ""
}

if [[ "${FAHREN_INSTALL_SH_NO_RUN:-0}" != "1" ]]; then
    parse_args "$@"
    configure_verbose
    main
fi
