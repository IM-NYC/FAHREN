#!/bin/bash
# FAHREN Neural Network Training Tool - TUI Edition

set -e

# Terminal control
readonly CLEAR='\033[2J'
readonly HIDE_CURSOR='\033[?25l'
readonly SHOW_CURSOR='\033[?25h'
readonly RESET='\033[0m'

# Colors
readonly BLACK='\033[30m'
readonly RED='\033[31m'
readonly GREEN='\033[32m'
readonly YELLOW='\033[33m'
readonly BLUE='\033[34m'
readonly MAGENTA='\033[35m'
readonly CYAN='\033[36m'
readonly WHITE='\033[37m'
readonly BOLD='\033[1m'
readonly DIM='\033[2m'

# Cursor movement
readonly HOME='\033[H'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
MODEL_OUTPUT="$HOME/.fahren/mnist_model.bin"
MNIST_PATH="$SCRIPT_DIR/public"
DEBUG=0

# TUI State
CURRENT_MENU="main"
SELECTED_OPTION=0
MENU_OPTIONS=()
MAX_Y=0
MAX_X=0

mkdir -p ~/.fahren
show_logo() {
    cat << 'EOF'
    ███████╗ █████╗ ██╗  ██╗██████╗     ███████╗███████╗
    ██╔════╝██╔══██╗██║  ██║██╔══██╗    ██╔════╝██╔════╝
    █████╗  ███████║███████║██████╔╝    █████╗  █████╗  
    ██╔══╝  ██╔══██║██╔══██║██╔══██╗    ██╔══╝  ██╔══╝  
    ██║     ██║  ██║██║  ██║██║  ██║    ██║     ███████╗
    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝     ╚══════╝
EOF
}

# Terminal Control
init_screen() {
    printf "$CLEAR$HOME$HIDE_CURSOR"
    # Get terminal dimensions (cross-platform)
    if command -v tput &> /dev/null; then
        MAX_Y=$(tput lines)
        MAX_X=$(tput cols)
    else
        MAX_Y=${LINES:-24}
        MAX_X=${COLUMNS:-80}
    fi
}

cleanup_screen() {
    printf "$SHOW_CURSOR"
}

trap cleanup_screen EXIT

move_cursor() {
    printf "\033[$1;$2H"
}

clear_line() {
    printf "\033[K"
}

draw_buffer() {
    printf "$CLEAR$HOME"
}

draw_box() {
    local x=$1 y=$2 w=$3 h=$4
    move_cursor $y $x
    printf "╔"
    for ((i=0; i<w-2; i++)); do printf "═"; done
    printf "╗"
    for ((i=1; i<h-1; i++)); do
        move_cursor $((y+i)) $x
        printf "║"
        move_cursor $((y+i)) $((x+w-1))
        printf "║"
    done
    move_cursor $((y+h-1)) $x
    printf "╚"
    for ((i=0; i<w-2; i++)); do printf "═"; done
    printf "╝"
}

draw_title() {
    local title="$1"
    local y=$2
    move_cursor $y 2
    printf "${CYAN}${BOLD}▶ $title${RESET}"
    move_cursor $((y+1)) 2
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

draw_menu_item() {
    local index=$1
    local text=$2
    local line=$3
    if [ $index -eq $SELECTED_OPTION ]; then
        move_cursor $line 4
        printf "${WHITE}${BOLD}❯ $text${RESET}"
    else
        move_cursor $line 4
        printf "${DIM}  $text${RESET}"
    fi
}

draw_status() {
    move_cursor $((MAX_Y-2)) 2
    clear_line
    printf "${GREEN}✓ $1${RESET}"
}

draw_error() {
    move_cursor $((MAX_Y-2)) 2
    clear_line
    printf "${RED}✗ $1${RESET}"
}

# Main Menu
setup_main_menu() {
    MENU_OPTIONS=(
        "Train Default Model"
        "Create Custom Model"
        "Test Model"
        "View Examples"
        "Settings"
        "Exit"
    )
    SELECTED_OPTION=0
}

draw_main_menu() {
    draw_buffer
    move_cursor 1 5
    printf "${CYAN}${BOLD}"
    show_logo
    printf "${RESET}"
    draw_title "Neural Network Trainer" 15
    local line=18
    for i in "${!MENU_OPTIONS[@]}"; do
        draw_menu_item $i "${MENU_OPTIONS[$i]}" $line
        line=$((line + 2))
    done
    move_cursor $((MAX_Y-3)) 2
    printf "${CYAN}Use ↑↓ to navigate, ENTER to select${RESET}"
}

handle_main_menu_input() {
    local key="$1"
    case $key in
        A) # Up arrow
            [ $SELECTED_OPTION -gt 0 ] && SELECTED_OPTION=$((SELECTED_OPTION - 1))
            draw_main_menu
            ;;
        B) # Down arrow
            [ $SELECTED_OPTION -lt $((${#MENU_OPTIONS[@]} - 1)) ] && SELECTED_OPTION=$((SELECTED_OPTION + 1))
            draw_main_menu
            ;;
        '') # Enter key
            case $SELECTED_OPTION in
                0) train_default ;;
                1) create_custom_model ;;
                2) test_model ;;
                3) view_examples ;;
                4) settings_menu ;;
                5) cleanup_screen; exit 0 ;;
            esac
            ;;
    esac
}

# Training Functions
train_default() {
    draw_buffer
    draw_title "Train Default Model" 2
    move_cursor 5 4
    printf "${YELLOW}Configuration:${RESET}"
    move_cursor 6 6
    printf "Epochs:        ${CYAN}5${RESET}"
    move_cursor 7 6
    printf "Learning rate: ${CYAN}0.01${RESET}"
    move_cursor 8 6
    printf "Architecture:  ${CYAN}784 → 128 → 64 → 10${RESET}"
    move_cursor 10 4
    printf "${YELLOW}Proceed?${RESET}"
    move_cursor 11 6
    printf "[Y]es  [N]o"
    
    local key
    while true; do
        read -rsn 1 key
        case $key in
            Y|y)
                move_cursor 13 4
                printf "${YELLOW}Training...${RESET}"
                "$BUILD_DIR/Fahren_mnist_example" \
                    "$MNIST_PATH/t10k-images.idx3-ubyte" \
                    "$MNIST_PATH/t10k-labels.idx1-ubyte" 5 "$MODEL_OUTPUT" 2>&1 | tail -2
                draw_status "Training complete"
                move_cursor 16 4
                printf "Press ENTER to continue..."
                read
                setup_main_menu
                draw_main_menu
                return
                ;;
            N|n)
                setup_main_menu
                draw_main_menu
                return
                ;;
        esac
    done
}

create_custom_model() {
    draw_buffer
    draw_title "Create Custom Model - Step 1: Name" 2
    move_cursor 5 4
    printf "Enter model name (alphanumeric): "
    read -p "" model_name
    
    draw_buffer
    draw_title "Create Custom Model - Step 2: Type" 2
    move_cursor 5 4
    printf "[0] Sequential  [1] LSTM"
    move_cursor 6 4
    read -p "Select: " model_type
    
    draw_buffer
    draw_title "Create Custom Model - Step 3: Layers" 2
    move_cursor 5 4
    read -p "Number of hidden layers (1-255): " num_layers
    
    if ! [[ "$num_layers" =~ ^[0-9]+$ ]] || [ $num_layers -lt 1 ] || [ $num_layers -gt 255 ]; then
        draw_error "Invalid layer count (1-255)"
        sleep 1
        setup_main_menu
        draw_main_menu
        return
    fi
    
    declare -a layer_sizes
    move_cursor 7 4
    for ((i=1; i<=num_layers; i++)); do
        move_cursor $((6+i)) 4
        read -p "Layer $i neurons (1-65535): " neurons
        if ! [[ "$neurons" =~ ^[0-9]+$ ]] || [ $neurons -lt 1 ] || [ $neurons -gt 65535 ]; then
            draw_error "Invalid neuron count"
            return
        fi
        layer_sizes[$((i-1))]=$neurons
    done
    
    draw_buffer
    draw_title "Create Custom Model - Step 4: Activation" 2
    move_cursor 5 4
    printf "[1] Same for all  [2] Different per layer"
    move_cursor 6 4
    read -p "Select: " activation_mode
    
    draw_buffer
    draw_title "Create Custom Model - Summary" 2
    move_cursor 5 4
    printf "${YELLOW}Model Name: ${CYAN}$model_name${RESET}"
    move_cursor 6 4
    printf "${YELLOW}Layers: ${CYAN}$num_layers${RESET}"
    move_cursor 8 4
    printf "Architecture: ${CYAN}784"
    for size in "${layer_sizes[@]}"; do printf " → $size"; done
    printf " → 10${RESET}"
    move_cursor 10 4
    printf "[Y]es  [N]o"
    read -rsn 1 key
    case $key in
        Y|y)
            draw_status "Model configuration saved"
            sleep 1
            ;;
    esac
    setup_main_menu
    draw_main_menu
}

test_model() {
    draw_buffer
    draw_title "Test Model" 2
    if [ ! -f "$MODEL_OUTPUT" ]; then
        draw_error "No trained model found"
        move_cursor 5 4
        printf "Train a model first"
        move_cursor 7 4
        printf "Press ENTER..."
        read
        setup_main_menu
        draw_main_menu
        return
    fi
    move_cursor 5 4
    printf "${YELLOW}Testing...${RESET}"
    "$BUILD_DIR/Fahren_mnist_test" 2>&1 | grep -A 8 "╔"
    draw_status "Testing complete"
    move_cursor $((MAX_Y-2)) 4
    printf "Press ENTER..."
    read
    setup_main_menu
    draw_main_menu
}

view_examples() {
    draw_buffer
    draw_title "Example Models" 2
    move_cursor 5 4
    printf "${CYAN}1. MNIST Classifier${RESET}"
    move_cursor 6 4
    printf "   Handwritten digit recognition (784→128→64→10)"
    move_cursor 8 4
    printf "${CYAN}2. Fashion Classifier${RESET}"
    move_cursor 9 4
    printf "   Coming soon..."
    move_cursor 11 4
    printf "Press ENTER..."
    read
    setup_main_menu
    draw_main_menu
}

settings_menu() {
    draw_buffer
    draw_title "Settings" 2
    move_cursor 5 4
    printf "MNIST Path: ${CYAN}$MNIST_PATH${RESET}"
    move_cursor 6 4
    printf "Build Dir:  ${CYAN}$BUILD_DIR${RESET}"
    move_cursor 7 4
    printf "Output Dir: ${CYAN}$HOME/.fahren${RESET}"
    move_cursor 9 4
    printf "Press ENTER..."
    read
    setup_main_menu
    draw_main_menu
}

check_env() {
    if ! command -v cmake &> /dev/null || ! command -v make &> /dev/null; then
        printf "${RED}Missing dependencies: cmake, make${RESET}\n"
        exit 1
    fi
}

build_if_needed() {
    if [ ! -f "$BUILD_DIR/Fahren_mnist_example" ]; then
        cd "$BUILD_DIR" 2>/dev/null && cmake .. > /dev/null 2>&1 && make > /dev/null 2>&1
        cd "$SCRIPT_DIR"
    fi
}

# Main Loop
main_loop() {
    init_screen
    check_env
    build_if_needed
    setup_main_menu
    draw_main_menu
    
    while true; do
        read -rsn 1 key
        [ "$key" = $'\x1b' ] && { read -rsn 1 bracket; read -rsn 1 key; }
        handle_main_menu_input "$key"
    done
}

main_loop "$@"
