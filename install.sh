#!/usr/bin/env bash
set -euo pipefail

########################################
# Logging helpers
########################################
log()   { printf "[INFO] %s\n" "$*"; }
warn()  { printf "[WARN] %s\n" "$*" >&2; }
error() { printf "[ERROR] %s\n" "$*" >&2; }

########################################
# Globals
########################################
OS_FAMILY="unknown"   # linux | mac
OS_ARCH="unknown"     # x86_64 | arm64 | ...
DISTRO_ID="unknown"
DISTRO_VERSION_ID="unknown"
DISTRO_CODENAME="unknown"

# Defaults
BUILD_MODE="release"  # release | debug_all | debug_nd
CPU_TARGET=""         # avx2 | avx512 | neon | sve2 (empty by default)
SKIP_DEPS=false       # if true, skip dependency installation step

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# dependencies list
pkg_debian_ubuntu=(cmake clang-19 build-essential libssl-dev libcurl4-openssl-dev unzip curl git)
pkg_redhat=(cmake openssl-devel libcurl-devel clang unzip curl git)
pkg_macos=(cmake unzip curl git openssl@3)


# ****************************************
# Debian/Ubuntu Shared Helper
# ****************************************
apt_install_if_missing_debian() {
    local to_install=()
    for pkg in "$@"; do
        if dpkg -s "$pkg" >/dev/null 2>&1; then
            log "Package already installed (apt): $pkg"
        else
            log "Package missing (apt), will install: $pkg"
            to_install+=("$pkg")
        fi
    done

    if ((${#to_install[@]})); then
        log "Running apt-get update..."
        sudo apt-get update
        log "Installing via apt-get: ${to_install[*]}"
        sudo apt-get install -y "${to_install[@]}"
    else
        log "All apt packages already present."
    fi
}

# ****************************************
# Debian-based system
# ****************************************
install_dependencies_debian_family() {
    local version=$1
    echo "--- Detected Debian (Version: $version) ---"
    apt_install_if_missing_debian "${pkg_debian_ubuntu[@]}"
}

# ****************************************
# Ubuntu-based system
# ****************************************
install_dependencies_ubuntu_family() {
    local version=$1
    version=$(echo "$version" | tr -d '"')
    echo "--- Detected Ubuntu (Version: $version) ---"
    
    if [[ "$version" == "22.04" ]]; then
        if dpkg -s "clang-19" >/dev/null 2>&1; then
            log "$DISTRO_ID $version: Clang-19 already detected. Skipping LLVM repo setup."
        else
            log "$DISTRO_ID $version: Clang-19 missing. Running LLVM repository setup..."
            sudo apt-get update && sudo apt-get install -y wget gnupg
            wget https://apt.llvm.org/llvm.sh
            chmod +x llvm.sh
            sudo ./llvm.sh 19
            rm llvm.sh
        fi
    elif ! dpkg --compare-versions "$version" "ge" "22.04"; then
        echo "Version is too old."
        exit 1
    fi

    apt_install_if_missing_debian "${pkg_debian_ubuntu[@]}"
}

# ****************************************
# RedHat-based system (Rocky/RHEL/Fedora)
# ****************************************
install_dependencies_redhat_family() {
    local version=$1
    echo "--- Detected RedHat-family (Version: $version) ---"

    local pkg_mgr="dnf"
    if command -v dnf5 >/dev/null 2>&1; then
        pkg_mgr="dnf5"
    fi

    if [[ "$DISTRO_ID" != "fedora" ]]; then
        if ! rpm -q epel-release >/dev/null 2>&1; then
            log "Installing EPEL repository via $pkg_mgr..."
            sudo $pkg_mgr install -y epel-release
        fi
    fi

    log "Checking for $pkg_mgr dependencies..."

    sudo $pkg_mgr install -y @development-tools
    sudo $pkg_mgr install -y "${pkg_redhat[@]}"
}

# ****************************************
# macOS
# ****************************************
brew_install_if_missing() {
    if ! command -v brew >/dev/null 2>&1; then
        error "Homebrew not found on macOS. Install it from https://brew.sh"
        exit 1
    fi

    local formula
    for formula in "$@"; do
        if command -v "$formula" >/dev/null 2>&1; then
            log "Command already available: $formula"
            continue
        fi

        if brew list --versions "$formula" >/dev/null 2>&1; then
            log "Brew formula already installed: $formula"
        else
            log "Installing brew formula: $formula"
            brew install "$formula"
            log "Installed: $formula"
        fi
    done
}
install_dependencies_macos() {
    brew_install_if_missing "${pkg_macos[@]}"

    if ! command -v clang >/dev/null 2>&1; then
        warn "clang not found. Install Xcode Command Line Tools: 'xcode-select --install'"
    fi
}

# ****************************************
# DISTRO FACTORY
# ****************************************
distro_factory() {
    # 1. Detect Kernel/Family
    OS_ARCH="$(uname -m)"
    case "$(uname -s)" in
        Linux)
            OS_FAMILY="linux"
            ;;
        Darwin)
            OS_FAMILY="mac"
            ;;
        *)
            error "Unsupported kernel: $uname_s"
            exit 1
            ;;
    esac

    if [[ "$OS_FAMILY" == "mac" ]]; then
        DISTRO_ID="macos"
        INSTALLER_FUNC="install_dependencies_macos"
        if [[ "$OS_ARCH" != "arm64" ]]; then
            error "Unsupported macOS architecture: $OS_ARCH. Only Apple Silicon (arm64/M-series) is supported."
            exit 1
        fi
        return 0
    fi

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO_ID="${ID:-unknown}"
        DISTRO_VERSION_ID="${VERSION_ID:-unknown}"
        DISTRO_CODENAME="${VERSION_CODENAME:-unknown}"

        case $DISTRO_ID in
            ubuntu) INSTALLER_FUNC="install_dependencies_ubuntu_family $DISTRO_VERSION_ID" ;;
            debian)          INSTALLER_FUNC="install_dependencies_debian_family $DISTRO_VERSION_ID" ;;
            rocky|almalinux|fedora|rhel) INSTALLER_FUNC="install_dependencies_redhat_family $DISTRO_VERSION_ID" ;;
            *) return 1 ;;
        esac
    else
        return 1
    fi
}
# ****************************************

add_frontend() {
    log "pulling frontend"
    mkdir -p $script_dir/frontend
    cd $script_dir/frontend
    curl -L -o react-dist.zip https://github.com/EndeeLabs/endee-web-ui/releases/download/v1.0.2/endee-web-ui.zip
    unzip -o react-dist.zip
    rm react-dist.zip
    log "frontend added"
}

# ****************************************
# Configure & build (cmake + make)
# ****************************************
get_njobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif [[ "$OS_FAMILY" == "mac" ]]; then
        sysctl -n hw.ncpu
    else
        echo 1
    fi
}

build_project() {
    local build_dir
    build_dir="${script_dir}/build"

    mkdir -p "$build_dir"
    cd "$build_dir"

    # Clean previous builds to ensure flags update
    rm -rf "${build_dir:?}"/*

    # 1. Prepare CMake Arguments Array
    local cmake_args=()

    # --- Apply Build Mode Flags ---
    case "$BUILD_MODE" in
        debug_all)
            log "Mode: Debug All"
            cmake_args+=("-DND_DEBUG=ON" "-DDEBUG=ON")
            ;;
        debug_nd)
            log "Mode: Debug ND"
            cmake_args+=("-DND_DEBUG=ON")
            ;;
        release)
            log "Mode: Release"
            # No specific flags for release, strictly speaking, 
            # unless you want -DCMAKE_BUILD_TYPE=Release
            ;;
        *)
            error "Unknown BUILD_MODE: $BUILD_MODE"
            exit 1
            ;;
    esac

    # --- Apply CPU Optimization Flags ---
    if [[ -n "$CPU_TARGET" ]]; then
        log "CPU Target: $CPU_TARGET"
        case "$CPU_TARGET" in
            avx2)   cmake_args+=("-DUSE_AVX2=ON") ;;
            avx512) cmake_args+=("-DUSE_AVX512=ON") ;;
            neon)   cmake_args+=("-DUSE_NEON=ON") ;;
            sve2)   cmake_args+=("-DUSE_SVE2=ON") ;;
            *) 
                error "Unknown CPU target: $CPU_TARGET"
                exit 1 
                ;;
        esac
    else
        log "No CPU optimization flag provided."
    fi

    # 2. Run CMake
    log "Running cmake with flags: ${cmake_args[*]}"
    if ! cmake "${cmake_args[@]}" ..; then
        error "cmake configuration failed."
        exit 1
    fi

    # 3. Build
    
    local jobs
    jobs="$(get_njobs)"
    log "Running make -j${jobs}"
    if ! make -j"${jobs}"; then
        error "make failed."
        exit 1
    fi
    log "Build finished successfully."

}
# ****************************************


print_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build Mode Options (Select one):
  --release       Build default release (default)
  --debug_all     Build with -DND_DEBUG=ON -DDEBUG=ON
  --debug_nd      Build with -DND_DEBUG=ON

CPU Optimization Options (Select one):
  --avx2          Add -DUSE_AVX2=ON
  --avx512        Add -DUSE_AVX512=ON
  --neon          Add -DUSE_NEON=ON
  --sve2          Add -DUSE_SVE2=ON

General Options:
  --skip-deps     Skip the dependency installation step
  --help, -h      Show this help message and exit

Example:
  $(basename "$0") --debug_nd --avx2
EOF
}

parse_args() {
    # >>> CHANGE: If no arguments are provided, show help and exit <<<
    if [[ $# -eq 0 ]]; then
        print_help
        exit 1
    fi

    BUILD_MODE="release"
    CPU_TARGET=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            # --- Build Modes ---
            --release)
                BUILD_MODE="release"
                shift
                ;;
            --debug_all)
                BUILD_MODE="debug_all"
                shift
                ;;
            --debug_nd)
                BUILD_MODE="debug_nd"
                shift
                ;;
            
            # --- CPU Flags ---
            --avx2)
                CPU_TARGET="avx2"
                shift
                ;;
            --avx512)
                CPU_TARGET="avx512"
                shift
                ;;
            --neon)
                CPU_TARGET="neon"
                shift
                ;;
            --sve2)
                CPU_TARGET="sve2"
                shift
                ;;

            # --- General ---
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                print_help
                exit 1
                ;;
        esac
    done

    log "Selected Build Mode: ${BUILD_MODE}"
    if [[ -n "$CPU_TARGET" ]]; then
        log "Selected CPU Flag:   ${CPU_TARGET}"
    fi
}



main() {
    parse_args "$@" 

    log "Identifying environment..."
    
    distro_factory || {
        error "Unsupported OS or /etc/os-release missing."
        exit 1
    }

    log "Environment: OS_FAMILY=$OS_FAMILY, ARCH=$OS_ARCH, DISTRO=$DISTRO_ID ($DISTRO_VERSION_ID)"

    if [[ "$SKIP_DEPS" == "false" ]]; then
        eval "$INSTALLER_FUNC" 
    else
        log "Skipping dependency installation step."
    fi

    build_project

    add_frontend


    log ""
    log "Build and installation successful!"
    log "You can now start the server by running:"
    log "  ./run.sh"
}

main "$@"
