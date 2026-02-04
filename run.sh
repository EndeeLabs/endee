#!/usr/bin/env bash
set -euo pipefail

log()   { printf "[INFO] %s\n" "$*"; }
warn()  { printf "[WARN] %s\n" "$*" >&2; }
error() { printf "[ERROR] %s\n" "$*" >&2; }

NDD_DATA_DIR="./data"
BINARY_FILE=""
NDD_AUTH_TOKEN=""


print_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  ndd_data_dir=DIR   Set the data directory (default: ./data)
  binary_file=FILE   Set the binary file to run (default: auto-detected in build/)
  ndd_auth_token=TOKEN Set the auth token (default: empty)
  --help, -h         Show this help message and exit

Description:
  Runs the ndd binary. It attempts to find a binary starting with 'ndd-*'
  in the 'build' directory if not explicitly provided.
EOF
}

main() {
    for ARG in "$@"; do
        case "$ARG" in
            ndd_data_dir=*)
                NDD_DATA_DIR="${ARG#*=}"
                ;;
            binary_file=*)
                BINARY_FILE="${ARG#*=}"
                ;;
            ndd_auth_token=*)
                NDD_AUTH_TOKEN="${ARG#*=}"
                ;;
            --help|-h)
                print_help
                exit 0
                ;;
        esac
    done

    if [[ -z "$BINARY_FILE" ]]; then
        # check if build folder exists and if any binary starting with ndd-* exists, fi yes then save the filename in a variable
        if [[ -d "build" && -n "$(find build -maxdepth 1 -name 'ndd-*' -type f)" ]]; then
            BINARY_FILE=$(find build -maxdepth 1 -name 'ndd-*' -type f | head -n 1)
            log "Found binary: $BINARY_FILE"
        else
            error "No binary found"
            exit 1
        fi
    fi

    # run the binary with the arguments passed to this script
    if [[ -n "$BINARY_FILE" ]]; then
        eval "NDD_DATA_DIR=$NDD_DATA_DIR NDD_AUTH_TOKEN=$NDD_AUTH_TOKEN $BINARY_FILE"
    fi
}

main "$@"
