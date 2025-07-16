#!/bin/bash

# Check for argument
if [ -z "$1" ]; then
    echo "Usage: ./view_frames.sh <folder-name>"
    echo "Example: ./view_frames.sh test_video"
    exit 1
fi

TARGET_DIR="outputs/$1"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Launch feh with desired settings
feh "$TARGET_DIR" --force-aliasing --zoom 600 --fullscreen
