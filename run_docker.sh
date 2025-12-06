#!/bin/bash

# Define the image name
IMAGE_NAME="mmaba-pseudo"

# Build the image
docker build -t "$IMAGE_NAME" .

# Run the container
docker run -it --rm -v "$(pwd)/wandb:/app/wandb" "$IMAGE_NAME" "$@"
