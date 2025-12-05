#!/bin/bash
docker build -t mmaba-pseudo .
docker run --rm -v $(pwd)/wandb:/app/wandb mmaba-pseudo "$@"
