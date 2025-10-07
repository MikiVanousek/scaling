#!/bin/bash

# Check that exactly one argument is passed
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <config_file>"
  exit 1
fi


runai submit \
  --job-name-prefix train \
  --image ic-registry.epfl.ch/tml/tml:v2 \
  --pvc tml-scratch:/tmlscratch \
  --working-dir / \
  -e USER_HOME=/tmlscratch/vanousek \
  -e HOME=/tmlscratch/vanousek \
  -e USER=vanousek \
  -e UID=1000 \
  -e GROUP=TML-unit \
  -e GID=11180 \
  -e HF_TOKEN=SECRET:my-secret,hf_token\
  -e WANDB_API_KEY=SECRET:my-secret,wandb_api_key\
  --cpu 1 \
  --cpu-limit 16 \
  --gpu 1 \
  --run-as-uid 1000 \
  --run-as-gid 11180 \
  --working-dir / \
  --image-pull-policy IfNotPresent \
  --memory 8G \
  --tty \
  --stdin \
  --allow-privilege-escalation \
  --command -- /bin/bash -c "cd /tmlscratch/vanousek/scaling && git pull && ../bin/pixi run python train.py --config $1"
