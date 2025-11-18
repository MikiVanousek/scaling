# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
  echo "Warning: You have uncommitted changes."
  git status --porcelain
  read -p "Do you want to continue anyway? [y/N]: " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 1
  fi
fi

# Check for unpushed commits
UNPUSHED=$(git log origin/main..HEAD --oneline)
if [ ! -z "$UNPUSHED" ]; then
  echo "Warning: You have unpushed commits:"
  echo "$UNPUSHED"
  read -p "Do you want to continue anyway? [y/N]: " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 1
  fi
fi
JOB_PREFIX="j"
JOB_NUM=1
while runai list | grep -q "${JOB_PREFIX}${JOB_NUM}"; do
  ((JOB_NUM++))
done

JOB_NAME="${JOB_PREFIX}${JOB_NUM}"
echo "Submitting job: $JOB_NAME"

runai submit \
  -p tml-vanousek \
  --name $JOB_NAME \
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
  --memory 64G \
  --tty \
  --stdin \
  --allow-privilege-escalation \
  --interactive
