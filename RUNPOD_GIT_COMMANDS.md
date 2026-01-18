# RunPod Git Pull Commands

## Quick Pull Commands

### Option 1: Simple Pull (if already in the repo directory)
```bash
cd /workspace/train_70b && git pull
```

### Option 2: Pull and Retrain
```bash
cd /workspace/train_70b && git pull && ./start.sh 32b
```

### Option 3: Full Path Pull
```bash
cd /workspace && cd train_70b && git pull
```

### Option 4: Pull with Reset (if you have local changes you want to discard)
```bash
cd /workspace/train_70b && git fetch && git reset --hard origin/master
```

### Option 5: Pull with Stash (save local changes, pull, then reapply)
```bash
cd /workspace/train_70b && git stash && git pull && git stash pop
```

## Complete Workflow: Clone → Pull → Train

### First Time Setup:
```bash
cd /workspace
git clone https://github.com/Sweetflips/train_70b.git
cd train_70b
chmod +x start.sh setup.sh run.sh train.py
./start.sh 32b
```

### Subsequent Runs (Pull Latest):
```bash
cd /workspace/train_70b
git pull
./start.sh 32b
```

## One-Liner: Pull and Train
```bash
cd /workspace/train_70b && git pull && ./start.sh 32b
```

## Troubleshooting

### If git pull fails with "not a git repository":
```bash
cd /workspace
# Check if repo exists
ls -la | grep train_70b

# If missing, clone it
git clone https://github.com/Sweetflips/train_70b.git

# Then pull
cd train_70b && git pull
```

### If you have uncommitted changes blocking pull:
```bash
cd /workspace/train_70b
# See what changed
git status

# Option A: Discard changes and pull
git reset --hard HEAD
git pull

# Option B: Stash changes, pull, then decide
git stash
git pull
git stash list  # See stashed changes
# git stash pop  # To reapply changes
```

### Check current branch and status:
```bash
cd /workspace/train_70b
git branch          # Show current branch
git status          # Show changes
git log --oneline -5  # Show last 5 commits
```

### Force pull (discard all local changes):
```bash
cd /workspace/train_70b
git fetch origin
git reset --hard origin/master
```

## RunPod Serverless Startup Script (Auto-Pull)
```bash
#!/bin/bash
set -e
cd /workspace
if [ -d "train_70b" ]; then
    cd train_70b
    git pull || echo "Pull failed, using existing code"
else
    git clone https://github.com/Sweetflips/train_70b.git
    cd train_70b
fi
chmod +x start.sh setup.sh run.sh train.py
./start.sh 32b
```
