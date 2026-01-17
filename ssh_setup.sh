#!/usr/bin/env bash
set -e

echo "🔐 Setting up GitHub SSH authentication..."

SSH_KEY="$HOME/.ssh/id_ed25519"

# 1. Ensure ~/.ssh exists
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# 2. Create SSH key if it doesn't exist
if [ ! -f "$SSH_KEY" ]; then
    echo "➡️  No SSH key found. Generating a new one..."
    ssh-keygen -t ed25519 -f "$SSH_KEY" -C "$(whoami)@$(hostname)" -N ""
else
    echo "✅ SSH key already exists: $SSH_KEY"
fi

# 3. Start ssh-agent if not running
if [ -z "$SSH_AUTH_SOCK" ]; then
    echo "🚀 Starting ssh-agent..."
    eval "$(ssh-agent -s)"
else
    echo "✅ ssh-agent already running"
fi

# 4. Add key to agent
ssh-add "$SSH_KEY" >/dev/null 2>&1 || true
echo "🔑 SSH key added to agent"

# 5. Print public key for GitHub
echo
echo "📋 Copy the following SSH public key and add it to GitHub:"
echo "👉 https://github.com/settings/ssh/new"
echo
echo "----------------------------------------"
cat "${SSH_KEY}.pub"
echo "----------------------------------------"
echo

# 6. Switch git remote to SSH (if in a git repo)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ORIGIN_URL=$(git remote get-url origin 2>/dev/null || true)

    if [[ "$ORIGIN_URL" == https://github.com/* ]]; then
        SSH_URL=$(echo "$ORIGIN_URL" | sed -E 's|https://github.com/|git@github.com:|')
        git remote set-url origin "$SSH_URL"
        echo "🔁 Git remote switched to SSH:"
        echo "   $SSH_URL"
    else
        echo "ℹ️  Git remote already using SSH or is non-GitHub:"
        echo "   $ORIGIN_URL"
    fi
else
    echo "ℹ️  Not inside a git repository, skipping remote update"
fi

echo
echo "✅ SSH setup complete!"
echo "➡️  After adding the key to GitHub, run: git push"
