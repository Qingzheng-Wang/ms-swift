#!/bin/bash
set -e

NODES="h1 h2 h3 h4 h5 h6 h7 h8"

setup_node() {
    local node=$1
    
    ssh $node "
        # Set apt proxies
        sudo tee /etc/apt/apt.conf.d/proxy.conf << EOF
Acquire::http::Proxy \"http://10.0.0.72:20172\";
Acquire::https::Proxy \"http://10.0.0.72:20172\";
EOF
        
        # Update package lists
        sudo apt update
        
        # Install required packages
        sudo apt install -y zsh ffmpeg build-essential
        
        # Clone and build opus
        wget https://raw.githubusercontent.com/fishaudio/opusenc/refs/heads/main/install_opus.sh
        sudo bash install_opus.sh

        echo 'Setup completed on $node'
    "
}

# Loop through all nodes and set them up
# for node in $NODES; do
#     echo "Setting up $node..."
#     setup_node $node &
# done

# wait

# echo "Setup completed on all nodes"
# exit

# Function to sync keys for a single user
sync_user_keys() {
    local user=$1
    local node=$2
    local home_dir=$(getent passwd "$user" | cut -d: -f6)

    # Create and set ownership of home directory
    ssh "$node" "sudo mkdir -p $home_dir"
    ssh "$node" "sudo chown $user:$(id -gn $user) $home_dir"
    ssh "$node" "sudo chmod 755 $home_dir"

    if [ -d "$home_dir/.ssh" ]; then
        echo "Syncing SSH keys for user $user to $node..."
        
        # Create .ssh directory with correct permissions
        ssh "$node" "sudo mkdir -p $home_dir/.ssh"
        ssh "$node" "sudo chown $user:$(id -gn $user) $home_dir/.ssh"
        ssh "$node" "sudo chmod 700 $home_dir/.ssh"
        
        # Sync authorized_keys if it exists
        if [ -f "$home_dir/.ssh/authorized_keys" ]; then
            scp "$home_dir/.ssh/authorized_keys" "$node:$home_dir/.ssh/"
            ssh "$node" "sudo chown $user:$(id -gn $user) $home_dir/.ssh/authorized_keys"
            ssh "$node" "sudo chmod 600 $home_dir/.ssh/authorized_keys"
        fi

        # Enable sshkey authentication
        ssh "$node" "sudo sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config"
        # Restart SSH service on remote node
        ssh "$node" "sudo systemctl restart sshd"

        echo "SSH key synchronization complete for user $user on $node"
    fi
}

# Get all regular users (UID >= 1000)
# USERS=$(getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 {print $1}')

USERS=("lengyue" "stardust" "rcell" "ylzz" "ermu2001")

for node in $NODES; do
    echo "Syncing to $node..."
    
    # Sync user database files
    scp /etc/passwd "$node":/etc/passwd
    scp /etc/group "$node":/etc/group
    scp /etc/shadow "$node":/etc/shadow
    scp /etc/gshadow "$node":/etc/gshadow
    
    # Restart user services
    ssh "$node" "systemctl restart nscd 2>/dev/null || true"
    ssh "$node" "systemctl restart nslcd 2>/dev/null || true"
    
    # Sync SSH keys for each user
    for user in "${USERS[@]}"; do
        sync_user_keys "$user" "$node"
    done

    # Verify user resolution
    ssh "$node" "getent passwd > /dev/null && echo '$node: User database test passed' || echo '$node: User database test failed'"

    # Verify sshkey authentication
    ssh -o BatchMode=yes "$node" "echo '$node: SSH key test passed'" || echo "$node: SSH key test failed"
done

echo "Synchronization complete."
