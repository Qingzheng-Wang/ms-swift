import json
import subprocess
from pathlib import Path

import click


@click.command()
@click.option("--nodes", default=1, help="Number of nodes")
@click.option("--ngpu", default=8, help="Number of GPUs per node")
@click.option("--project", required=True, help="Project name")
@click.option("--cmd", required=True, help="Command to run")
@click.option("--qos", default="normal", help="QOS")
@click.option("--partition", default=None, help="SLURM partition (e.g., gpu, preprocess)")
@click.option("--time", default="7-00:00:00", help="Job time limit (default: 7-00:00:00)")
@click.option("--mem", default=None, help="Memory per node (e.g., 128G, 512G). If not specified, uses 128G for <8 GPUs, exclusive for 8 GPUs")
@click.option("--submit", is_flag=True, help="Submit job immediately")
@click.option("--conda-script", default="~/miniconda3/etc/profile.d/conda.sh", help="Conda script path")
@click.option(
    "--conda-env",
    required=True,
    help="Conda environment path",
)
@click.option(
    "--envs", default=[], multiple=True, help="Environment variables to expose"
)
@click.option(
    "--inject-shell",
    default=None,
    type=click.Path(),
    help="Additional shell commands to inject before the main command",
)
def main(
    nodes,
    ngpu,
    project,
    cmd,
    qos,
    partition,
    time,
    mem,
    submit,
    conda_script,
    conda_env,
    envs,
    inject_shell,
):
    """SLURM job submission tool"""
    project_root = Path.cwd().resolve()

    # Create base directory
    base_dir = Path(f"results/{project}")
    base_dir.mkdir(parents=True, exist_ok=True)

    base_dir = base_dir.resolve()

    # Save parameters
    params = {
        "nodes": nodes,
        "ngpu": ngpu,
        "cmd": cmd,
        "project": project,
        "qos": qos,
        "time": time,
        "mem": mem,
        "conda_env": conda_env,
    }
    with open(base_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Create logs directory
    (base_dir / "logs").mkdir(exist_ok=True)

    # Generate submit.sh
    if inject_shell is None:
        inject_shell = ""
    else:
        with open(inject_shell, "r") as f:
            inject_shell = f.read()

    generate_submit_script(
        base_dir, nodes, ngpu,
        project, qos, partition, time, mem, conda_script, conda_env, envs,
        project_root
    )

    # Generate run.sh
    generate_run_script(
        base_dir, ngpu, cmd, project_root, inject_shell
    )

    print(f"Job structure created at: {base_dir}")

    if submit:
        print("Submitting job...")
        submit_cmd = f"sbatch {base_dir}/submit.sh"
        result = subprocess.run(
            submit_cmd.split(), capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Job submitted successfully: {result.stdout.strip()}")
        else:
            print(f"Error submitting job: {result.stderr}")
    else:
        print(f"To submit the job, run: sbatch {base_dir}/submit.sh")


def generate_submit_script(base_dir, nodes, ngpu, project, qos, partition, time, mem, conda_script, conda_env, envs, project_root):
    expose_envs = "\n".join([f"export {env}" for env in envs])

    # Determine resource allocation
    if mem is None:
        # Auto-determine based on GPU count
        if ngpu < 8:
            additional = "#SBATCH --mem=128G\n#SBATCH --cpus-per-task=32"
        else:
            additional = "#SBATCH --exclusive"
    else:
        # Use specified memory
        additional = f"#SBATCH --mem={mem}\n#SBATCH --cpus-per-task=32"

    partition_line = f"#SBATCH --partition={partition}" if partition else ""

    submit_content = f"""#!/bin/bash
#SBATCH --job-name={project}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={ngpu}
{additional}
#SBATCH --time={time}
#SBATCH --qos={qos}
{partition_line}
#SBATCH --output={base_dir}/logs/%j/%j.out
#SBATCH --error={base_dir}/logs/%j/%j.err

set -e

source {conda_script}
conda activate {conda_env}

# Setup environment variables
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Find an available port by checking if it's open
find_available_port() {{
    local port=$1
    while true; do
        if ! nc -z $master_addr $port &>/dev/null; then
            echo $port
            return 0
        fi
        port=$((port + 1))
        if [ $port -gt 65535 ]; then
            port=29400  # Reset to start of range if we exceed max port
        fi
    done
}}

# Start with the random port and find the first available one
master_port=$(find_available_port 29400)

export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
# export NCCL_DEBUG=WARN
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
{'export ENABLE_INTRA_NODE_COMM=1' if nodes > 1 else ''}
export NCCL_IB_TIMEOUT=22
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# export NCCL_SOCKET_IFNAME=intranet
export NCCL_DEBUG=INFO

echo "Job started at: $(date)"
echo "Hostname: $(hostname)"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

{expose_envs}

# Print all environment variables
echo ""
echo "Environment variables:"
env | sort
echo ""

cd {project_root}

# Fix SLURM env variable conflicts inherited from global defaults
unset SLURM_MEM_PER_CPU
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK

srun --ntasks-per-node=1 \\
    --output={base_dir}/logs/%j/%t_%N.out \\
    --error={base_dir}/logs/%j/%t_%N.err \\
    bash {base_dir}/run.sh
"""
    submit_path = base_dir / "submit.sh"
    with open(submit_path, "w") as f:
        f.write(submit_content)
    submit_path.chmod(0o755)


def generate_run_script(
    base_dir, ngpu, cmd, project_root, inject_shell
):
    run_content = f"""#!/bin/bash
set -e

echo "Diagnostic Environment Variables:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $SLURM_JOB_NUM_NODES"
echo "NODE_RANK: $SLURM_NODEID"

echo "Running hardware diagnostics..."
nvidia-smi -L

echo "Attempting to ping $MASTER_ADDR..."
if ping -c 1 $MASTER_ADDR > /dev/null 2>&1; then
    echo "Successfully pinged $MASTER_ADDR"
else
    echo "Failed to ping $MASTER_ADDR"
    exit 1
fi

cd {project_root}

echo "Testing distributed communication..."
torchrun \\
    --nnodes=$SLURM_JOB_NUM_NODES \\
    --nproc_per_node={ngpu} \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    tools/lepton/test_communicate.py

{inject_shell}

torchrun \\
    --nnodes=$SLURM_JOB_NUM_NODES \\
    --nproc_per_node={ngpu} \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    {cmd}
"""
    run_path = base_dir / "run.sh"
    with open(run_path, "w") as f:
        f.write(run_content)
    run_path.chmod(0o755)


if __name__ == "__main__":
    main()
