#!/usr/bin/env python3
"""Check SLURM GPU partition usage and estimate how many nodes you can occupy."""

import subprocess
import sys
from collections import defaultdict

import click


def run_cmd(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_partition_info(partition: str) -> dict:
    """Get total/allocated/idle/down node counts for a partition."""
    output = run_cmd(
        f"sinfo -p {partition} -h -o '%D %A' --noconvert"
    )
    # Format: total_nodes allocated/idle
    parts = output.split()
    total = int(parts[0])
    alloc_idle = parts[1].split("/")
    allocated = int(alloc_idle[0])
    idle = int(alloc_idle[1])
    return {"total": total, "allocated": allocated, "idle": idle}


def get_user_node_usage(partition: str) -> dict[str, int]:
    """Get per-user node usage from running/pending jobs on the partition."""
    output = run_cmd(
        f"squeue -p {partition} -t RUNNING -h -o '%u %D'"
    )
    usage = defaultdict(int)
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        user = parts[0]
        nodes = int(parts[1])
        usage[user] += nodes
    return dict(usage)


def get_pending_jobs(partition: str) -> dict[str, int]:
    """Get per-user pending job node requests."""
    output = run_cmd(
        f"squeue -p {partition} -t PENDING -h -o '%u %D'"
    )
    pending = defaultdict(int)
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        user = parts[0]
        nodes = int(parts[1])
        pending[user] += nodes
    return dict(pending)


def get_current_user() -> str:
    return run_cmd("whoami")


@click.command()
@click.option(
    "--partition", "-p", type=str, default="gpu", help="SLURM partition name"
)
@click.option(
    "--max-ratio",
    type=float,
    default=1.5,
    help="Max ratio of fair share you're willing to take (e.g., 1.5 = 150%% of fair share)",
)
def main(partition: str, max_ratio: float):
    """Check GPU partition usage and estimate available nodes for you."""
    current_user = get_current_user()

    # Partition info
    info = get_partition_info(partition)
    total = info["total"]
    allocated = info["allocated"]
    idle = info["idle"]

    # Per-user usage
    user_usage = get_user_node_usage(partition)
    pending = get_pending_jobs(partition)
    active_users = set(user_usage.keys()) | set(pending.keys()) | {current_user}
    num_active_users = len(active_users)

    my_running = user_usage.get(current_user, 0)
    my_pending = pending.get(current_user, 0)

    # Fair share calculation
    fair_share = total / max(num_active_users, 1)
    max_nodes = min(int(fair_share * max_ratio), total)
    can_add = max(0, max_nodes - my_running)
    can_add = min(can_add, idle)  # Can't use more than what's idle

    # Print report
    print(f"{'=' * 55}")
    print(f"  SLURM Partition: {partition}")
    print(f"{'=' * 55}")
    print(f"  总节点数:     {total}")
    print(f"  已分配节点:   {allocated}")
    print(f"  空闲节点:     {idle}")
    print(f"  活跃用户数:   {num_active_users}")
    print(f"{'=' * 55}")

    # User table
    print(f"\n  {'用户':<16} {'运行中节点':>10} {'排队中节点':>10}")
    print(f"  {'-' * 36}")
    total_running = 0
    all_users = sorted(active_users)
    for user in all_users:
        running = user_usage.get(user, 0)
        total_running += running
        pend = pending.get(user, 0)
        marker = " ← 你" if user == current_user else ""
        print(f"  {user:<16} {running:>10} {pend:>10}{marker}")

    # Recommendation
    print(f"\n{'=' * 55}")
    print(f"  你的情况:")
    print(f"    当前运行:     {my_running} 个节点")
    print(f"    当前排队:     {my_pending} 个任务")
    print(f"    还可申请:     {total - total_running} 个节点")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
