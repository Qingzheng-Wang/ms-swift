import os
import time

import torch
import torch.distributed as dist

TEST_TOTAL_ALL_GATHER_SIZE = 100_000_000
TEST_TENSOR_SIZE = 100_000_000


def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return rank, local_rank, world_size


def all_gather_test(local_rank, world_size):
    tensor_size = TEST_TOTAL_ALL_GATHER_SIZE // world_size
    tensor = torch.randn(tensor_size, dtype=torch.float32).cuda(local_rank)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

    torch.cuda.synchronize()
    start_time = time.time()
    dist.all_gather(gathered, tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    bandwidth = (tensor_size * 4 * world_size) / (end_time - start_time) / 1e9  # GB/s
    return bandwidth


def all_reduce_test(local_rank, world_size):
    tensor = torch.randn(TEST_TENSOR_SIZE, dtype=torch.float32).cuda(local_rank)

    torch.cuda.synchronize()
    start_time = time.time()
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    bandwidth = (
        (TEST_TENSOR_SIZE * 4 * (world_size - 1)) / (end_time - start_time) / 1e9
    )  # GB/s
    return bandwidth


def matmul_test(local_rank, tensor_size=10000):
    # Create square matrices
    matrix_a = torch.randn(tensor_size, tensor_size, dtype=torch.bfloat16).cuda(
        local_rank
    )
    matrix_b = torch.randn(tensor_size, tensor_size, dtype=torch.bfloat16).cuda(
        local_rank
    )

    torch.cuda.synchronize()
    start_time = time.time()
    result = torch.matmul(matrix_a, matrix_b)
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate TFLOPS: 2 * N^3 operations for matrix multiplication
    flops = 2 * tensor_size**3
    tflops = flops / (end_time - start_time) / 1e12
    return tflops


def main():
    rank, local_rank, world_size = setup()
    all_gather_test(local_rank, world_size)
    all_reduce_test(local_rank, world_size)
    matmul_test(local_rank)  # Warmup

    all_gather_total = all_reduce_total = matmul_total = 0
    for _ in range(10):
        all_gather_total += all_gather_test(local_rank, world_size)
        all_reduce_total += all_reduce_test(local_rank, world_size)
        matmul_total += matmul_test(local_rank)

    if rank == 0:
        print(f"Local Rank: {local_rank}")
        print(f"World Size: {world_size}")
        print(f"Global Rank: {rank}")
        print(f"All-Gather Bandwidth: {all_gather_total / 10:.2f} GB/s")
        print(f"All-Reduce Bandwidth: {all_reduce_total / 10:.2f} GB/s")
        print(f"Matrix Multiplication Performance: {matmul_total / 10:.2f} TFLOPS")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
