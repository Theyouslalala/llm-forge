import os
from typing import Optional

import torch
import torch.distributed as dist

from .logger import get_logger

logger = get_logger(__name__)


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    if not dist.is_available():
        return 0, 1, 0

    if "RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)

    logger.info(f"Distributed: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt
