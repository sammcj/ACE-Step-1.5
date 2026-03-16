"""Sequence state tracking and simple block-based KV cache allocator."""

from collections import deque
from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Optional, Callable, Any

from acestep.customized_vllm.sampling import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """Represents one generation sequence with its token state and sampling config."""

    block_size = 256
    _counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams(),
                 is_unconditional: bool = False):
        self.seq_id = next(Sequence._counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.cfg_scale = sampling_params.cfg_scale
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.repetition_penalty = sampling_params.repetition_penalty
        self.is_unconditional = is_unconditional
        self.paired_seq: Optional["Sequence"] = None
        self.logits_processor: Optional[Any] = sampling_params.logits_processor
        self.logits_processor_update_state: Optional[Callable[[int], None]] = (
            sampling_params.logits_processor_update_state
        )

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


class BlockAllocator:
    """Simple block-based KV cache allocator (no prefix caching)."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_ids: deque[int] = deque(range(num_blocks))
        self.total = num_blocks

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.free_ids) >= num_blocks

    def allocate(self, seq: Sequence):
        """Allocate KV cache blocks for a new sequence."""
        for _ in range(seq.num_blocks):
            seq.block_table.append(self.free_ids.popleft())

    def deallocate(self, seq: Sequence):
        """Return all blocks held by a sequence to the free pool."""
        for bid in reversed(seq.block_table):
            self.free_ids.append(bid)
        seq.block_table.clear()

    def may_append(self, seq: Sequence):
        """Allocate a new block if the sequence is at a block boundary."""
        if len(seq) % self.block_size == 1 and len(seq) > seq.num_prompt_tokens:
            seq.block_table.append(self.free_ids.popleft())
