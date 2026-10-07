"""Write updated GPT-2 weights back into an OpenAI TensorFlow checkpoint.

The reader already knows where each tensor lives: the `.index` file stores a
name, a byte offset, and a byte count. This writer uses those same records.
It never rewrites the index, the vocabulary, or the hyperparameters. Only the
float values inside the `.data` shard change.

The write is atomic: the shard is copied first, the copy is updated, and the
copy replaces the original only after every tensor has been checked and
written. A failed check leaves the original file untouched.
"""

import os
import shutil
from typing import NamedTuple

import numpy as np

from .tensorflow_checkpoint_reader import TensorflowCheckpointReader

_FLOAT32_DTYPE_ID = 1  # DT_FLOAT in TensorFlow's DataType enum


class _PlannedWrite(NamedTuple):
    checkpoint_name: str
    shard_id: int
    offset: int
    size: int
    payload: bytes


class TensorflowCheckpointWriter:
    """Overwrite the learned numbers inside an existing checkpoint."""

    def __init__(self, checkpoint_path):
        """`checkpoint_path` is the prefix returned by `TensorflowCheckpointReader.latest_checkpoint`,
        without `.index` or `.data-...`."""
        self._reader = TensorflowCheckpointReader(checkpoint_path)

    def overwrite(self, adjustable_weights):
        """Replace every listed tensor in the checkpoint with its current values."""
        planned_writes = self._plan_writes(adjustable_weights)

        data_path = self._reader.data_shard_path(planned_writes[0].shard_id)
        temporary_path = data_path + ".tmp"

        shutil.copy2(data_path, temporary_path)
        try:
            with open(temporary_path, "r+b") as shard:
                for write in planned_writes:
                    shard.seek(write.offset)
                    written = shard.write(write.payload)
                    if written != write.size:
                        raise IOError(
                            f"Wrote {written} bytes for {write.checkpoint_name}, expected {write.size}."
                        )
            os.replace(temporary_path, data_path)
        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise

    def _plan_writes(self, adjustable_weights):
        """Look up every tensor and refuse the whole save if any one does not fit."""
        planned_writes = []

        for weights in adjustable_weights:
            name = weights.checkpoint_name

            try:
                location = self._reader.locate_variable(name)
            except KeyError:
                raise KeyError(f"Checkpoint is missing {name!r}; nothing was written.") from None

            if location.dtype_id != _FLOAT32_DTYPE_ID:
                raise ValueError(
                    f"{name} has TensorFlow dtype {location.dtype_id}, "
                    f"expected DT_FLOAT ({_FLOAT32_DTYPE_ID}). Nothing was written."
                )

            payload = np.ascontiguousarray(weights.values, dtype=np.float32).tobytes()
            if len(payload) != location.size:
                raise ValueError(
                    f"{name} is {len(payload)} bytes in memory "
                    f"but {location.size} bytes on disk. Nothing was written."
                )

            planned_writes.append(
                _PlannedWrite(name, location.shard_id, location.offset, location.size, payload)
            )

        shard_ids = {write.shard_id for write in planned_writes}
        if len(shard_ids) != 1:
            raise ValueError(
                f"Refusing to write across multiple shards {sorted(shard_ids)}. "
                "Nothing was written."
            )
        return planned_writes
