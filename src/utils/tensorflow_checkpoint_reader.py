"""
Pure-Python TF v1 checkpoint reader.

Tensorflow is not a required dependency now.

TF v1 checkpoint bundle layout
────────────────────────────────
  <prefix>.index              – SSTable; key = variable name,
                                value = BundleEntryProto (protobuf)
  <prefix>.data-NNNNN-of-MMMMM – raw binary shard; tensors stored
                                  at the byte offsets recorded in the index

SSTable layout (tensorflow/core/lib/io/format.h)
────────────────────────────────────────────────
  [data block 0]
  ...
  [data block N]
  [metaindex block]   ← we skip this
  [index block]       ← entry per data block: key → BlockHandle
  [footer: 48 bytes]
    metaindex_handle (≤20 B varint-encoded BlockHandle)
    index_handle     (≤20 B varint-encoded BlockHandle)
    padding          (zeroes)
    magic u64 LE     = 0xdb4775248b80fb57

Each block ends with a 5-byte trailer (1-byte type + 4-byte crc32c).
Type 0 = uncompressed (GPT-2 checkpoints are uncompressed).

Block entry encoding (prefix-compressed keys):
  varint32 shared_key_len
  varint32 non_shared_key_len
  varint32 value_len
  bytes    key_delta[non_shared_key_len]
  bytes    value[value_len]
  …
  uint32   restart[0]  ← offsets of entries where shared_key_len == 0
  …
  uint32   num_restarts

BundleEntryProto (tensorflow/core/protobuf/saver.proto, fields we need)
────────────────────────────────────────────────────────────────────────
  field 1  (varint)  dtype     – DataType enum
  field 2  (LEN)     shape     – TensorShapeProto
  field 3  (varint)  shard_id
  field 4  (varint)  offset    – byte offset in the raw shard file
  field 5  (varint)  size      – byte count in the raw shard file

TensorShapeProto / Dim
  TensorShapeProto.field 2 (LEN, repeated) – Dim
  Dim.field 1 (varint)                     – size (int64, may be negative)
"""

import os
import re
import struct

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────

_MAGIC = 0xDB4775248B80FB57
_BLOCK_TRAILER = 5  # 1-byte compression type + 4-byte crc32c

# TF DataType enum → numpy dtype (only the numeric types picoGPT will encounter)
_DTYPE_MAP = {
    1: np.float32,    # DT_FLOAT
    2: np.float64,    # DT_DOUBLE
    3: np.int32,      # DT_INT32
    4: np.uint8,      # DT_UINT8
    5: np.int16,      # DT_INT16
    6: np.int8,       # DT_INT8
    9: np.int64,      # DT_INT64
    10: np.bool_,     # DT_BOOL
    17: np.float16,   # DT_HALF
    19: np.complex64, # DT_COMPLEX64
    22: np.complex128,# DT_COMPLEX128
}


class _IndexFile:
    """Lazy-loading SSTable wrapper for a TF checkpoint .index file."""

    @staticmethod
    def _varint(buf: bytes, pos: int):
        """Read a protobuf base-128 varint. Returns (value, new_pos)."""
        result = shift = 0
        while True:
            b = buf[pos]; pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result, pos
            shift += 7

    @staticmethod
    def _decode_handle(buf: bytes, pos: int):
        """Decode a BlockHandle (offset, size). Returns (offset, size, new_pos)."""
        off, pos = _IndexFile._varint(buf, pos)
        sz,  pos = _IndexFile._varint(buf, pos)
        return off, sz, pos

    @staticmethod
    def _read_block(f, blk_offset: int, blk_size: int) -> bytes:
        """Read and return the raw (uncompressed) bytes of a block."""
        f.seek(blk_offset)
        buf = f.read(blk_size + _BLOCK_TRAILER)
        if buf[blk_size] != 0:
            raise NotImplementedError(
                f"Compressed blocks (type={buf[blk_size]}) are not supported"
            )
        return buf[:blk_size]

    @staticmethod
    def _iter_block(block: bytes):
        """Yield (key: bytes, value: bytes) for every entry in a data block."""
        n = len(block)
        num_restarts = struct.unpack_from("<I", block, n - 4)[0]
        limit = n - 4 - 4 * num_restarts
        pos = 0
        key = b""
        while pos < limit:
            shared,     pos = _IndexFile._varint(block, pos)
            non_shared, pos = _IndexFile._varint(block, pos)
            vlen,       pos = _IndexFile._varint(block, pos)
            key = key[:shared] + block[pos:pos + non_shared]
            pos += non_shared
            val = block[pos:pos + vlen]
            pos += vlen
            yield key, val

    def __init__(self, path: str):
        self._f = open(path, "rb")
        self._f.seek(0, 2)
        fsize = self._f.tell()
        self._f.seek(fsize - 48)
        footer = self._f.read(48)
        magic = struct.unpack_from("<Q", footer, 40)[0]
        if magic != _MAGIC:
            raise ValueError(
                f"{path!r} does not look like a TF SSTable "
                f"(magic={hex(magic)}, expected {hex(_MAGIC)})"
            )
        # Skip metaindex handle, keep index handle
        pos = 0
        _, _, pos = self._decode_handle(footer, pos)
        self._idx_off, self._idx_sz, _ = self._decode_handle(footer, pos)
        self._index: list | None = None  # [(last_key, blk_off, blk_sz), …]

    # ── private ──────────────────────────────────────────────────────────────

    def _load_index(self):
        blk = self._read_block(self._f, self._idx_off, self._idx_sz)
        self._index = []
        for k, v in self._iter_block(blk):
            off, sz, _ = self._decode_handle(v, 0)
            self._index.append((k, off, sz))

    def _find_block(self, key: bytes):
        """Return the data block that should contain *key*, or None."""
        if self._index is None:
            self._load_index()
        idx = self._index
        # Binary search: find first entry whose separator key >= search key
        lo, hi = 0, len(idx)
        while lo < hi:
            mid = (lo + hi) // 2
            if idx[mid][0] < key:
                lo = mid + 1
            else:
                hi = mid
        # In an SSTable the separator key may be >= all keys in its block,
        # so also try lo-1 in case the key lives in the previous block.
        candidates = [lo] if lo == 0 else [lo - 1, lo]
        for i in candidates:
            if i < len(idx):
                yield self._read_block(self._f, idx[i][1], idx[i][2])

    # ── public ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> bytes | None:
        """Return the raw BundleEntryProto bytes for *name*, or None."""
        key = name.encode()
        for block in self._find_block(key):
            for k, v in self._iter_block(block):
                if k == key:
                    return v
        return None

    def items(self):
        """Yield (name, BundleEntryProto_bytes) for every real variable."""
        if self._index is None:
            self._load_index()
        for _, off, sz in self._index:
            for k, v in self._iter_block(self._read_block(self._f, off, sz)):
                name = k.decode()
                if name:  # skip the empty-key BundleHeaderProto entry
                    yield name, v

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class TensorflowCheckpointReader:
    """Pure-Python reader for TensorFlow v1 checkpoint bundles."""

    @staticmethod
    def _signed_varint(buf: bytes, pos: int):
        """Read a varint then sign-extend to a Python int."""
        v, pos = _IndexFile._varint(buf, pos)
        if v >= (1 << 63):
            v -= (1 << 64)
        return v, pos

    @staticmethod
    def _parse_dim(buf: bytes) -> int:
        """Parse TensorShapeProto.Dim → integer size."""
        pos, n = 0, len(buf)
        while pos < n:
            tag, pos = _IndexFile._varint(buf, pos)
            fn, wt = tag >> 3, tag & 7
            if wt == 0:
                v, pos = TensorflowCheckpointReader._signed_varint(buf, pos)
                if fn == 1:
                    return v
            elif wt == 2:
                l, pos = _IndexFile._varint(buf, pos); pos += l
            elif wt == 1:
                pos += 8
            elif wt == 5:
                pos += 4
            else:
                break
        return 0

    @staticmethod
    def _parse_shape(buf: bytes) -> list:
        """Parse TensorShapeProto → list of dim sizes."""
        pos, n, dims = 0, len(buf), []
        while pos < n:
            tag, pos = _IndexFile._varint(buf, pos)
            fn, wt = tag >> 3, tag & 7
            if wt == 2:
                l, pos = _IndexFile._varint(buf, pos)
                sub = buf[pos:pos + l]; pos += l
                if fn == 2:
                    dims.append(TensorflowCheckpointReader._parse_dim(sub))
            elif wt == 0:
                _, pos = _IndexFile._varint(buf, pos)
            elif wt == 1:
                pos += 8
            elif wt == 5:
                pos += 4
            else:
                break
        return dims

    @staticmethod
    def _parse_bundle_entry(buf: bytes):
        """Parse BundleEntryProto → (dtype_id, shape, shard_id, offset, size)."""
        pos, n = 0, len(buf)
        dtype_id = 1; shape = []; shard_id = 0; offset = 0; size = 0
        while pos < n:
            tag, pos = _IndexFile._varint(buf, pos)
            fn, wt = tag >> 3, tag & 7
            if wt == 0:
                v, pos = _IndexFile._varint(buf, pos)
                if fn == 1:   dtype_id = v
                elif fn == 3: shard_id = v
                elif fn == 4: offset   = v
                elif fn == 5: size     = v
            elif wt == 2:
                l, pos = _IndexFile._varint(buf, pos)
                sub = buf[pos:pos + l]; pos += l
                if fn == 2:
                    shape = TensorflowCheckpointReader._parse_shape(sub)
            elif wt == 1:
                pos += 8
            elif wt == 5:
                pos += 4
            else:
                break
        return dtype_id, shape, shard_id, offset, size

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    @staticmethod
    def latest_checkpoint(model_dir: str) -> str | None:
        """Return the path of the latest checkpoint in *model_dir*, or None."""
        ckpt_file = os.path.join(model_dir, "checkpoint")
        if not os.path.exists(ckpt_file):
            return None
        with open(ckpt_file) as f:
            content = f.read()
        m = re.search(r'model_checkpoint_path:\s*"([^"]+)"', content)
        if not m:
            return None
        path = m.group(1)
        if not os.path.isabs(path):
            path = os.path.join(model_dir, path)
        return path

    def list_variables(self) -> list:
        """Return [(name, shape), …] for every variable in the checkpoint."""
        with _IndexFile(self.checkpoint_path + ".index") as idx:
            return [
                (name, self._parse_bundle_entry(raw)[1])
                for name, raw in idx.items()
            ]

    def load_variable(self, name: str) -> np.ndarray:
        """Load a single variable from the checkpoint and return it as ndarray."""
        with _IndexFile(self.checkpoint_path + ".index") as idx:
            raw_entry = idx.get(name)
        if raw_entry is None:
            raise KeyError(
                f"Variable {name!r} not found in {self.checkpoint_path!r}"
            )

        dtype_id, shape, shard_id, offset, size = self._parse_bundle_entry(raw_entry)
        np_dtype = _DTYPE_MAP.get(dtype_id, np.float32)

        # Locate the correct data shard (raw binary file)
        ckpt_dir  = os.path.dirname(self.checkpoint_path) or "."
        base_name = os.path.basename(self.checkpoint_path)
        shards = sorted(
            f for f in os.listdir(ckpt_dir) if f.startswith(base_name + ".data")
        )
        if shard_id >= len(shards):
            raise FileNotFoundError(
                f"Shard {shard_id} not found; available shards: {shards}"
            )
        data_path = os.path.join(ckpt_dir, shards[shard_id])

        # Read raw bytes from the shard at the recorded offset
        with open(data_path, "rb") as f:
            f.seek(offset)
            raw_bytes = f.read(size)

        arr = np.frombuffer(raw_bytes, dtype=np_dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr.copy()  # make writable
