"""
Lock-free Single-Producer Single-Consumer (SPSC) ring buffer over shared memory.

Memory layout per buffer:
    [0:8]    write_index  (uint64) — only producer increments
    [8:16]   read_index   (uint64) — only consumer increments
    [16:24]  slot_size    (uint64) — bytes per slot (fixed at creation)
    [24:32]  slot_count   (uint64) — number of slots
    [32:]    slot_count * slot_size bytes of payload

Each slot layout:
    [0:4]    length  (uint32) — actual bytes written (0..slot_size-4)
    [4:]     payload bytes

Lock-free guarantee: one writer, one reader, no mutex.
The writer never touches read_index; the reader never touches write_index.
"""

import struct
from multiprocessing.shared_memory import SharedMemory

HEADER_SIZE = 32  # write_idx(8) + read_idx(8) + slot_size(8) + slot_count(8)
SLOT_HEADER_SIZE = 4  # uint32 length prefix per slot


class ShmRingBuffer:
    """SPSC ring buffer backed by OS shared memory."""

    def __init__(self, name: str, slot_count: int, slot_size: int, create: bool = False):
        """
        Parameters
        ----------
        name : str
            Shared memory segment name (OS-visible).
        slot_count : int
            Number of slots in the ring.
        slot_size : int
            Max payload bytes per slot (excluding the 4-byte length header).
        create : bool
            True = allocate new shared memory (producer side).
            False = attach to existing (consumer side).
        """
        self._name = name
        self._slot_count = slot_count
        # Internal slot size includes the 4-byte length prefix
        self._slot_size = slot_size + SLOT_HEADER_SIZE
        total_size = HEADER_SIZE + self._slot_count * self._slot_size

        if create:
            # Clean up any stale segment from a previous crash
            try:
                stale = SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._shm = SharedMemory(name=name, create=True, size=total_size)
            self._buf = self._shm.buf
            # Initialize header
            struct.pack_into("<Q", self._buf, 0, 0)   # write_index = 0
            struct.pack_into("<Q", self._buf, 8, 0)   # read_index  = 0
            struct.pack_into("<Q", self._buf, 16, self._slot_size)
            struct.pack_into("<Q", self._buf, 24, self._slot_count)
        else:
            self._shm = SharedMemory(name=name, create=False)
            self._buf = self._shm.buf
            stored_slot_size = struct.unpack_from("<Q", self._buf, 16)[0]
            stored_slot_count = struct.unpack_from("<Q", self._buf, 24)[0]
            if stored_slot_size != self._slot_size or stored_slot_count != self._slot_count:
                raise ValueError(
                    f"SHM config mismatch: expected slot_size={self._slot_size} count={self._slot_count}, "
                    f"got slot_size={stored_slot_size} count={stored_slot_count}"
                )

    @property
    def name(self) -> str:
        return self._name

    def _slot_offset(self, index: int) -> int:
        return HEADER_SIZE + (index % self._slot_count) * self._slot_size

    def _read_write_idx(self) -> int:
        return struct.unpack_from("<Q", self._buf, 0)[0]

    def _read_read_idx(self) -> int:
        return struct.unpack_from("<Q", self._buf, 8)[0]

    def _set_write_idx(self, val: int):
        struct.pack_into("<Q", self._buf, 0, val)

    def _set_read_idx(self, val: int):
        struct.pack_into("<Q", self._buf, 8, val)

    def write(self, data: bytes) -> bool:
        """
        Write data into the next slot. Non-blocking.

        Returns True on success, False if the ring is full (consumer too slow).
        Payload is silently truncated if it exceeds slot_size.
        """
        w = self._read_write_idx()
        r = self._read_read_idx()

        # Full when writer is one full lap ahead of reader
        if (w - r) >= self._slot_count:
            return False

        max_payload = self._slot_size - SLOT_HEADER_SIZE
        n = min(len(data), max_payload)

        off = self._slot_offset(w)
        struct.pack_into("<I", self._buf, off, n)
        self._buf[off + SLOT_HEADER_SIZE: off + SLOT_HEADER_SIZE + n] = data[:n]

        # Publish — consumer only sees the new slot after write_index advances
        self._set_write_idx(w + 1)
        return True

    def read(self) -> bytes | None:
        """
        Read the next available slot. Non-blocking.

        Returns payload bytes, or None if the ring is empty.
        """
        r = self._read_read_idx()
        w = self._read_write_idx()

        if r >= w:
            return None

        off = self._slot_offset(r)
        n = struct.unpack_from("<I", self._buf, off)[0]
        data = bytes(self._buf[off + SLOT_HEADER_SIZE: off + SLOT_HEADER_SIZE + n])

        # Advance — producer only sees the freed slot after read_index advances
        self._set_read_idx(r + 1)
        return data

    def available(self) -> int:
        """Number of unread slots."""
        return self._read_write_idx() - self._read_read_idx()

    def close(self):
        """Detach from shared memory (does not destroy it)."""
        self._shm.close()

    def unlink(self):
        """Destroy the shared memory segment. Call only from the creator."""
        self._shm.close()
        self._shm.unlink()
