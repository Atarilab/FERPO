"""Lossless packing of JAX simulator snapshots for fewer host transfers."""

from dataclasses import dataclass
from math import prod
from typing import Any


@dataclass(frozen=True)
class PackedQuerySnapshot:
    codec: Any
    buffers: tuple
    passthrough: tuple
    shared_info: dict


class QuerySnapshotCodec:
    """Pack strong arrays by dtype; retain weak scalar metadata verbatim.

    No physics fields are discarded. The codec retains shapes and a pytree
    definition, never the input arrays. Reset data shared by the caller stays
    on the device and is reattached after decoding.
    """

    def __init__(self, jax, state):
        self.jax = jax
        leaves, self.treedef = jax.tree.flatten(state)
        self.signature = self.state_signature(jax, state)
        self.packed_indices = []
        self.passthrough_indices = []
        groups = {}
        for index, leaf in enumerate(leaves):
            if isinstance(leaf, jax.Array) and not leaf.weak_type:
                slot = len(self.packed_indices)
                self.packed_indices.append(index)
                groups.setdefault(str(leaf.dtype), []).append((slot, tuple(leaf.shape)))
            else:
                self.passthrough_indices.append(index)
        self.leaf_count = len(leaves)
        groups = tuple(tuple(group) for group in groups.values())
        packed_count = len(self.packed_indices)

        def pack(arrays):
            return tuple(jax.numpy.concatenate([arrays[i].reshape(-1) for i, _ in group]) for group in groups)

        def unpack(buffers):
            # Do not capture the codec itself: a codec -> jit -> closure ->
            # codec cycle could be collected during an unrelated Warp graph
            # capture, when unloading its CUDA executable is illegal.
            arrays = [None] * packed_count
            for buffer, group in zip(buffers, groups):
                offset = 0
                for index, shape in group:
                    size = prod(shape)
                    arrays[index] = buffer[offset:offset + size].reshape(shape)
                    offset += size
            return tuple(arrays)

        self.pack_fn = jax.jit(pack)
        self.unpack_fn = jax.jit(unpack)

    @staticmethod
    def state_signature(jax, state):
        leaves, treedef = jax.tree.flatten(state)
        return treedef, tuple(
            (tuple(x.shape), str(x.dtype), x.weak_type) if isinstance(x, jax.Array)
            else (type(x), x) for x in leaves
        )

    def offload(self, state, shared_info):
        leaves = self.jax.tree.leaves(state)
        buffers = self.pack_fn(tuple(leaves[i] for i in self.packed_indices))
        cpu = self.jax.devices("cpu")[0]
        buffers = self.jax.device_put(buffers, cpu)
        passthrough = tuple(
            self.jax.device_put(leaves[i], cpu) if isinstance(leaves[i], self.jax.Array) else leaves[i]
            for i in self.passthrough_indices
        )
        # Bound GPU retention to one snapshot; one large copy per dtype replaces
        # hundreds of small, separately dispatched copies.
        self.jax.block_until_ready((buffers, passthrough))
        return PackedQuerySnapshot(self, buffers, passthrough, shared_info)

    def prepare(self, snapshot, device):
        arrays = self.unpack_fn(self.jax.device_put(snapshot.buffers, device))
        leaves = [None] * self.leaf_count
        for index, value in zip(self.packed_indices, arrays):
            leaves[index] = value
        for index, value in zip(self.passthrough_indices, snapshot.passthrough):
            leaves[index] = self.jax.device_put(value, device) if isinstance(value, self.jax.Array) else value
        state = self.jax.tree.unflatten(self.treedef, leaves)
        if snapshot.shared_info:
            state = state.replace(info={**state.info, **snapshot.shared_info})
        return state
