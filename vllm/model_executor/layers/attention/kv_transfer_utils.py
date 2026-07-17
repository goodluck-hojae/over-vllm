# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import time
from collections.abc import Callable
from functools import wraps

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.logger import init_logger
from vllm.utils.torch_utils import _resolve_layer_name

logger = init_logger(__name__)


def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # Import at runtime to avoid circular dependency
    from vllm.model_executor.layers.attention.attention import get_attention_context

    # Inspect the signature ONCE when the decorator is applied.
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find the index of 'layer_name' parameter.
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(*args, **kwargs)

        layer_name = _resolve_layer_name(args[layer_name_index])

        # Extract attention context (metadata, layer, kv_cache, layer_slot_mapping)
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            return func(*args, **kwargs)

        is_prefill = False
        is_prefilling = getattr(attn_metadata, "is_prefilling", None)
        if is_prefilling is not None:
            is_prefill = bool(is_prefilling.any().item())
        else:
            query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            query_boundaries = query_start_loc_cpu
            if query_boundaries is None:
                query_boundaries = query_start_loc
            if query_boundaries is not None and query_boundaries.numel() > 1:
                query_lens = query_boundaries[1:] - query_boundaries[:-1]
                is_prefill = bool((query_lens > 1).any().item())

        # Wait for KV layer on entry
        wait_start = time.perf_counter()
        if is_prefill:
            logger.info("COMPUTE enter layer=%s", layer_name)
        connector.wait_for_layer_load(layer_name)
        wait_ms = (time.perf_counter() - wait_start) * 1000.0
        if is_prefill:
            logger.info("COMPUTE resume layer=%s wait_ms=%.3f", layer_name, wait_ms)

        # Execute the function
        compute_start = time.perf_counter()
        result = func(*args, **kwargs)
        compute_ms = (time.perf_counter() - compute_start) * 1000.0
        if is_prefill:
            logger.info("COMPUTE done layer=%s compute_ms=%.3f", layer_name,
                        compute_ms)

        # Save KV cache layer on exit
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)

        return result

    return wrapper
