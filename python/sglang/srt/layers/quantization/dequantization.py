"""
Utilities to manage the dequantization of weights.
"""

from typing import Optional

import torch

from sglang.srt.utils.common import is_cuda_alike

NVFP4_BLOCK_SIZE = 16
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

FLOAT4_E2M1_MAX = 6.0
FLOAT4_E2M1_MAX_RECIPROCAL = 1.0 / FLOAT4_E2M1_MAX

_FP4_E2M1_LUT_BY_DEVICE: dict[torch.device, torch.Tensor] = {}


def warmup_fp4_e2m1_lut(device: torch.device) -> torch.Tensor:
    """Materialize (and cache) the E2M1 lookup table on ``device``.

    Call once before CUDA graph capture (e.g. in process_weights_after_loading)
    so the subsequent dequantize_nvfp4 calls hit the cache instead of copying
    the table from host during capture.
    """
    cached = _FP4_E2M1_LUT_BY_DEVICE.get(device)
    if cached is None:
        cached = _FP4_E2M1_LUT.to(device=device)
        _FP4_E2M1_LUT_BY_DEVICE[device] = cached
    return cached


# Shared scratch buffers for dequantized BF16 weights, keyed by
# (shape, dtype, device). All MoE layers share weight shapes, so every layer
# reuses the same buffer instead of allocating a fresh one.
_DEQUANT_WEIGHT_BUFFERS: dict = {}


def get_dequant_weight_buffer(
    shape: tuple, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Get-or-create a reusable dequantized-weight buffer for ``shape``."""
    key = (tuple(shape), dtype, device)
    buf = _DEQUANT_WEIGHT_BUFFERS.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _DEQUANT_WEIGHT_BUFFERS[key] = buf
    return buf


def dequantize_nvfp4(
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_s2: Optional[torch.Tensor],
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 -> ``out_dtype``. ``w_q``: uint8 [..., out, in/2] packed e2m1
    (low nibble = even idx). ``w_s``: fp8 e4m3 [..., out, in/16] per-block.
    ``w_s2``: optional fp32 per-tensor / per-expert scalar that multiplies the
    per-block scale (ModelOpt / AMD Quark NVFP4).

    Assumes linear (non-swizzled) per-block scale layout. ``out`` optionally
    supplies a preallocated/reused output buffer (Triton path only) so repeated
    calls do not accumulate allocations inside a CUDA graph capture pool.
    """
    if _TRITON_AVAILABLE and is_cuda_alike():
        w_s2_eff = (
            w_s2
            if w_s2 is not None
            else torch.ones(1, device=w_q.device, dtype=torch.float32)
        )
        return _triton_dequantize_nvfp4(
            w_q.view(torch.uint8),
            w_s,
            w_s2_eff,
            out_dtype,
            NVFP4_BLOCK_SIZE,
            out=out,
        )

    device = w_q.device
    *batch, out_dim, half_in = w_q.shape
    in_dim = half_in * 2

    low = (w_q & 0xF).to(torch.int64)
    high = (w_q >> 4).to(torch.int64)
    # Use the device-cached LUT to avoid a per-call host->device copy (which is
    # not allowed during CUDA graph capture). Falls back to a one-time copy if
    # warmup_fp4_e2m1_lut was not called for this device.
    lut = _FP4_E2M1_LUT_BY_DEVICE.get(device)
    if lut is None:
        lut = warmup_fp4_e2m1_lut(device)
    deq = torch.empty(*batch, out_dim, in_dim, dtype=torch.float32, device=device)
    deq[..., 0::2] = lut[low]
    deq[..., 1::2] = lut[high]

    scale = w_s.to(torch.float32)
    if w_s2 is not None:
        w_s2_f32 = w_s2.to(torch.float32)
        # For 3D (MoE) inputs, w_s2 may be per-expert [E] or [E, 1].
        # Unsqueeze trailing dims so it broadcasts against scale [..., out, k//g].
        while w_s2_f32.dim() < scale.dim():
            w_s2_f32 = w_s2_f32.unsqueeze(-1)
        scale = scale * w_s2_f32
    scale = scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1)
    result = (deq * scale).to(out_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


# ---------------------------------------------------------------------------
# NVFP4 emulation utilities (ported from vLLM nvfp4_emulation_utils.py)
# Used when hardware lacks native NVFP4 support.
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _e2m1_inline(magnitude):
        """Binary-tree E2M1 lookup: maps a 3-bit magnitude to its float value
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] using bit decomposition."""
        b2 = (magnitude >> 2) & 1
        b1 = (magnitude >> 1) & 1
        b0 = magnitude & 1
        low_group = tl.where(
            b1 == 1, tl.where(b0 == 1, 1.5, 1.0), tl.where(b0 == 1, 0.5, 0.0)
        )
        high_group = tl.where(
            b1 == 1, tl.where(b0 == 1, 6.0, 4.0), tl.where(b0 == 1, 3.0, 2.0)
        )
        return tl.where(b2 == 1, high_group, low_group)

    @triton.jit
    def _dequantize_nvfp4_kernel(
        fp4_ptr,
        scale_ptr,
        global_scale_ptr,
        output_ptr,
        rows_per_batch: tl.constexpr,
        num_blocks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        has_batch_global_scale: tl.constexpr,
        TILE_BLOCKS: tl.constexpr,
    ):
        """Dequantize packed NVFP4 (linear/non-swizzled scale) to the output
        dtype. 2D grid (rows x tiles)."""
        BLOCK_PACKED: tl.constexpr = BLOCK_SIZE // 2

        row_idx = tl.program_id(0)
        tile_idx = tl.program_id(1)

        if has_batch_global_scale:
            batch_idx = row_idx // rows_per_batch
            global_scale = tl.load(global_scale_ptr + batch_idx).to(tl.float32)
        else:
            global_scale = tl.load(global_scale_ptr).to(tl.float32)

        # Compute flat element offsets in int64: for large per-expert weights
        # int32 index arithmetic overflows and faults. scale_row_offset stays
        # well within int32.
        row_idx_i64 = row_idx.to(tl.int64)
        fp4_row_offset = row_idx_i64 * num_blocks * BLOCK_PACKED
        scale_row_offset = row_idx * num_blocks
        output_row_offset = row_idx_i64 * num_blocks * BLOCK_SIZE

        start_block = tile_idx * TILE_BLOCKS
        block_offsets = tl.arange(0, TILE_BLOCKS)
        block_mask = (start_block + block_offsets) < num_blocks

        raw_scales = tl.load(
            scale_ptr + scale_row_offset + start_block + block_offsets,
            mask=block_mask,
            other=0,
        )
        scale_f32 = tl.cast(raw_scales, tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_values = (scale_f32 * global_scale)[:, None]

        packed_offsets = tl.arange(0, BLOCK_PACKED)[None, :]
        byte_indices = (
            fp4_row_offset
            + (start_block + block_offsets[:, None]) * BLOCK_PACKED
            + packed_offsets
        )
        elem_mask = block_mask[:, None]
        raw_bytes = tl.load(fp4_ptr + byte_indices, mask=elem_mask, other=0)

        low_nibble = raw_bytes & 0x0F
        high_nibble = (raw_bytes >> 4) & 0x0F

        low_val = _e2m1_inline(low_nibble & 0x07)
        low_sign = (low_nibble >> 3) & 1
        low_result = tl.where(low_sign == 1, -low_val, low_val) * scale_values

        high_val = _e2m1_inline(high_nibble & 0x07)
        high_sign = (high_nibble >> 3) & 1
        high_result = tl.where(high_sign == 1, -high_val, high_val) * scale_values

        result = tl.interleave(low_result, high_result)

        elem_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
        out_indices = (
            output_row_offset
            + (start_block + block_offsets[:, None]) * BLOCK_SIZE
            + elem_offsets
        )
        tl.store(output_ptr + out_indices, result, mask=block_mask[:, None])

    def _triton_dequantize_nvfp4(
        tensor_fp4: torch.Tensor,
        tensor_sf: torch.Tensor,
        global_scale: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int = NVFP4_BLOCK_SIZE,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Triton NVFP4 weight dequant (linear scale layout), 2D or 3D.

        ``out`` lets the caller pass a preallocated/reused buffer so repeated
        calls (e.g. one per MoE layer) do not accumulate allocations inside a
        CUDA graph capture pool.
        """
        assert tensor_fp4.dtype == torch.uint8
        is_3d = tensor_fp4.ndim == 3
        if is_3d:
            dim0, m_per_batch, packed_k = tensor_fp4.shape
            tensor_fp4_2d = tensor_fp4.reshape(-1, packed_k)
            tensor_sf_2d = tensor_sf.reshape(-1, tensor_sf.shape[-1])
            total_rows = dim0 * m_per_batch
        else:
            m_per_batch, packed_k = tensor_fp4.shape
            tensor_fp4_2d = tensor_fp4
            tensor_sf_2d = tensor_sf
            total_rows = m_per_batch

        k = packed_k * 2
        num_blocks = k // block_size

        if out is None:
            out = torch.empty(total_rows, k, dtype=out_dtype, device=tensor_fp4.device)
        out_2d = out.reshape(total_rows, k)

        scale_raw = tensor_sf_2d.contiguous().view(torch.uint8)
        global_scale = global_scale.reshape(-1).to(torch.float32).contiguous()

        # Three supported global_scale variations:
        #   1                    - one scalar for the whole tensor
        #   dim0 (=num_experts)  - one scalar per expert / batch element (3-D only)
        #   total_rows           - one scalar per output row (e.g. distinct gate/up
        #                          scales for fused w13 where gate != up (e.g. nvidia/GLM-5.1-NVFP4)
        # The kernel's has_batch_global_scale + rows_per_batch handles all three:
        # rows_per_batch=m_per_batch -> one scale per expert
        # rows_per_batch=1          -> one scale per output row
        n_scales = global_scale.numel()
        if n_scales == total_rows:
            has_batch_gscale = True
            rows_per_batch = 1
        elif is_3d and n_scales > 1:
            has_batch_gscale = True
            rows_per_batch = m_per_batch
        else:
            has_batch_gscale = False
            rows_per_batch = total_rows

        # Cap the per-program tile so large K is split across the
        # grid's tile dimension instead of one giant [TILE_BLOCKS, BLOCK_SIZE]
        # tensor per program.
        tile_blocks = min(64, triton.next_power_of_2(num_blocks))
        num_tiles = (num_blocks + tile_blocks - 1) // tile_blocks
        grid = (total_rows, num_tiles)
        _dequantize_nvfp4_kernel[grid](
            tensor_fp4_2d,
            scale_raw,
            global_scale,
            out_2d,
            rows_per_batch,
            num_blocks,
            block_size,
            has_batch_gscale,
            tile_blocks,
        )
        if is_3d:
            return out_2d.reshape(dim0, m_per_batch, k)
        return out_2d

    @triton.jit
    def _round_to_fp4(x):
        """Round float values to nearest E2M1 representable value."""
        sign = tl.where(x < 0.0, -1.0, 1.0)
        abs_x = tl.abs(x)
        result = tl.where(abs_x > 5.0, 6.0, 0.0)
        result = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 4.0, result)
        result = tl.where((abs_x > 2.5) & (abs_x < 3.5), 3.0, result)
        result = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 2.0, result)
        result = tl.where((abs_x > 1.25) & (abs_x < 1.75), 1.5, result)
        result = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 1.0, result)
        result = tl.where((abs_x > 0.25) & (abs_x < 0.75), 0.5, result)
        return result * sign

    @triton.jit
    def _nvfp4_quant_dequant_kernel(
        input_ptr,
        output_ptr,
        global_scale_ptr,
        k: tl.constexpr,
        num_blocks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        FP4_MAX_RECIPROCAL: tl.constexpr,
        TILE_BLOCKS: tl.constexpr,
    ):
        """Fused NVFP4 quantize-dequantize kernel (2D grid: rows × tiles)."""
        row_idx = tl.program_id(0)
        tile_idx = tl.program_id(1)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        row_offset = row_idx * k

        start_block = tile_idx * TILE_BLOCKS
        block_offsets = tl.arange(0, TILE_BLOCKS)
        block_mask = (start_block + block_offsets) < num_blocks

        indices = (
            row_offset
            + (start_block + block_offsets[:, None]) * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE)[None, :]
        )
        mask_2d = block_mask[:, None]
        x = tl.load(input_ptr + indices, mask=mask_2d, other=0.0).to(tl.float32)

        vec_max = tl.max(tl.abs(x), axis=1)
        scale = global_scale * (vec_max * FP4_MAX_RECIPROCAL)
        scale = tl.clamp(scale, -448.0, 448.0)
        scale = scale.to(tl.float8e4nv).to(tl.float32)

        output_scale = tl.where(scale == 0.0, 0.0, global_scale / scale)[:, None]
        scaled_x = tl.clamp(x * output_scale, -6.0, 6.0)
        fp4_val = _round_to_fp4(scaled_x)

        dequant_scale = (scale / global_scale)[:, None]
        result = fp4_val * dequant_scale

        tl.store(output_ptr + indices, result, mask=mask_2d)

    def _triton_nvfp4_quant_dequant(
        x: torch.Tensor,
        global_scale: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Triton-accelerated NVFP4 quantize-dequantize (CUDA and ROCm)."""
        x_m, x_k = x.shape
        assert x_k % block_size == 0
        output_dtype = x.dtype
        num_blocks = x_k // block_size
        output = torch.empty(x_m, x_k, dtype=output_dtype, device=x.device)
        tile_blocks = min(64, triton.next_power_of_2(num_blocks))
        num_tiles = (num_blocks + tile_blocks - 1) // tile_blocks
        grid = (x_m, num_tiles)
        _nvfp4_quant_dequant_kernel[grid](
            x,
            output,
            global_scale,
            x_k,
            num_blocks,
            block_size,
            FLOAT4_E2M1_MAX_RECIPROCAL,
            tile_blocks,
        )
        return output


def _get_reciprocal(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (x + (x == 0) * 1e8)


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round float values to the nearest E2M1 representable value (in-place on abs).

    Caller must pass a fresh tensor (not aliased elsewhere) since this mutates x.
    """
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def ref_nvfp4_quant(
    x: torch.Tensor, global_scale: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE
):
    """Per-group NVFP4 quantization.

    ``global_scale`` is the inverse activation scale (1/input_scale), consistent
    with how SGLang stores ``layer.input_scale_inv`` and how ``fp4_quantize`` uses it.

    Returns (fp4_vals_f32, block_scales_f32).
    """
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // block_size, block_size))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * FLOAT4_E2M1_MAX_RECIPROCAL)
    scale = torch.clamp(scale, max=448, min=-448)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = _get_reciprocal(scale * _get_reciprocal(global_scale))
    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def ref_nvfp4_quant_dequant(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """NVFP4 quant-dequant

    ``global_scale`` must be the inverse activation scale (1/input_scale).
    Dispatches to Triton on CUDA/ROCm, falls back to pure PyTorch otherwise.
    """
    if _TRITON_AVAILABLE and is_cuda_alike():
        return _triton_nvfp4_quant_dequant(x, global_scale, block_size)

    x_m, x_k = x.shape
    output_dtype = x.dtype
    x_fp4, x_blockscale = ref_nvfp4_quant(x, global_scale, block_size)
    x_fp4 = x_fp4.reshape(x_m, x_k // block_size, block_size)
    x_blockscale = x_blockscale.unsqueeze(-1) / global_scale
    x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)
    return x_dq


def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Software emulation of NVFP4 GEMM via dequant -> BF16 matmul.

    ``input_global_scale`` must be the inverse activation scale (``layer.input_scale_inv``).
    ``weight_scale`` must be in linear (non-swizzled) layout, the emulation branch in
    process_weights_after_loading() skips swizzling to preserve this invariant.
    """
    if output_dtype is None:
        output_dtype = x.dtype

    x_dq = ref_nvfp4_quant_dequant(x, input_global_scale)

    w_dq = dequantize_nvfp4(
        weight.view(torch.uint8),
        weight_scale,
        weight_global_scale,
        out_dtype=output_dtype,
    )

    return torch.matmul(x_dq, w_dq.t())
