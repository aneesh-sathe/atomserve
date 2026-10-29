# Custom Metal kernels for the two hottest spots of a decode step.
#
# At decode, a step does very little arithmetic per kernel: what costs time is
# the number of kernels (each launch has a fixed cost) and memory traffic.
# PyTorch runs each small op as its own kernel, so these fuse whole sequences
# of ops into one:
#
#   qk_norm_rope_store       per-head RMSNorm of q and k, RoPE on both, and the
#                            K/V write into their pool slots: ~16 launches -> 1
#   paged_decode_attention   attention for single-query sequences that reads
#                            K/V straight from the paged pool through the block
#                            tables: no gather copy, two launches for the batch
#   topk_rows                the sampler's top-k over the 151,936-entry vocabulary,
#                            several times faster than torch.topk on MPS
#   silu_mul                 SwiGLU's silu(gate) * up in one pass
#
# model.py keeps the plain-PyTorch versions of the same math (the readable
# reference, also used on CPU); tests check that both agree. The kernels
# compute in fp32 and round once at the end. They assume head_dim == 128,
# which holds for every Qwen3 size.
#
# Compiled from source at import time by torch.mps.compile_shader (a few ms).

import torch

from atomserve.config import ModelConfig

HEAD_DIM = 128

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// One threadgroup per (token, head slot), head slots ordered q heads, k heads,
// v heads as they come out of the stacked qkv projection. 64 threads per group;
// thread i owns the RoPE pair (i, i + 64) of the head.
template <typename T>
kernel void qk_norm_rope_store(
    device T* q_out             [[buffer(0)]],   // [tokens, H, 128]
    device T* k_pool            [[buffer(1)]],   // [slots, KV, 128], one layer
    device T* v_pool            [[buffer(2)]],   // [slots, KV, 128], one layer
    constant T* qkv             [[buffer(3)]],   // [tokens, H + 2 KV, 128]
    constant T* q_norm          [[buffer(4)]],   // [128]
    constant T* k_norm          [[buffer(5)]],   // [128]
    constant long* positions    [[buffer(6)]],   // [tokens]
    constant long* slot_mapping [[buffer(7)]],   // [tokens]
    constant float* inv_freq    [[buffer(8)]],   // [64]
    constant long& num_heads    [[buffer(9)]],
    constant long& num_kv_heads [[buffer(10)]],
    constant float& eps         [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],   // (head slot, token)
    uint2 tid   [[thread_position_in_threadgroup]],
    uint sg     [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    const uint D = 128, half_d = 64, i = tid.x;
    const uint H = num_heads, KV = num_kv_heads, head = group.x, t = group.y;
    constant T* x = qkv + (t * (H + 2 * KV) + head) * D;
    float a = float(x[i]), b = float(x[i + half_d]);

    if (head >= H + KV) {  // a value head: no norm, no rotation, just store it
        device T* dst = v_pool + (slot_mapping[t] * KV + (head - H - KV)) * D;
        dst[i] = T(a);
        dst[i + half_d] = T(b);
        return;  // uniform across the threadgroup, so no barrier is skipped unevenly
    }

    // RMSNorm over the head's 128 values: reduce within each simdgroup, then across the two.
    threadgroup float partial[2];
    float sum_sq = simd_sum(a * a + b * b);
    if (lane == 0) partial[sg] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = rsqrt((partial[0] + partial[1]) / float(D) + eps);
    constant T* w = head < H ? q_norm : k_norm;
    a *= scale * float(w[i]);
    b *= scale * float(w[i + half_d]);

    // RoPE: rotate the pair by position * frequency_i (precise trig: angles reach ~4e4 rad).
    float angle = float(positions[t]) * inv_freq[i];
    float c = precise::cos(angle), s = precise::sin(angle);
    device T* dst = head < H ? q_out + (t * H + head) * D
                             : k_pool + (slot_mapping[t] * KV + (head - H)) * D;
    dst[i] = T(a * c - b * s);
    dst[i + half_d] = T(b * c + a * s);
}

// Decode attention in two passes (the PagedAttention "v2" scheme):
//
// 1. partial: one threadgroup per (kv head, sequence, partition of 512 keys).
//    It handles all G query heads that share the kv head (GQA), so every K/V
//    row is read once. Each simdgroup takes 32 keys at a time, one per lane:
//    lane l scores key l against the G queries, then the simdgroup updates an
//    online softmax and accumulates V, lane l owning dims 4l..4l+3. Results
//    are unnormalised (running max m, sum l, weighted sum of V).
// 2. reduce: merges the partitions of each (sequence, head) into the output.
//
// Splitting long contexts into partitions keeps the GPU busy even for a
// single sequence. With a single partition (short contexts) there is nothing
// to merge: pass 1 writes the normalised output itself and pass 2 is skipped.
template <typename T, uint G>
kernel void decode_attention_partial(
    device float* part_m        [[buffer(0)]],   // [B, H, P]
    device float* part_l        [[buffer(1)]],   // [B, H, P]
    device float4* part_acc     [[buffer(2)]],   // [B, H, P, 128 / 4]
    device vec<T, 4>* out       [[buffer(3)]],   // [B, H, 128 / 4], written only if P == 1
    constant T* q               [[buffer(4)]],   // [B, H, 128]
    constant T* k_pool          [[buffer(5)]],   // [slots, KV, 128], one layer
    constant T* v_pool          [[buffer(6)]],   // [slots, KV, 128], one layer
    constant long* block_tables [[buffer(7)]],   // [B, max_blocks]
    constant long* context_lens [[buffer(8)]],   // [B]
    constant long& max_blocks   [[buffer(9)]],
    constant long& block_size   [[buffer(10)]],
    constant long& num_kv_heads [[buffer(11)]],
    constant long& partition    [[buffer(12)]],
    constant float& scale       [[buffer(13)]],
    uint3 group  [[threadgroup_position_in_grid]],   // (kv head, sequence, partition)
    uint3 groups [[threadgroups_per_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint sg      [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]])
{
    constexpr uint D = 128, NSG = 4;
    const uint kv_head = group.x, seq = group.y, part = group.z, P = groups.z;
    const uint KV = num_kv_heads, H = KV * G;
    const uint len = context_lens[seq];
    const uint start = part * partition, end = min(start + uint(partition), len);
    constant long* table = block_tables + seq * max_blocks;

    threadgroup float4 q_sh[G][D / 4];  // the G queries, pre-scaled
    for (uint i = tid; i < G * D; i += NSG * 32)
        ((threadgroup float*)q_sh)[i] = float(q[(seq * H + kv_head * G) * D + i]) * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m[G], l[G];
    float4 acc[G];
    for (uint g = 0; g < G; ++g) { m[g] = -INFINITY; l[g] = 0.0f; acc[g] = 0.0f; }

    for (uint c = start + sg * 32; c < end; c += NSG * 32) {
        const uint j = c + lane;
        const bool valid = j < end;
        const uint slot = valid ? uint(table[j / block_size] * block_size + j % block_size) : 0;

        float p[G];  // this lane's key: score against each query, then its softmax weight
        for (uint g = 0; g < G; ++g) p[g] = 0.0f;
        constant T* k_base = k_pool + (ulong(slot) * KV + kv_head) * D;
        constant vec<T, 4>* k_row = (constant vec<T, 4>*)k_base;
        for (uint d = 0; d < D / 4; ++d) {
            float4 k4 = float4(k_row[d]);
            for (uint g = 0; g < G; ++g) p[g] += dot(k4, q_sh[g][d]);
        }
        for (uint g = 0; g < G; ++g) {
            float score = valid ? p[g] : -INFINITY;
            float m_new = max(m[g], simd_max(score));  // finite: lane 0's key is always valid
            float correction = exp(m[g] - m_new);
            p[g] = valid ? exp(score - m_new) : 0.0f;
            l[g] = l[g] * correction + simd_sum(p[g]);
            acc[g] *= correction;
            m[g] = m_new;
        }
        const uint n = min(32u, end - c);
        for (uint k = 0; k < n; ++k) {  // V rows of the chunk's keys, read once for all G queries
            const uint slot_k = simd_shuffle(slot, k);
            constant T* v_base = v_pool + (ulong(slot_k) * KV + kv_head) * D;
            float4 v4 = float4(((constant vec<T, 4>*)v_base)[lane]);
            for (uint g = 0; g < G; ++g) acc[g] += simd_shuffle(p[g], k) * v4;
        }
    }

    // Merge the simdgroups' partial softmaxes.
    threadgroup float sh_m[NSG][G], sh_l[NSG][G];
    threadgroup float4 sh_acc[NSG][G][32];
    for (uint g = 0; g < G; ++g) {
        if (lane == 0) { sh_m[sg][g] = m[g]; sh_l[sg][g] = l[g]; }
        sh_acc[sg][g][lane] = acc[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    for (uint g = 0; g < G; ++g) {
        float m_all = -INFINITY;
        for (uint s = 0; s < NSG; ++s) m_all = max(m_all, sh_m[s][g]);
        float l_all = 0.0f;
        float4 a = 0.0f;
        for (uint s = 0; s < NSG; ++s) {
            float w = sh_l[s][g] > 0.0f ? exp(sh_m[s][g] - m_all) : 0.0f;  // empty: weight 0
            l_all += sh_l[s][g] * w;
            a += sh_acc[s][g][lane] * w;
        }
        const uint idx = (seq * H + kv_head * G + g) * P + part;
        if (P == 1) {
            out[idx * (D / 4) + lane] = vec<T, 4>(a / l_all);
            continue;
        }
        if (lane == 0) { part_m[idx] = m_all; part_l[idx] = l_all; }
        part_acc[idx * (D / 4) + lane] = a;
    }
}

// One threadgroup of 32 threads per (query head, sequence); thread t owns dims 4t..4t+3.
template <typename T>
kernel void decode_attention_reduce(
    device vec<T, 4>* out        [[buffer(0)]],  // [B, H, 128 / 4]
    constant float* part_m       [[buffer(1)]],
    constant float* part_l       [[buffer(2)]],
    constant float4* part_acc    [[buffer(3)]],
    constant long& P             [[buffer(4)]],
    uint2 group  [[threadgroup_position_in_grid]],   // (head, sequence)
    uint2 groups [[threadgroups_per_grid]],
    uint t       [[thread_index_in_threadgroup]])
{
    const uint row = group.y * groups.x + group.x;  // seq * H + head
    float m_all = -INFINITY;
    for (uint p = 0; p < P; ++p)
        if (part_l[row * P + p] > 0.0f) m_all = max(m_all, part_m[row * P + p]);
    float l_all = 0.0f;
    float4 a = 0.0f;
    for (uint p = 0; p < P; ++p) {
        float l = part_l[row * P + p];
        if (l == 0.0f) continue;  // a partition past the end of this sequence
        float w = exp(part_m[row * P + p] - m_all);
        l_all += l * w;
        a += part_acc[(row * P + p) * 32 + t] * w;
    }
    out[row * 32 + t] = vec<T, 4>(a / l_all);
}

// Merge step shared by topk_rows: each lane offers the head of its own sorted
// list (values[lane][*] in threadgroup memory, `len` entries each); k rounds of
// simd_max pick the k largest overall, in order. Inside one simdgroup, so no
// barriers. Ties go to the lowest lane.
template <typename Emit>
inline void simd_merge(threadgroup float (*values)[64], uint lanes, uint len, uint K,
                       uint lane, Emit emit) {
    uint h = 0;
    for (uint n = 0; n < K; ++n) {
        const float mine = lane < lanes && h < len ? values[lane][h] : -INFINITY;
        const float best = simd_max(mine);
        if (lane == simd_min(mine == best ? lane : 32u)) {
            emit(n, mine, lane, h);
            ++h;
        }
    }
}

// Top-k of each row of a [B, V] matrix, sorted, for k <= 64 (the sampler's top-k).
// One threadgroup of 1024 threads per row:
//   1. a threshold: each thread takes the max of its strided share of the row;
//      tau = the k-th largest of those 1024 maxima. Those k maxima are k distinct
//      entries >= tau, so the row's whole top k is >= tau.
//   2. each thread rescans its share keeping a sorted list of its entries >= tau;
//      thanks to the threshold, almost every entry is skipped by one comparison.
//   3. each simdgroup merges its lanes' lists, then simdgroup 0 merges those.
kernel void topk_rows(
    device float* out_values  [[buffer(0)]],   // [B, k]
    device long* out_indices  [[buffer(1)]],   // [B, k]
    constant float* x         [[buffer(2)]],   // [B, V]
    constant long& V          [[buffer(3)]],
    constant long& k          [[buffer(4)]],
    uint row      [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]],
    uint sg       [[simdgroup_index_in_threadgroup]],
    uint nsg      [[simdgroups_per_threadgroup]],
    uint lane     [[thread_index_in_simdgroup]])
{
    constexpr uint MAX_K = 64;
    const uint K = k;
    constant float* r = x + row * V;
    threadgroup float sg_vals[32][MAX_K];
    threadgroup uint sg_idxs[32][MAX_K];
    threadgroup float tau;

    // 1. The threshold: per-simdgroup top k of the lanes' maxima (a lane's max is
    //    taken once, then replaced by -inf), then the k-th largest across simdgroups.
    float m = -INFINITY;
    for (uint i = tid; i < V; i += nthreads) m = max(m, r[i]);
    for (uint n = 0; n < K; ++n) {
        const float best = simd_max(m);
        if (lane == simd_min(m == best ? lane : 32u)) {
            sg_vals[sg][n] = m;
            m = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0)
        simd_merge(sg_vals, nsg, K, K, lane, [&](uint n, float v, uint, uint) {
            if (n == K - 1) tau = v;  // -inf if the row has fewer than k finite entries
        });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float threshold = tau;

    // 2. Sorted local lists of the entries >= threshold.
    float vals[MAX_K];
    uint idxs[MAX_K];
    uint count = 0;
    for (uint i = tid; i < V; i += nthreads) {
        const float v = r[i];
        if (v < threshold || (count == K && v <= vals[K - 1])) continue;
        uint pos = count < K ? count++ : K - 1;
        while (pos > 0 && vals[pos - 1] < v) {  // insertion into the descending list
            vals[pos] = vals[pos - 1];
            idxs[pos] = idxs[pos - 1];
            --pos;
        }
        vals[pos] = v;
        idxs[pos] = i;
    }

    // 3. Merge: lanes -> simdgroups (via each lane's own registers) -> simdgroup 0.
    uint head = 0;
    for (uint n = 0; n < K; ++n) {
        const float mine = head < count ? vals[head] : -INFINITY;
        const float best = simd_max(mine);
        if (lane == simd_min(mine == best ? lane : 32u)) {
            sg_vals[sg][n] = mine;
            sg_idxs[sg][n] = head < count ? idxs[head] : 0;
            head += head < count;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    simd_merge(sg_vals, nsg, K, K, lane, [&](uint n, float v, uint s, uint h) {
        out_values[row * K + n] = v;
        out_indices[row * K + n] = h < K ? sg_idxs[s][h] : 0;
    });
}

// SwiGLU's gating, fused: out = silu(gate) * up, reading the stacked [T, 2I]
// output of the gate_up projection directly (two launches and a temporary -> one).
template <typename T>
kernel void silu_mul(
    device T* out         [[buffer(0)]],   // [T, I]
    constant T* gate_up   [[buffer(1)]],   // [T, 2I]: gate in the first half, up in the second
    constant long& inner  [[buffer(2)]],   // I
    uint2 pos [[thread_position_in_grid]])  // (column, token)
{
    const ulong i = pos.x, t = pos.y, I = inner;
    const float g = float(gate_up[t * 2 * I + i]), u = float(gate_up[t * 2 * I + I + i]);
    out[t * I + i] = T(g / (1.0f + exp(-g)) * u);
}

#define INSTANTIATE_PARTIAL(T, NAME, G)                                                   \
template [[host_name("decode_attention_partial_" NAME "_g" #G)]] kernel void              \
decode_attention_partial<T, G>(device float*, device float*, device float4*,              \
    device vec<T, 4>*, constant T*, constant T*, constant T*, constant long*,             \
    constant long*, constant long&, constant long&, constant long&, constant long&,       \
    constant float&, uint3, uint3, uint, uint, uint);
#define INSTANTIATE(T, NAME)                                                              \
template [[host_name("qk_norm_rope_store_" NAME)]] kernel void qk_norm_rope_store<T>(      \
    device T*, device T*, device T*, constant T*, constant T*, constant T*,               \
    constant long*, constant long*, constant float*, constant long&, constant long&,      \
    constant float&, uint2, uint2, uint, uint);                                           \
template [[host_name("silu_mul_" NAME)]] kernel void silu_mul<T>(                          \
    device T*, constant T*, constant long&, uint2);                                       \
template [[host_name("decode_attention_reduce_" NAME)]] kernel void                       \
decode_attention_reduce<T>(device vec<T, 4>*, constant float*, constant float*,           \
    constant float4*, constant long&, uint2, uint2, uint);                                \
INSTANTIATE_PARTIAL(T, NAME, 1) INSTANTIATE_PARTIAL(T, NAME, 2)                           \
INSTANTIATE_PARTIAL(T, NAME, 4) INSTANTIATE_PARTIAL(T, NAME, 5)                           \
INSTANTIATE_PARTIAL(T, NAME, 8)
INSTANTIATE(bfloat, "bfloat")
INSTANTIATE(float, "float")
"""

_library = None
_SUFFIX = {torch.bfloat16: "bfloat", torch.float32: "float"}
_PARTITION = 512  # keys per threadgroup in decode attention
_GQA_GROUPS = (1, 2, 4, 5, 8)  # query heads per kv head that the kernel is compiled for


def available(device: torch.device, dtype: torch.dtype, config: ModelConfig) -> bool:
    """Whether the kernels can run: Apple GPU, bf16/fp32, 128-wide heads, a compiled GQA ratio."""
    return (
        device.type == "mps"
        and dtype in _SUFFIX
        and config.head_dim == HEAD_DIM
        and config.num_heads // config.num_kv_heads in _GQA_GROUPS
    )


def _lib():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(_SOURCE)
    return _library


def qk_norm_rope_store(
    qkv: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    inv_freq: torch.Tensor,
    k_slots: torch.Tensor,
    v_slots: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    eps: float,
) -> torch.Tensor:
    """Normalise and rotate q and k, and store k and v in the KV pool, in one launch.

    qkv:          [T, H + 2*KV, 128]  output of the stacked qkv projection (contiguous)
    positions:    [T]  slot_mapping: [T]  inv_freq: [64] fp32 (see model.rope_inv_freq)
    k_slots, v_slots: [num_slots, KV, 128]  one layer's pool, written in place
    returns       q, [T, H, 128]

    Same math as model.rms_norm + model.apply_rope + KVPool.write.
    """
    T = qkv.shape[0]
    q = torch.empty(T, num_heads, HEAD_DIM, device=qkv.device, dtype=qkv.dtype)
    kernel = getattr(_lib(), "qk_norm_rope_store_" + _SUFFIX[qkv.dtype])
    kernel(
        q, k_slots, v_slots, qkv, q_norm, k_norm, positions, slot_mapping, inv_freq,
        num_heads, num_kv_heads, eps,
        threads=(64 * (num_heads + 2 * num_kv_heads), T),
        group_size=(64, 1),
    )  # fmt: skip
    return q


def paged_decode_attention(
    q: torch.Tensor,
    k_slots: torch.Tensor,
    v_slots: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    max_context_len: int,
) -> torch.Tensor:
    """Attention of one new query per sequence over its whole context, read from the pool.

    q:            [B, H, 128]       each sequence's newest query
    k_slots, v_slots: [num_slots, KV, 128]  one layer's pool
    block_tables: [B, max_blocks]   physical blocks of each sequence (padding unused)
    context_lens: [B]               keys each sequence attends to
    max_context_len: host int, the longest of them (sets the number of partitions)
    returns       [B, H, 128]

    Same math as model.decode_attention over keys gathered by the block tables.
    """
    B, H, _ = q.shape
    group = H // num_kv_heads
    partitions = -(-max_context_len // _PARTITION)
    lib, suffix = _lib(), _SUFFIX[q.dtype]
    part_m = torch.empty(B, H, partitions, device=q.device, dtype=torch.float32)
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty(B, H, partitions, HEAD_DIM, device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    getattr(lib, f"decode_attention_partial_{suffix}_g{group}")(
        part_m, part_l, part_acc, out, q.contiguous(), k_slots, v_slots, block_tables,
        context_lens, block_tables.shape[1], block_size, num_kv_heads, _PARTITION,
        HEAD_DIM**-0.5,
        threads=(128 * num_kv_heads, B, partitions),
        group_size=(128, 1, 1),
    )  # fmt: skip
    if partitions > 1:
        getattr(lib, f"decode_attention_reduce_{suffix}")(
            out, part_m, part_l, part_acc, partitions, threads=(32 * H, B), group_size=(32, 1)
        )
    return out


TOPK_MAX = 64


def topk_rows(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The k largest entries of each row, sorted descending, for k <= TOPK_MAX.

    x: [B, V] fp32  ->  (values [B, k] fp32, indices [B, k] int64)

    Same result as torch.topk(x, k) (ties may be ordered differently).
    """
    B, V = x.shape
    values = torch.empty(B, k, device=x.device, dtype=torch.float32)
    indices = torch.empty(B, k, device=x.device, dtype=torch.long)
    _lib().topk_rows(values, indices, x.contiguous(), V, k, threads=1024 * B, group_size=1024)
    return values, indices


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up for the stacked gate_up projection output.

    gate_up: [T, 2I]  ->  returns [T, I]
    """
    T, two_i = gate_up.shape
    out = torch.empty(T, two_i // 2, device=gate_up.device, dtype=gate_up.dtype)
    getattr(_lib(), "silu_mul_" + _SUFFIX[gate_up.dtype])(
        out, gate_up.contiguous(), two_i // 2, threads=(two_i // 2, T), group_size=(256, 1)
    )
    return out
