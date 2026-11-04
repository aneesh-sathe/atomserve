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
#                            (these two also handle an int8 pool: quantize on
#                            write, dequantize on read, see kv_cache.py)
#   topk_rows                the sampler's top-k over the 151,936-entry vocabulary,
#                            several times faster than torch.topk on MPS
#   silu_mul                 SwiGLU's silu(gate) * up in one pass
#   gather_dequantize_kv     a sequence's K or V out of an int8 pool, as bf16, for prefill
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

// x <- H x for one 128-value head held by 64 threads, thread i owning elements
// a = x[i] and b = x[i + 64]; H is the orthonormal Hadamard matrix in Sylvester
// order (kv_cache.hadamard). A fast Walsh-Hadamard transform: one butterfly per
// bit of the element index, pairing e with e ^ bit as (x[e] + x[e^bit], x[e] - x[e^bit]).
// Bit 64 pairs a with b in registers; bit 32 pairs the two simdgroups, through
// threadgroup memory; bits 16..1 pair lanes, by shuffles. Every thread of the
// threadgroup must call it (it has a barrier).
inline void hadamard_128(thread float& a, thread float& b, uint i, threadgroup float* buf)
{
    const float sum = a + b, diff = a - b;
    a = sum;
    b = diff;
    buf[i] = a;
    buf[i + 64] = b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool high = i & 32;
    const float pa = buf[i ^ 32], pb = buf[(i ^ 32) + 64];
    a = high ? pa - a : a + pa;
    b = high ? pb - b : b + pb;
    for (ushort s = 16; s > 0; s >>= 1) {
        const bool hi = i & s;
        const float qa = simd_shuffle_xor(a, s), qb = simd_shuffle_xor(b, s);
        a = hi ? qa - a : a + qa;
        b = hi ? qb - b : b + qb;
    }
    a *= M_SQRT1_2_F / 8.0f;  // 1 / sqrt(128)
    b *= M_SQRT1_2_F / 8.0f;
}

// One threadgroup per (token, head slot), head slots ordered q heads, k heads,
// v heads as they come out of the stacked qkv projection. 64 threads per group;
// thread i owns the RoPE pair (i, i + 64) of the head.
//
// C is the pool's storage type: T itself, or char for an int8 pool, where each
// k or v head is stored as round(x * 127 / max|x|) plus the scale max|x| / 127
// (the same integers as kv_cache.quantize_heads), and q and k are first rotated
// by the 128 x 128 Hadamard matrix (see kv_cache.py for why).
template <typename T, typename C>
kernel void qk_norm_rope_store(
    device T* q_out             [[buffer(0)]],   // [tokens, H, 128]
    device C* k_pool            [[buffer(1)]],   // [slots, KV, 128], one layer
    device C* v_pool            [[buffer(2)]],   // [slots, KV, 128], one layer
    device float* k_scale       [[buffer(3)]],   // [slots, KV], used only for an int8 pool
    device float* v_scale       [[buffer(4)]],   // [slots, KV], used only for an int8 pool
    constant T* qkv             [[buffer(5)]],   // [tokens, H + 2 KV, 128]
    constant T* q_norm          [[buffer(6)]],   // [128]
    constant T* k_norm          [[buffer(7)]],   // [128]
    constant long* positions    [[buffer(8)]],   // [tokens]
    constant long* slot_mapping [[buffer(9)]],   // [tokens]
    constant float* inv_freq    [[buffer(10)]],  // [64]
    constant long& num_heads    [[buffer(11)]],
    constant long& num_kv_heads [[buffer(12)]],
    constant float& eps         [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],   // (head slot, token)
    uint2 tid   [[thread_position_in_threadgroup]],
    uint sg     [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    constexpr bool QUANT = sizeof(C) == 1;
    const uint D = 128, half_d = 64, i = tid.x;
    const uint H = num_heads, KV = num_kv_heads, head = group.x, t = group.y;
    constant T* x = qkv + (t * (H + 2 * KV) + head) * D;
    float a = float(x[i]), b = float(x[i + half_d]);
    // Every branch below depends only on `head`, so it is uniform across the
    // threadgroup and no thread skips a barrier the others wait at.
    threadgroup float partial_sq[2], partial_max[2], rotation_buf[128];

    if (head < H + KV) {  // q and k heads: RMSNorm, then RoPE (a v head is stored as is)
        // RMSNorm over the head's 128 values: reduce within each simdgroup, then across the two.
        float sum_sq = simd_sum(a * a + b * b);
        if (lane == 0) partial_sq[sg] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = rsqrt((partial_sq[0] + partial_sq[1]) / float(D) + eps);
        constant T* w = head < H ? q_norm : k_norm;
        a *= scale * float(w[i]);
        b *= scale * float(w[i + half_d]);

        // RoPE: rotate the pair by position * frequency_i (precise trig: angles reach ~4e4 rad).
        float angle = float(positions[t]) * inv_freq[i];
        float c = precise::cos(angle), s = precise::sin(angle);
        const float ra = a * c - b * s, rb = b * c + a * s;
        a = ra;
        b = rb;
        if (QUANT) hadamard_128(a, b, i, rotation_buf);
        if (head < H) {
            device T* dst = q_out + (t * H + head) * D;
            dst[i] = T(a);
            dst[i + half_d] = T(b);
            return;
        }
    }

    const bool is_k = head < H + KV;
    const ulong row = ulong(slot_mapping[t]) * KV + (is_k ? head - H : head - H - KV);
    device C* dst = (is_k ? k_pool : v_pool) + row * D;
    if (QUANT) {
        float amax = simd_max(max(abs(a), abs(b)));
        if (lane == 0) partial_max[sg] = amax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        amax = max(partial_max[0], partial_max[1]);
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        dst[i] = C(clamp(rint(a * inv), -127.0f, 127.0f));
        dst[i + half_d] = C(clamp(rint(b * inv), -127.0f, 127.0f));
        if (i == 0) (is_k ? k_scale : v_scale)[row] = amax / 127.0f;
    } else {
        dst[i] = C(a);
        dst[i + half_d] = C(b);
    }
}

// Decode attention in two passes (the PagedAttention "v2" scheme):
//
// 1. partial: one threadgroup per (kv head, sequence, partition of 512 keys).
//    It handles all G query heads that share the kv head (GQA), so every K/V
//    row is read once. Each simdgroup takes 32 keys at a time, one per lane:
//    lane l scores key l against the G queries, then the simdgroup updates an
//    online softmax and accumulates V two keys at a time: lanes 0-15 take one
//    key and lanes 16-31 the next, each lane owning 8 dims (one 8- or 16-byte
//    load per key; one key per pass, at 4 bytes a lane, was load-latency bound).
//    The two halves' sums are added once, at the end. Results are
//    unnormalised (running max m, sum l, weighted sum of V).
// 2. reduce: merges the partitions of each (sequence, head) into the output.
//
// Splitting long contexts into partitions keeps the GPU busy even for a
// single sequence. With a single partition (short contexts) there is nothing
// to merge: pass 1 writes the normalised output itself and pass 2 is skipped.
//
// With an int8 pool (C = char), a key's scale multiplies its score (q . (s k_q)
// = s (q . k_q)) and a value's scale its softmax weight, so dequantizing costs
// two multiplies per key instead of one per element.
template <typename T, typename C, uint G>
kernel void decode_attention_partial(
    device float* part_m        [[buffer(0)]],   // [B, H, P]
    device float* part_l        [[buffer(1)]],   // [B, H, P]
    device float4* part_acc     [[buffer(2)]],   // [B, H, P, 128 / 4]
    device vec<T, 4>* out       [[buffer(3)]],   // [B, H, 128 / 4], written only if P == 1
    constant T* q               [[buffer(4)]],   // [B, H, 128]
    constant C* k_pool          [[buffer(5)]],   // [slots, KV, 128], one layer
    constant C* v_pool          [[buffer(6)]],   // [slots, KV, 128], one layer
    constant float* k_scale     [[buffer(7)]],   // [slots, KV], used only for an int8 pool
    constant float* v_scale     [[buffer(8)]],   // [slots, KV], used only for an int8 pool
    constant long* block_tables [[buffer(9)]],   // [B, max_blocks]
    constant long* context_lens [[buffer(10)]],  // [B]
    constant long& max_blocks   [[buffer(11)]],
    constant long& block_size   [[buffer(12)]],
    constant long& num_kv_heads [[buffer(13)]],
    constant long& partition    [[buffer(14)]],
    constant float& scale       [[buffer(15)]],
    uint3 group  [[threadgroup_position_in_grid]],   // (kv head, sequence, partition)
    uint3 groups [[threadgroups_per_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint sg      [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]])
{
    constexpr uint D = 128, NSG = 4;
    constexpr bool QUANT = sizeof(C) == 1;
    const uint kv_head = group.x, seq = group.y, part = group.z, P = groups.z;
    const uint KV = num_kv_heads, H = KV * G;
    const uint len = context_lens[seq];
    const uint start = part * partition, end = min(start + uint(partition), len);
    constant long* table = block_tables + seq * max_blocks;

    threadgroup float4 q_sh[G][D / 4];  // the G queries, pre-scaled
    for (uint i = tid; i < G * D; i += NSG * 32)
        ((threadgroup float*)q_sh)[i] = float(q[(seq * H + kv_head * G) * D + i]) * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // V is accumulated KPP keys per pass: 32 / KPP lanes per key, each owning 4 * KPP dims
    // (4 keys per pass was 1.8x slower: register pressure).
    constexpr uint KPP = 2, LPK = 32 / KPP, F4 = KPP;  // F4: float4s per lane
    const uint sub = lane / LPK, dl = lane % LPK;  // which key of the pass, which dims
    float m[G], l[G];
    float4 acc[G][F4];
    for (uint g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        for (uint f = 0; f < F4; ++f) acc[g][f] = 0.0f;
    }

    for (uint c = start + sg * 32; c < end; c += NSG * 32) {
        const uint j = c + lane;
        const bool valid = j < end;
        const uint slot = valid ? uint(table[j / block_size] * block_size + j % block_size) : 0;

        const ulong row = ulong(slot) * KV + kv_head;
        float p[G];  // this lane's key: score against each query, then its softmax weight
        for (uint g = 0; g < G; ++g) p[g] = 0.0f;
        if (QUANT) {  // 16-byte loads (int8 keys in 4-byte loads were load-bound)
            constant uint4* k_row = (constant uint4*)(k_pool + row * D);
            for (uint d = 0; d < D / 16; ++d) {
                const uint4 w = k_row[d];
                for (uint e = 0; e < 4; ++e) {
                    const float4 k4 = float4(as_type<char4>(w[e]));
                    for (uint g = 0; g < G; ++g) p[g] += dot(k4, q_sh[g][4 * d + e]);
                }
            }
        } else {
            constant vec<C, 4>* k_row = (constant vec<C, 4>*)(k_pool + row * D);
            for (uint d = 0; d < D / 4; ++d) {
                float4 k4 = float4(k_row[d]);
                for (uint g = 0; g < G; ++g) p[g] += dot(k4, q_sh[g][d]);
            }
        }
        const float k_s = QUANT ? k_scale[row] : 1.0f, v_s = QUANT ? v_scale[row] : 1.0f;
        float pv[G];  // the softmax weight times the value's scale
        for (uint g = 0; g < G; ++g) {
            float score = valid ? p[g] * k_s : -INFINITY;
            float m_new = max(m[g], simd_max(score));  // finite: lane 0's key is always valid
            float correction = exp(m[g] - m_new);
            p[g] = valid ? exp(score - m_new) : 0.0f;
            pv[g] = p[g] * v_s;
            l[g] = l[g] * correction + simd_sum(p[g]);
            for (uint f = 0; f < F4; ++f) acc[g][f] *= correction;
            m[g] = m_new;
        }
        // V rows of the chunk's keys, read once for all G queries. k0 < n <= 32 and
        // KPP divides 32, so k <= 31; a k >= n is an invalid lane: weight 0, slot 0.
        const uint n = min(32u, end - c);
        for (uint k0 = 0; k0 < n; k0 += KPP) {
            const uint k = k0 + sub;
            const uint slot_k = simd_shuffle(slot, k);
            constant C* v_base = v_pool + (ulong(slot_k) * KV + kv_head) * D + 4 * F4 * dl;
            float4 v4[F4];
            for (uint f = 0; f < F4; ++f) {
                if (QUANT) v4[f] = float4(as_type<char4>(((constant uint*)v_base)[f]));
                else v4[f] = float4(((constant vec<C, 4>*)v_base)[f]);
            }
            for (uint g = 0; g < G; ++g) {
                const float w = simd_shuffle(pv[g], k);
                for (uint f = 0; f < F4; ++f) acc[g][f] += w * v4[f];
            }
        }
    }

    // Merge the simdgroups' partial softmaxes.
    threadgroup float sh_m[NSG][G], sh_l[NSG][G];
    threadgroup float4 sh_acc[NSG][G][32];
    for (uint g = 0; g < G; ++g) {
        if (lane == 0) { sh_m[sg][g] = m[g]; sh_l[sg][g] = l[g]; }
        // Add up the KPP keys' partial sums, then regroup to lane l owning dims 4l..4l+3.
        float4 mine = 0.0f;
        for (uint f = 0; f < F4; ++f) {
            float4 a = acc[g][f];
            for (uint s = LPK; s < 32; s *= 2) a += simd_shuffle_xor(a, s);
            const float4 from = simd_shuffle(a, lane / F4);  // lane l: dims 4l..4l+3
            if (lane % F4 == f) mine = from;
        }
        sh_acc[sg][g][lane] = mine;
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

// int8 weight-only matmul for decode-sized inputs: y = x @ (W_q * scale)^T [+ residual].
// x has M <= 8 rows (M is a template parameter, so every loop below unrolls and
// the accumulators stay in registers); W_q is [N, K] int8 with one float scale per
// output row. Decode reads every weight once per step, so halving the bytes (int8
// vs bf16) halves the dominant cost. Each simdgroup computes ROWS output rows; each
// lane streams 8 weights at a time along K, dequantising in registers, and reuses
// every activation it loads for all ROWS rows. K must be a multiple of 256.
// (int8_rows_per_group in Python mirrors ROWS: 8 simdgroups per threadgroup.)
template <typename T, uint M>
kernel void int8_linear(
    device T* y                 [[buffer(0)]],   // [M, N]
    constant T* x               [[buffer(1)]],   // [M, K]
    constant char* w            [[buffer(2)]],   // [N, K] int8
    constant float* scale       [[buffer(3)]],   // [N]
    constant T* residual        [[buffer(4)]],   // [M, N], read only if has_residual
    constant long& N_           [[buffer(5)]],
    constant long& K_           [[buffer(6)]],
    constant long& has_residual [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    constexpr uint ROWS = M <= 4 ? 4 : 2;  // fewer rows for bigger M: keeps acc in registers
    const uint N = N_, K = K_;
    const uint n0 = (group * 8 + sg) * ROWS;  // 8 simdgroups per threadgroup
    if (n0 >= N) return;
    const uint rows = min(ROWS, N - n0);

    float acc[ROWS][M] = {};
    for (uint k = lane * 8; k < K; k += 32 * 8) {
        float4 xa[M], xb[M];
        for (uint m = 0; m < M; ++m) {
            constant vec<T, 4>* xp = (constant vec<T, 4>*)(x + ulong(m) * K + k);
            xa[m] = float4(xp[0]);
            xb[m] = float4(xp[1]);
        }
        for (uint r = 0; r < ROWS; ++r) {
            if (r >= rows) break;
            constant char4* wp = (constant char4*)(w + ulong(n0 + r) * K + k);
            const float4 w0 = float4(wp[0]), w1 = float4(wp[1]);
            for (uint m = 0; m < M; ++m) acc[r][m] += dot(w0, xa[m]) + dot(w1, xb[m]);
        }
    }
    for (uint r = 0; r < rows; ++r) {
        const uint n = n0 + r;
        for (uint m = 0; m < M; ++m) {
            float v = simd_sum(acc[r][m]) * scale[n];
            if (lane == 0) {
                if (has_residual) v += float(residual[ulong(m) * N + n]);
                y[ulong(m) * N + n] = T(v);
            }
        }
    }
}

// int8 -> T in one pass: out[n, k] = q[n, k] * scale[n]. For big inputs (prefill),
// where the matmul itself is compute-bound and runs best as a regular matmul.
template <typename T>
kernel void dequantize_int8(
    device vec<T, 4>* out    [[buffer(0)]],   // [N, K / 4]
    constant char4* q        [[buffer(1)]],   // [N, K / 4]
    constant float* scale    [[buffer(2)]],   // [N]
    constant long& K4        [[buffer(3)]],   // K / 4
    uint2 pos [[thread_position_in_grid]])    // (column / 4, row)
{
    const ulong i = ulong(pos.y) * K4 + pos.x;
    out[i] = vec<T, 4>(float4(q[i]) * scale[pos.y]);
}

// Gather positions of one sequence out of an int8 pool layer and dequantize them:
// out[s, h, :] = pool[slots[s], h, :] * scale[slots[s], h]. Prefill attention's
// keys and values, in one pass instead of two gathers plus a dequantize (which
// made an 8k-token prefill 16% slower than with a bf16 pool).
template <typename T>
kernel void gather_dequantize_kv(
    device vec<T, 4>* out    [[buffer(0)]],   // [S, KV, 128 / 4]
    constant char4* pool     [[buffer(1)]],   // [slots, KV, 128 / 4], one layer
    constant float* scale    [[buffer(2)]],   // [slots, KV]
    constant long* slots     [[buffer(3)]],   // [S]
    constant long& KV        [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]])    // (kv head * 32 + column / 4, position)
{
    const ulong row = ulong(slots[pos.y]) * KV + pos.x / 32;  // (slot, kv head) of this thread
    const float4 x = float4(pool[row * 32 + pos.x % 32]) * scale[row];
    out[ulong(pos.y) * KV * 32 + pos.x] = vec<T, 4>(x);
}

#define INSTANTIATE_INT8(T, NAME, M)                                                      \
template [[host_name("int8_linear_" NAME "_m" #M)]] kernel void int8_linear<T, M>(         \
    device T*, constant T*, constant char*, constant float*, constant T*, constant long&, \
    constant long&, constant long&, uint, uint, uint);
#define INSTANTIATE_PARTIAL(T, C, NAME, G)                                                \
template [[host_name("decode_attention_partial_" NAME "_g" #G)]] kernel void              \
decode_attention_partial<T, C, G>(device float*, device float*, device float4*,           \
    device vec<T, 4>*, constant T*, constant C*, constant C*, constant float*,            \
    constant float*, constant long*, constant long*, constant long&, constant long&,      \
    constant long&, constant long&, constant float&, uint3, uint3, uint, uint, uint);
#define INSTANTIATE_KV(T, C, NAME)                                                        \
template [[host_name("qk_norm_rope_store_" NAME)]] kernel void qk_norm_rope_store<T, C>(   \
    device T*, device C*, device C*, device float*, device float*, constant T*,           \
    constant T*, constant T*, constant long*, constant long*, constant float*,            \
    constant long&, constant long&, constant float&, uint2, uint2, uint, uint);           \
INSTANTIATE_PARTIAL(T, C, NAME, 1) INSTANTIATE_PARTIAL(T, C, NAME, 2)                     \
INSTANTIATE_PARTIAL(T, C, NAME, 4) INSTANTIATE_PARTIAL(T, C, NAME, 5)                     \
INSTANTIATE_PARTIAL(T, C, NAME, 8)
#define INSTANTIATE(T, NAME)                                                              \
INSTANTIATE_KV(T, T, NAME) INSTANTIATE_KV(T, char, NAME "_int8")                          \
template [[host_name("dequantize_int8_" NAME)]] kernel void dequantize_int8<T>(            \
    device vec<T, 4>*, constant char4*, constant float*, constant long&, uint2);          \
template [[host_name("gather_dequantize_kv_" NAME)]] kernel void gather_dequantize_kv<T>(  \
    device vec<T, 4>*, constant char4*, constant float*, constant long*, constant long&,  \
    uint2);                                                                               \
INSTANTIATE_INT8(T, NAME, 1) INSTANTIATE_INT8(T, NAME, 2) INSTANTIATE_INT8(T, NAME, 3)     \
INSTANTIATE_INT8(T, NAME, 4) INSTANTIATE_INT8(T, NAME, 5) INSTANTIATE_INT8(T, NAME, 6)     \
INSTANTIATE_INT8(T, NAME, 7) INSTANTIATE_INT8(T, NAME, 8)                                  \
template [[host_name("silu_mul_" NAME)]] kernel void silu_mul<T>(                          \
    device T*, constant T*, constant long&, uint2);                                       \
template [[host_name("decode_attention_reduce_" NAME)]] kernel void                       \
decode_attention_reduce<T>(device vec<T, 4>*, constant float*, constant float*,           \
    constant float4*, constant long&, uint2, uint2, uint);
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
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalise and rotate q and k, and store k and v in the KV pool, in one launch.

    qkv:          [T, H + 2*KV, 128]  output of the stacked qkv projection (contiguous)
    positions:    [T]  slot_mapping: [T]  inv_freq: [64] fp32 (see model.rope_inv_freq)
    k_slots, v_slots: [num_slots, KV, 128]  one layer's pool, written in place
    k_scale, v_scale: [num_slots, KV] fp32  the pool's scales, if it is int8
    returns       q, [T, H, 128]

    Same math as model.rms_norm + model.apply_rope + KVPool.write.
    """
    T = qkv.shape[0]
    q = torch.empty(T, num_heads, HEAD_DIM, device=qkv.device, dtype=qkv.dtype)
    kernel = getattr(_lib(), "qk_norm_rope_store_" + _kv_suffix(qkv.dtype, k_slots))
    kernel(
        q, k_slots, v_slots, _or(k_scale, k_slots), _or(v_scale, v_slots), qkv, q_norm, k_norm,
        positions, slot_mapping, inv_freq, num_heads, num_kv_heads, eps,
        threads=(64 * (num_heads + 2 * num_kv_heads), T),
        group_size=(64, 1),
    )  # fmt: skip
    return q


def _kv_suffix(dtype: torch.dtype, pool: torch.Tensor) -> str:
    """Kernel name suffix for activations in `dtype` and a pool stored as `pool.dtype`."""
    return _SUFFIX[dtype] + ("_int8" if pool.dtype == torch.int8 else "")


def _or(scale: torch.Tensor | None, placeholder: torch.Tensor) -> torch.Tensor:
    """The scale buffer, or any tensor to bind in its place (a plain pool never reads it)."""
    return placeholder if scale is None else scale


def paged_decode_attention(
    q: torch.Tensor,
    k_slots: torch.Tensor,
    v_slots: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    max_context_len: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention of one new query per sequence over its whole context, read from the pool.

    q:            [B, H, 128]       each sequence's newest query
    k_slots, v_slots: [num_slots, KV, 128]  one layer's pool
    k_scale, v_scale: [num_slots, KV] fp32  the pool's scales, if it is int8
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
    getattr(lib, f"decode_attention_partial_{_kv_suffix(q.dtype, k_slots)}_g{group}")(
        part_m, part_l, part_acc, out, q.contiguous(), k_slots, v_slots,
        _or(k_scale, k_slots), _or(v_scale, v_slots), block_tables, context_lens,
        block_tables.shape[1], block_size, num_kv_heads, _PARTITION, HEAD_DIM**-0.5,
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


INT8_MAX_ROWS = 8


def int8_linear(
    x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, residual: torch.Tensor | None = None
) -> torch.Tensor:
    """x @ (w * scale[:, None])^T, plus `residual` if given, for at most INT8_MAX_ROWS rows.

    x: [M, K]  w: [N, K] int8  scale: [N] fp32  residual: [M, N]  ->  [M, N] in x's dtype
    """
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    if M == 0:  # e.g. a step whose only row is a prompt chunk with nothing to sample
        return y
    rows_per_group = 8 * (4 if M <= 4 else 2)  # 8 simdgroups x ROWS (see the kernel)
    getattr(_lib(), f"int8_linear_{_SUFFIX[x.dtype]}_m{M}")(
        y, x.contiguous(), w, scale, residual if residual is not None else y,
        N, K, int(residual is not None),
        threads=256 * -(-N // rows_per_group),
        group_size=256,
    )  # fmt: skip
    return y


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """q * scale[:, None] as a regular [N, K] matrix in `dtype` (K a multiple of 4)."""
    N, K = q.shape
    out = torch.empty(N, K, device=q.device, dtype=dtype)
    getattr(_lib(), "dequantize_int8_" + _SUFFIX[dtype])(
        out, q, scale, K // 4, threads=(K // 4, N), group_size=(256, 1)
    )
    return out


def gather_dequantize_kv(
    values: torch.Tensor, scale: torch.Tensor, slots: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Gather and dequantize positions of an int8 pool layer (head_dim 128).

    values: [num_slots, KV, 128] int8  scale: [num_slots, KV] fp32  slots: [S]
    returns [S, KV, 128] in `dtype`

    Same as values[slots].float() * scale[slots][..., None], cast to `dtype`.
    """
    S, KV = slots.shape[0], values.shape[1]
    out = torch.empty(S, KV, HEAD_DIM, device=values.device, dtype=dtype)
    if S:
        getattr(_lib(), "gather_dequantize_kv_" + _SUFFIX[dtype])(
            out, values, scale, slots, KV, threads=(KV * 32, S), group_size=(min(KV * 32, 256), 1)
        )
    return out
