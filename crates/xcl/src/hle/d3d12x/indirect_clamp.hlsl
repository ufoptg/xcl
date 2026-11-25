// Bounds a title's indirect draw and dispatch records before ExecuteIndirect reads them
// (`indirect_clamp.rs`). One group walks every record in chunks of THREADS: each record is
// copied whole, its draw or dispatch arguments are clamped to per-record limits, and a running
// total of the work the records ask for (a prefix sum across the chunk) empties every record
// past BUDGET.

#define THREADS 1024
// per record: vertices (or indices) per instance, vertices times instances, groups per
// dimension, groups per dispatch; a group counts as 64 threads, a mesh group as 128
#define MAX_VERTICES (1u << 24)
#define MAX_VERTEX_INSTANCES (1u << 26)
#define MAX_GROUPS_DIM 65535u
#define MAX_GROUPS (1u << 22)
#define BUDGET (1u << 29)

RWByteAddressBuffer source : register(u0);  // the UMA window holding the title's records
RWByteAddressBuffer bounded : register(u1); // our copy, which ExecuteIndirect reads
RWByteAddressBuffer counts : register(u2);  // the UMA window holding the count (if any)

cbuffer Config : register(b0) {
    uint records;   // the call's maximum command count
    uint source_at; // byte offset of record 0 in `source`
    uint stride;    // bytes per record (a multiple of 4)
    uint action_at; // byte offset of the draw or dispatch arguments in a record
    uint kind;      // 0 draw, 1 indexed draw, 2 dispatch, 3 mesh dispatch
    uint count_at;  // byte offset of the count in `counts`; 0xFFFFFFFF: no count buffer
};

groupshared uint scan[THREADS];
groupshared uint carry;

uint sat_add(uint a, uint b) {
    uint s = a + b;
    return s < a ? 0xFFFFFFFFu : s;
}

// Clamps one record's arguments in place and returns the work they ask for.
uint clamp_args(inout uint3 a) {
    if (kind <= 1) {
        a.x = min(a.x, MAX_VERTICES);
        a.y = min(a.y, MAX_VERTEX_INSTANCES / max(a.x, 1u));
        return a.x * a.y;
    }
    a = min(a, uint3(MAX_GROUPS_DIM, MAX_GROUPS_DIM, MAX_GROUPS_DIM));
    a.y = min(a.y, MAX_GROUPS / max(a.x, 1u));
    a.z = min(a.z, MAX_GROUPS / max(a.x * a.y, 1u));
    return a.x * a.y * a.z * (kind == 3 ? 128u : 64u);
}

[numthreads(THREADS, 1, 1)]
void main(uint t : SV_GroupIndex) {
    if (t == 0)
        carry = 0;
    uint live = records;
    if (count_at != 0xFFFFFFFFu)
        live = min(live, counts.Load(count_at));
    GroupMemoryBarrierWithGroupSync();
    for (uint first = 0; first < records; first += THREADS) {
        uint i = first + t;
        uint3 a = uint3(0, 0, 0);
        uint work = 0;
        if (i < records) {
            uint src = source_at + i * stride;
            uint dst = i * stride;
            for (uint k = 0; k < stride; k += 4)
                bounded.Store(dst + k, source.Load(src + k));
            a = source.Load3(src + action_at);
            work = clamp_args(a);
        }
        // records past the count are never executed: they take none of the budget
        scan[t] = i < live ? work : 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint d = 1; d < THREADS; d <<= 1) {
            uint before = t >= d ? scan[t - d] : 0;
            GroupMemoryBarrierWithGroupSync();
            scan[t] = sat_add(scan[t], before);
            GroupMemoryBarrierWithGroupSync();
        }
        uint total = sat_add(carry, scan[t]);
        if (i < records) {
            if (total > BUDGET) {
                if (kind <= 1)
                    a.y = 0;
                else
                    a = uint3(0, 0, 0);
            }
            bounded.Store3(i * stride + action_at, a);
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == THREADS - 1)
            carry = total;
        GroupMemoryBarrierWithGroupSync();
    }
}
