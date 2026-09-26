// GatedResidual decode mix (gr_mix) kernels at Qwen3.8-Flash-Next shapes: H 4, D 2560, LR 320.
// Times the two phases separately over NS rotated weight sets (cold caches).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s line %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr int H = 4, D = 2560, LR = 320, NS = 48;
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// ---- A3: one wave per (fn row, stream); block = 2 rows x 4 streams; extra blocks: sums of squares
template <int NCH>
__global__ __launch_bounds__(256) void dotsA3(const float* __restrict__ streams, const half* __restrict__ fn, float* __restrict__ dots, int M, int R)
{
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = w & 3;
    const int nb = (M + 1) / 2;
    if ((int) blockIdx.x >= nb)
    {
        for (int r = w >> 2; r < R; r += 2)
        {
            const float4* s4 = (const float4*) (streams + ((size_t) r * H + h) * D);
            float a = 0.0f;
            for (int c = lane; c < D / 4; c += 32) { float4 s = s4[c]; a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a)))); }
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o);
            if (lane == 0) dots[((size_t) r * (M + 1) + M) * H + h] = a;
        }
        return;
    }
    const int j = blockIdx.x * 2 + (w >> 2);
    if (j >= M) return;
    const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
    int4 f[NCH];
    #pragma unroll
    for (int k = 0; k < NCH; ++k) f[k] = f8[lane + 32 * k];
    for (int r = 0; r < R; ++r)
    {
        const float4* s4 = (const float4*) (streams + ((size_t) r * H + h) * D);
        float a = 0.0f;
        #pragma unroll
        for (int k = 0; k < NCH; ++k)
        {
            const int c = lane + 32 * k;
            const float4 s0 = s4[2 * c], s1 = s4[2 * c + 1];
            const half2* w2 = (const half2*) &f[k];
            float2 w0 = __half22float2(w2[0]), w1 = __half22float2(w2[1]), w2_ = __half22float2(w2[2]), w3 = __half22float2(w2[3]);
            a = fmaf(s0.x, w0.x, a); a = fmaf(s0.y, w0.y, a); a = fmaf(s0.z, w1.x, a); a = fmaf(s0.w, w1.y, a);
            a = fmaf(s1.x, w2_.x, a); a = fmaf(s1.y, w2_.y, a); a = fmaf(s1.z, w3.x, a); a = fmaf(s1.w, w3.y, a);
        }
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o);
        if (lane == 0) dots[((size_t) r * (M + 1) + j) * H + h] = a;
    }
}

// ---- head: rmr, silu latent t (R, LR), post (R, H) -- one block
__global__ __launch_bounds__(256) void headK(const float* __restrict__ dots, float* __restrict__ t, float* __restrict__ rmr, float* __restrict__ post, int M, int R)
{
    __shared__ float rm[32 * H];
    for (int i = threadIdx.x; i < R * H; i += 256)
    {
        const int r = i / H, h = i % H;
        rm[i] = rsqrtf(dots[((size_t) r * (M + 1) + M) * H + h] / (float) D + 1e-6f);
        rmr[i] = rm[i];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < R * LR; idx += 256)
    {
        const int r = idx / LR, i = idx % LR;
        const float* dr = dots + ((size_t) r * (M + 1) + i) * H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h) v = fmaf(rm[r * H + h], dr[h], v);
        v *= 0.25f;
        t[idx] = v * sigmoidf_(v);
    }
    if (post)
        for (int i = threadIdx.x; i < R * H; i += 256)
        {
            const int r = i / H, h = i % H;
            const float* dr = dots + ((size_t) r * (M + 1) + LR + h) * H;
            float v = 0.0f;
            #pragma unroll
            for (int hh = 0; hh < H; ++hh) v = fmaf(rm[r * H + hh], dr[hh], v);
            post[i] = 2.0f * sigmoidf_(v * 0.25f);
        }
}

// ---- B3: block = QPB column quads x 4 streams, one wave per (quad, stream); t in LDS; stream gates
// combined through LDS
template <int QPB, int NIT>
__global__ __launch_bounds__(QPB * 128) void finB3(const float* __restrict__ streams, const float* __restrict__ tg, const float* __restrict__ rmr,
    const half* __restrict__ upt, const half* __restrict__ w, half* __restrict__ mixed, int R)
{
    __shared__ float ts[8 * LR];
    __shared__ float4 gs[QPB][H][8];
    const int nt = QPB * 128;
    for (int i = threadIdx.x; i < R * LR; i += nt) ts[i] = tg[i];
    __syncthreads();
    const int wv = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q = wv >> 2, h = wv & 3;
    const int c = blockIdx.x * QPB + q;
    const int D4 = D / 4;
    // (h, c) slab: LR x 4 halves contiguous; lane covers ranks 2 * lane + 64 * it, +1
    int4 u[NIT];
    #pragma unroll
    for (int it = 0; it < NIT; ++it) u[it] = *(const int4*) (upt + (((size_t) h * D4 + c) * LR + it * 64 + lane * 2) * 4);
    for (int r = 0; r < R; ++r)
    {
        float4 g = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            const int i0 = it * 64 + lane * 2;
            const float t0 = ts[r * LR + i0], t1 = ts[r * LR + i0 + 1];
            const half2* u2 = (const half2*) &u[it];
            float2 a0 = __half22float2(u2[0]), a1 = __half22float2(u2[1]), b0 = __half22float2(u2[2]), b1 = __half22float2(u2[3]);
            g.x = fmaf(t0, a0.x, fmaf(t1, b0.x, g.x)); g.y = fmaf(t0, a0.y, fmaf(t1, b0.y, g.y));
            g.z = fmaf(t0, a1.x, fmaf(t1, b1.x, g.z)); g.w = fmaf(t0, a1.y, fmaf(t1, b1.y, g.w));
        }
        for (int o = 16; o > 0; o >>= 1) { g.x += __shfl_xor(g.x, o); g.y += __shfl_xor(g.y, o); g.z += __shfl_xor(g.z, o); g.w += __shfl_xor(g.w, o); }
        if (lane == 0)
        {
            const float4 sv = ((const float4*) (streams + ((size_t) r * H + h) * D))[c];
            const half2* wq = (const half2*) (w + (size_t) h * D + 4 * c);
            float2 w0 = __half22float2(wq[0]), w1 = __half22float2(wq[1]);
            const float coef = rmr[r * H + h] * 0.25f;
            gs[q][h][r & 7] = make_float4(sigmoidf_(g.x) * coef * w0.x * sv.x, sigmoidf_(g.y) * coef * w0.y * sv.y,
                                          sigmoidf_(g.z) * coef * w1.x * sv.z, sigmoidf_(g.w) * coef * w1.y * sv.w);
        }
        __syncthreads();
        if (h == 0 && lane == 0)
        {
            float4 o = gs[q][0][r & 7];
            #pragma unroll
            for (int hh = 1; hh < H; ++hh) { float4 v = gs[q][hh][r & 7]; o.x += v.x; o.y += v.y; o.z += v.z; o.w += v.w; }
            half2* out2 = (half2*) (mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        __syncthreads();
    }
}

int main(int argc, char** argv)
{
    const int R = argc > 1 ? atoi(argv[1]) : 1;
    const int M = LR + H;
    std::vector<half*> fn(NS), up(NS), wv(NS);
    std::vector<half> hb((size_t) M * H * D);
    for (auto& x : hb) x = __float2half((rand() / (float) RAND_MAX - 0.5f) * 0.02f);
    for (int i = 0; i < NS; ++i)
    {
        CK(hipMalloc(&fn[i], (size_t) M * H * D * 2)); CK(hipMemcpy(fn[i], hb.data(), (size_t) M * H * D * 2, hipMemcpyHostToDevice));
        CK(hipMalloc(&up[i], (size_t) H * D * LR * 2)); CK(hipMemcpy(up[i], hb.data(), (size_t) H * D * LR * 2, hipMemcpyHostToDevice));
        CK(hipMalloc(&wv[i], (size_t) H * D * 2)); CK(hipMemcpy(wv[i], hb.data(), (size_t) H * D * 2, hipMemcpyHostToDevice));
    }
    float *st, *dots, *t, *rmr, *post; half* mixed;
    std::vector<float> hs((size_t) R * H * D); for (auto& x : hs) x = rand() / (float) RAND_MAX - 0.5f;
    CK(hipMalloc(&st, hs.size() * 4)); CK(hipMemcpy(st, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dots, (size_t) R * (M + 1) * H * 4)); CK(hipMalloc(&t, (size_t) R * LR * 4));
    CK(hipMalloc(&rmr, R * H * 4)); CK(hipMalloc(&post, R * H * 4)); CK(hipMalloc(&mixed, (size_t) R * D * 2));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto timeit = [&] (const char* name, auto fnc, double mb)
    {
        for (int i = 0; i < 2 * NS; ++i) fnc(i);
        CK(hipDeviceSynchronize());
        const int n = 20 * NS;
        CK(hipEventRecord(e0));
        for (int i = 0; i < n; ++i) fnc(i);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        const double us = ms * 1000.0 / n;
        printf("%-28s %7.2f us  %6.0f GB/s\n", name, us, mb > 0 ? mb * 1e-3 / (us * 1e-6) : 0.0);
    };
    const double mbA = (double) M * H * D * 2 / 1e6, mbB = (double) H * D * LR * 2 / 1e6;
    timeit("A3 dots", [&] (int i) { dotsA3<10><<<(M + 1) / 2 + 1, 256>>>(st, fn[i % NS], dots, M, R); }, mbA);
    timeit("head", [&] (int i) { headK<<<1, 256>>>(dots, t, rmr, post, M, R); }, 0);
    timeit("B3 finalize QPB=2", [&] (int i) { finB3<2, 5><<<D / 4 / 2, 256>>>(st, t, rmr, up[i % NS], wv[i % NS], mixed, R); }, mbB);
    timeit("B3 finalize QPB=4", [&] (int i) { finB3<4, 5><<<D / 4 / 4, 512>>>(st, t, rmr, up[i % NS], wv[i % NS], mixed, R); }, mbB);
    timeit("A3+head+B3(2)", [&] (int i) {
        dotsA3<10><<<(M + 1) / 2 + 1, 256>>>(st, fn[i % NS], dots, M, R);
        headK<<<1, 256>>>(dots, t, rmr, post, M, R);
        finB3<2, 5><<<D / 4 / 2, 256>>>(st, t, rmr, up[i % NS], wv[i % NS], mixed, R); }, mbA + mbB);
    return 0;
}
