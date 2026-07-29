#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// matmul_backward_inp
// Forward: out[bt,oc] = bias[oc] + sum_i inp[bt,i] * w[oc,i]
// Backward wrt input:  grad_inp[bt,i] = sum_oc gout[bt,oc] * w[oc,i]
// Grid: (C, BT) — one thread per (bt, i). Same gid convention as the
// forward naive kernel (gid.x = inner dim index, gid.y = batch index).
// params: [0]=BT [1]=C [2]=OC
// ----------------------------------------------------------------
kernel void matmul_backward_inp(
    device const float* gout [[buffer(0)]],   // [BT, OC]
    device const float* w    [[buffer(1)]],   // [OC, C]
    device float* ginp       [[buffer(2)]],   // [BT, C]  (must be zeroed)
    constant int* params     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2];
    int bt = gid.y, i = gid.x;
    if (bt >= BT || i >= C) return;
    float acc = 0.0f;
    for (int oc = 0; oc < OC; oc++) {
        acc += gout[bt * OC + oc] * w[oc * C + i];
    }
    ginp[bt * C + i] += acc;   /* accumulate: shared activation grads (e.g. t_norm1 read by Q/K/V) */
}


// ----------------------------------------------------------------
// matmul_backward_inp_bf16
// Same as above but weight is bf16 (cast to float for compute).
// ----------------------------------------------------------------
#if __HAVE_BFLOAT__
kernel void matmul_backward_inp_bf16(
    device const float* gout [[buffer(0)]],
    device const bfloat* w   [[buffer(1)]],
    device float* ginp       [[buffer(2)]],
    constant int* params     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2];
    int bt = gid.y, i = gid.x;
    if (bt >= BT || i >= C) return;
    float acc = 0.0f;
    for (int oc = 0; oc < OC; oc++) {
        acc += gout[bt * OC + oc] * (float)w[oc * C + i];
    }
    ginp[bt * C + i] += acc;
}
#endif

// ----------------------------------------------------------------
// matmul_backward_inp_fp16
// ----------------------------------------------------------------
kernel void matmul_backward_inp_fp16(
    device const float* gout [[buffer(0)]],
    device const half* w     [[buffer(1)]],
    device float* ginp       [[buffer(2)]],
    constant int* params     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2];
    int bt = gid.y, i = gid.x;
    if (bt >= BT || i >= C) return;
    float acc = 0.0f;
    for (int oc = 0; oc < OC; oc++) {
        acc += gout[bt * OC + oc] * (float)w[oc * C + i];
    }
    ginp[bt * C + i] += acc;
}


// ----------------------------------------------------------------
// matmul_backward_w
// Backward wrt weight:  grad_w[oc,i] = sum_bt gout[bt,oc] * inp[bt,i]
// Grid: (C, OC) — one thread per (oc, i).
// params: [0]=BT [1]=C [2]=OC
// ----------------------------------------------------------------
kernel void matmul_backward_w(
    device const float* gout [[buffer(0)]],   // [BT, OC]
    device const float* inp  [[buffer(1)]],   // [BT, C]
    device float* gw         [[buffer(2)]],   // [OC, C]  (must be zeroed)
    constant int* params     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2];
    int oc = gid.y, i = gid.x;
    if (oc >= OC || i >= C) return;
    float acc = 0.0f;
    for (int bt = 0; bt < BT; bt++) {
        acc += gout[bt * OC + oc] * inp[bt * C + i];
    }
    gw[oc * C + i] = acc;
}

// ----------------------------------------------------------------
// rmsnorm_backward_x
// Forward: ss = sum(x^2)/C ; inv_rms = 1/sqrt(ss+eps) ; n[j]=x[j]*inv_rms
//          out[j] = n[j]*w[j]
// Backward wrt input (per row, row length C):
//   c1 = (1/C) * sum_j gout[j]*out[j]
//   grad_x[j] = inv_rms * (gout[j]*w[j] - n[j]*c1)
// One thread per row (matches forward rmsnorm naive convention).
// buffers: [0]=gout[N,C] [1]=w[C] [2]=x[N,C] [3]=gx[N,C]  params[4]=[N,C]  eps[5]
// ----------------------------------------------------------------
kernel void rmsnorm_backward_x(
    device const float* gout [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device const float* x    [[buffer(2)]],
    device float* gx         [[buffer(3)]],
    constant int* p          [[buffer(4)]],
    constant float& eps      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int row = (int)gid;
    if (row >= N) return;
    const device float* xr = x + row * C;
    const device float* gr = gout + row * C;
    device float* gxr = gx + row * C;

    float ss = 0.0f;
    for (int j = 0; j < C; j++) ss += xr[j] * xr[j];
    ss /= (float)C;
    float inv_rms = 1.0f / sqrt(ss + eps);

    // c1 = (1/C) * sum_j gout[j]*out[j], with out[j] = x[j]*inv_rms*w[j]
    float c1 = 0.0f;
    for (int j = 0; j < C; j++) c1 += gr[j] * xr[j] * inv_rms * w[j];
    c1 /= (float)C;

    for (int j = 0; j < C; j++) {
        float n = xr[j] * inv_rms;
        gxr[j] += inv_rms * (gr[j] * w[j] - n * c1);   /* accumulate (residual stream has multiple producers) */
    }
}

// ----------------------------------------------------------------
// rmsnorm_backward_w
// Backward wrt weight:  grad_w[j] = sum_row gout[row,j]*n[row,j]
//   where n[row,j] = x[row,j]*inv_rms(row)
// One thread per column j; loops over N rows accumulating. gw must be zeroed.
// buffers: [0]=gout[N,C] [1]=x[N,C] [2]=gw[C]  params[3]=[N,C]  eps[4]
// ----------------------------------------------------------------
kernel void rmsnorm_backward_w(
    device const float* gout [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float* gw         [[buffer(2)]],
    constant int* p          [[buffer(3)]],
    constant float& eps      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int j = (int)gid;
    if (j >= C) return;
    float acc = 0.0f;
    for (int row = 0; row < N; row++) {
        const device float* xr = x + row * C;
        float ss = 0.0f;
        for (int k = 0; k < C; k++) ss += xr[k] * xr[k];
        ss /= (float)C;
        float inv_rms = 1.0f / sqrt(ss + eps);
        float n = xr[j] * inv_rms;
        acc += gout[row * C + j] * n;
    }
    gw[j] = acc;
}

// ----------------------------------------------------------------
// rope_backward
// Forward (rotate_half, Llama style): for each head h, hd2 = HD/2,
//   freq = 1/theta^(2i/HD); c=cos(pos*freq), s=sin(pos*freq)
//   out[i]    = x[i]*c - x[i+hd2]*s
//   out[i+hd2] = x[i]*s + x[i+hd2]*c
// Backward = transposed rotation R^T (= R(-theta), orthogonal): reads the
// upstream grad gout and writes g. Independent of the forward input values
// (only cos/sin + gout). No weights -> grad_input only.
// One thread per head (matches forward). Grid = NH.
// buffers: [0]=gout_q [1]=gout_k [2]=gq [3]=gk
//          [4]=head_size [5]=pos [6]=theta [7]=n_kv_heads
// ----------------------------------------------------------------
kernel void rope_backward(
    device const float* gout_q [[buffer(0)]],
    device const float* gout_k [[buffer(1)]],
    device float* gq           [[buffer(2)]],
    device float* gk           [[buffer(3)]],
    constant int& head_size    [[buffer(4)]],
    constant int& pos          [[buffer(5)]],
    constant float& theta      [[buffer(6)]],
    constant int& n_kv_heads   [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int h = (int)gid;
    int hd2 = head_size / 2;
    device const float* gqin = gout_q + h * head_size;
    device float* gqout = gq + h * head_size;
    for (int i = 0; i < hd2; i++) {
        float freq = 1.0f / pow(theta, (float)(2*i) / (float)head_size);
        float c = cos((float)pos * freq);
        float s = sin((float)pos * freq);
        float g0 = gqin[i], g1 = gqin[i + hd2];
        gqout[i]      =  g0 * c + g1 * s;          /* R^T row 0 */
        gqout[i + hd2] = -g0 * s + g1 * c;         /* R^T row 1 */
    }
    if (h < n_kv_heads) {
        device const float* gkin = gout_k + h * head_size;
        device float* gkout = gk + h * head_size;
        for (int i = 0; i < hd2; i++) {
            float freq = 1.0f / pow(theta, (float)(2*i) / (float)head_size);
            float c = cos((float)pos * freq);
            float s = sin((float)pos * freq);
            float g0 = gkin[i], g1 = gkin[i + hd2];
            gkout[i]      =  g0 * c + g1 * s;
            gkout[i + hd2] = -g0 * s + g1 * c;
        }
    }
}

// ----------------------------------------------------------------
// gelu_backward (tanh approximation, matches forward gelu_forward)
// Forward: u = 0.79788456*x*(1+0.044715*x^2); t=tanh(u); cdf=0.5*(1+t); out=x*cdf
// Backward: grad_x = gout*(cdf + x*dcdx)
//   dcdx = 0.5*(1-t^2) * 0.79788456*(1 + 3*0.044715*x^2)   [du/dx = 0.79788456*(1+3*0.044715*x^2)]
// One thread per element. buffers: [0]=gout [1]=x [2]=gx  [3]=N
// ----------------------------------------------------------------
kernel void gelu_backward(
    device const float* gout [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float* gx         [[buffer(2)]],
    constant int& N          [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= N) return;
    float xv = x[gid];
    float u = 0.79788456f * xv * (1.0f + 0.044715f * xv * xv);
    float t = tanh(u);
    float cdf = 0.5f * (1.0f + t);
    float dcdx = 0.5f * (1.0f - t * t) * 0.79788456f * (1.0f + 3.0f * 0.044715f * xv * xv);
    gx[gid] = gout[gid] * (cdf + xv * dcdx);
}

// ----------------------------------------------------------------
// swiglu_backward
// Forward: silu = x/(1+e^-x); out = silu*up   (x=gate, up=up projection)
// Backward (s = sigmoid(x) = 1/(1+e^-x)):
//   grad_gate = gout * up * s * (1 + x*(1-s))
//   grad_up   = gout * silu = gout * (x*s)
// One thread per element. buffers: [0]=gate [1]=up [2]=gout [3]=ggate [4]=gup  [5]=N
// ----------------------------------------------------------------
kernel void swiglu_backward(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device const float* gout [[buffer(2)]],
    device float* ggate      [[buffer(3)]],
    device float* gup        [[buffer(4)]],
    constant int& N          [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= N) return;
    float x = gate[gid];
    float u = up[gid];
    float g = gout[gid];
    float s = 1.0f / (1.0f + exp(-x));
    ggate[gid] = g * u * s * (1.0f + x * (1.0f - s));
    gup[gid]   = g * (x * s);
}

// ----------------------------------------------------------------
// add_backward (GPU, no host sync)
// Forward: out = x + y
// Backward: grad_x[i] += grad_out[i]; grad_y[i] += grad_out[i]
// (accumulate, since residual grads have multiple producers)
// One thread per element. buffers: [0]=gout [1]=gx(atomic) [2]=gy(atomic) [3]=N
// ----------------------------------------------------------------
kernel void add_backward(
    device const float* gout [[buffer(0)]],
    device atomic_float* gx  [[buffer(1)]],
    device atomic_float* gy  [[buffer(2)]],
    constant int& N          [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= N) return;
    float g = gout[gid];
    atomic_fetch_add_explicit(gx + gid, g, memory_order_relaxed);
    atomic_fetch_add_explicit(gy + gid, g, memory_order_relaxed);
}

// ----------------------------------------------------------------
// xent_backward (cross-entropy loss backward, the seed of the backward pass)
// Forward: loss = -mean_n log(softmax(logits[n])[target[n]])
//          = -(1/N) sum_n (logits[n,target] - logsumexp(logits[n]))
// Backward: grad_logits[n,j] = (softmax(logits[n])[j] - 1[j==target[n]]) / N
// Fused: one thread per row n. Pass 1 writes exp(x-max) into glogits + sums;
// pass 2 normalizes to softmax and subtracts onehot, divides by N.
// buffers: [0]=logits[N,V] [1]=targets[N] [2]=glogits[N,V]  params[3]=[N,V]
// ----------------------------------------------------------------
kernel void xent_backward(
    device const float* logits [[buffer(0)]],
    device const int*   targets[[buffer(1)]],
    device float* glogits       [[buffer(2)]],
    constant int* p             [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], V = p[1];
    int n = (int)gid;
    if (n >= N) return;
    const device float* lr = logits + n * V;
    device float* gr = glogits + n * V;
    float mx = -INFINITY;
    for (int j = 0; j < V; j++) if (lr[j] > mx) mx = lr[j];
    float sum = 0.0f;
    for (int j = 0; j < V; j++) { float e = exp(lr[j] - mx); gr[j] = e; sum += e; }
    float inv = 1.0f / sum;
    int tgt = targets[n];
    float invN = 1.0f / (float)N;
    for (int j = 0; j < V; j++) {
        float p_j = gr[j] * inv;            /* softmax prob */
        gr[j] = (p_j - (j == tgt ? 1.0f : 0.0f)) * invN;
    }
}

// ----------------------------------------------------------------
// embedding_backward
// Forward: out[t] = wte[token[t]]   (row lookup, [V,D] table)
// Backward: grad_wte[token[t]] += grad_out[t]  (scatter-add; tokens repeat,
//          so accumulation across positions -> atomic float adds).
// gwte MUST be zeroed before dispatch. One thread per (t, d) element.
// buffers: [0]=tokens[T] [1]=gout[T,D] [2]=gwte[V,D] (atomic)  [3]=D
// grid = T*D
// ----------------------------------------------------------------
kernel void embedding_backward(
    device const int* tokens  [[buffer(0)]],
    device const float* gout  [[buffer(1)]],
    device atomic_float* gwte [[buffer(2)]],
    constant int& D           [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int t = (int)gid / D;
    int d = (int)gid - t * D;
    int tok = tokens[t];
    atomic_fetch_add_explicit(gwte + tok * D + d, gout[t * D + d], memory_order_relaxed);
}

// ----------------------------------------------------------------
// attention_backward (causal + GQA + full softmax, sequence mode)
// Forward (per query pos i, head h, kh=h/kv_mul):
//   scores[j] = (Q[h,i] . K[kh,j]) * scale,  j in [0,i]   (causal)
//   p = softmax(scores);  out[h,i] = sum_j p[j] * V[kh,j]
// Backward (given gout[h,i]):
//   grad_p[j]  = gout[h,i] . V[kh,j]
//   gscore[j]  = p[j] * (grad_p[j] - sum_k p[k]*grad_p[k])   (softmax bwd)
//   grad_Q[h,i] = sum_{j<=i} gscore[j] * K[kh,j]             (local)
//   grad_K[kh,j]+= gscore[j] * Q[h,i]        (accumulate i>=j, atomic)
//   grad_V[kh,j]+= p[j]     * gout[h,i]      (accumulate i>=j, atomic)
// One thread per (h,i). gradK/gradV must be zeroed. MAXS bounds S.
// buffers: [0]=gout[NH,S,HD] [1]=Q [2]=K[NKV,S,HD] [3]=V [4]=gQ [5]=gK(atomic) [6]=gV(atomic)
//          [7]=params[NH,S,HD,NKV,kv_mul]  [8]=scale
// ----------------------------------------------------------------
#define ATTN_MAXS 256
kernel void attention_backward(
    device const float* gout  [[buffer(0)]],
    device const float* Q     [[buffer(1)]],
    device const float* K     [[buffer(2)]],
    device const float* V     [[buffer(3)]],
    device float* gQ          [[buffer(4)]],
    device atomic_float* gK   [[buffer(5)]],
    device atomic_float* gV   [[buffer(6)]],
    constant int* p           [[buffer(7)]],
    constant float& scale     [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int NH = p[0], S = p[1], HD = p[2], NKV = p[3], kv_mul = p[4];
    int h = (int)gid / S;
    int i = (int)gid - h * S;
    if (h >= NH || i >= S) return;
    int kh = h / kv_mul;

    const device float* qi = Q   + (i * NH + h) * HD;   /* [S,NH,HD] layout */
    const device float* gi = gout + (i * NH + h) * HD;

    thread float scores[ATTN_MAXS], pr[ATTN_MAXS], gp[ATTN_MAXS], gs[ATTN_MAXS];
    int n = i + 1;   // causal: j in [0, i]

    // forward: scores[j] = (qi . K[kh,j]) * scale ; softmax -> pr[j]
    float mx = -INFINITY;
    for (int j = 0; j < n; j++) {
        const device float* kj = K + (j * NKV + kh) * HD;
        float s = 0.0f;
        for (int d = 0; d < HD; d++) s += qi[d] * kj[d];
        s *= scale;
        scores[j] = s;
        if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (int j = 0; j < n; j++) { pr[j] = exp(scores[j] - mx); sum += pr[j]; }
    float inv = 1.0f / sum;
    for (int j = 0; j < n; j++) pr[j] *= inv;

    // grad_p[j] = gi . V[kh,j] ; gp_dot_p = sum_j p[j]*grad_p[j]
    float gp_dot_p = 0.0f;
    for (int j = 0; j < n; j++) {
        const device float* vj = V + (j * NKV + kh) * HD;
        float dp = 0.0f;
        for (int d = 0; d < HD; d++) dp += gi[d] * vj[d];
        gp[j] = dp;
        gp_dot_p += pr[j] * dp;
    }
    // gscore[j] = p[j]*(grad_p[j] - gp_dot_p)
    for (int j = 0; j < n; j++) gs[j] = pr[j] * (gp[j] - gp_dot_p);

    // grad_Q[h,i,d] = scale * sum_j gscore[j]*K[kh,j,d]   (scale: scores = Q.K * scale)
    device float* gqi = gQ + (i * NH + h) * HD;   /* [S,NH,HD] */
    for (int d = 0; d < HD; d++) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += gs[j] * K[(j * NKV + kh) * HD + d];
        gqi[d] = acc * scale;
    }

    // grad_K[kh,j,d] += scale*gscore[j]*qi[d] ; grad_V[kh,j,d] += p[j]*gi[d]  (atomic, i>=j)
    // scale on grad_K only (scores carry scale); grad_V has no scale (V is linear in out).
    for (int j = 0; j < n; j++) {
        for (int d = 0; d < HD; d++) {
            atomic_fetch_add_explicit(gK + (j * NKV + kh) * HD + d, scale * gs[j] * qi[d], memory_order_relaxed);
            atomic_fetch_add_explicit(gV + (j * NKV + kh) * HD + d, pr[j] * gi[d], memory_order_relaxed);
        }
    }
}

// ----------------------------------------------------------------
// rope_backward_seq (training backward RoPE, full sequence, [S,NH,HD] layout)
// Backward = transposed rotation R^T by pos=s. Independent of forward input values.
//   gq[i]    = gout_q[i]*c + gout_q[i+hd2]*s ;  gq[i+hd2] = -gout_q[i]*s + gout_q[i+hd2]*c
// One thread per (s,h). Grid = S*NH.
// buffers: [0]=gout_q[S,NH,HD] [1]=gout_k[S,NKV,HD] [2]=gq [3]=gk
//          [4]=params{head_size,n_kv_heads,S,NH}  [5]=theta
// ----------------------------------------------------------------
kernel void rope_backward_seq(
    device const float* gout_q [[buffer(0)]],
    device const float* gout_k [[buffer(1)]],
    device float* gq           [[buffer(2)]],
    device float* gk           [[buffer(3)]],
    constant int* p           [[buffer(4)]],
    constant float& theta      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int head_size = p[0], n_kv_heads = p[1], S = p[2], NH = p[3];
    int s = (int)gid / NH;
    int h = (int)gid - s * NH;
    if (s >= S || h >= NH) return;
    int hd2 = head_size / 2;
    const device float* gqin = gout_q + (s * NH + h) * head_size;
    device float* gqout = gq + (s * NH + h) * head_size;
    for (int i = 0; i < hd2; i++) {
        float freq = 1.0f / pow(theta, (float)(2*i) / (float)head_size);
        float c = cos((float)s * freq);
        float s_ = sin((float)s * freq);
        float g0 = gqin[i], g1 = gqin[i + hd2];
        gqout[i]      =  g0 * c + g1 * s_;
        gqout[i + hd2] = -g0 * s_ + g1 * c;
    }
    if (h < n_kv_heads) {
        const device float* gkin = gout_k + (s * n_kv_heads + h) * head_size;
        device float* gkout = gk + (s * n_kv_heads + h) * head_size;
        for (int i = 0; i < hd2; i++) {
            float freq = 1.0f / pow(theta, (float)(2*i) / (float)head_size);
            float c = cos((float)s * freq);
            float s_ = sin((float)s * freq);
            float g0 = gkin[i], g1 = gkin[i + hd2];
            gkout[i]      =  g0 * c + g1 * s_;
            gkout[i + hd2] = -g0 * s_ + g1 * c;
        }
    }
}

// ----------------------------------------------------------------
// residual_rmsnorm_backward
// Fused backward for: y = x + residual; out = rmsnorm(y, weight)
// Forward modifies x in-place to store y (for inference efficiency).
// Backward reads y from x buffer (which contains y after forward).
// Backward:
//   grad_y = rmsnorm_backward_x(grad_out, weight, y)
//   grad_x += grad_y; grad_residual += grad_y
// One thread per row (matches rmsnorm_backward_x convention).
// buffers: [0]=gout[N,C] [1]=w[C] [2]=y[N,C] (stored in x after forward)
//          [3]=gx[N,C] [4]=gres[N,C] params[5]=[N,C] eps[6]
// ----------------------------------------------------------------
kernel void residual_rmsnorm_backward(
    device const float* gout [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device const float* y    [[buffer(2)]],
    device atomic_float* gx  [[buffer(3)]],
    device atomic_float* gres [[buffer(4)]],
    constant int* p          [[buffer(5)]],
    constant float& eps      [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int row = (int)gid;
    if (row >= N) return;
    const device float* yr = y + row * C;
    const device float* gr = gout + row * C;

    float ss = 0.0f;
    for (int j = 0; j < C; j++) ss += yr[j] * yr[j];
    ss /= (float)C;
    float inv_rms = 1.0f / sqrt(ss + eps);

    float c1 = 0.0f;
    for (int j = 0; j < C; j++) c1 += gr[j] * yr[j] * inv_rms * w[j];
    c1 /= (float)C;

    for (int j = 0; j < C; j++) {
        float n = yr[j] * inv_rms;
        float gy = inv_rms * (gr[j] * w[j] - n * c1);
        atomic_fetch_add_explicit(gx + row * C + j, gy, memory_order_relaxed);
        atomic_fetch_add_explicit(gres + row * C + j, gy, memory_order_relaxed);
    }
}

// ----------------------------------------------------------------
// residual_rmsnorm_backward_w
// Weight gradient for fused residual + rmsnorm.
// Forward stores y = x + residual in x buffer (in-place).
// grad_w[j] = sum_n gout[n,j] * y[n,j] * inv_rms[n]
// One thread per weight element j.
// buffers: [0]=gout[N,C] [1]=y[N,C] [2]=gw[C] params[3]=[N,C] eps[4]
// ----------------------------------------------------------------
kernel void residual_rmsnorm_backward_w(
    device const float* gout [[buffer(0)]],
    device const float* y    [[buffer(1)]],
    device float* gw         [[buffer(2)]],
    constant int* p          [[buffer(3)]],
    constant float& eps      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int j = (int)gid;
    if (j >= C) return;

    float acc = 0.0f;
    for (int n = 0; n < N; n++) {
        float y_val = y[n * C + j];
        float ss = 0.0f;
        for (int k = 0; k < C; k++) {
            float yk = y[n * C + k];
            ss += yk * yk;
        }
        ss /= (float)C;
        float inv_rms = 1.0f / sqrt(ss + eps);
        acc += gout[n * C + j] * y_val * inv_rms;
    }
    gw[j] = acc;
}
