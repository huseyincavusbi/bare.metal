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
    ginp[bt * C + i] = acc;
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
        gxr[j] = inv_rms * (gr[j] * w[j] - n * c1);
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
