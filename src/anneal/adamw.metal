#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// adamw_step (AdamW optimizer, one step, bias-corrected)
// Per element i (in-place update of m, v, w):
//   m = b1*m + (1-b1)*g
//   v = b2*v + (1-b2)*g*g
//   m_hat = m / bias1        (bias1 = 1 - b1^t, precomputed on CPU)
//   v_hat = v / bias2        (bias2 = 1 - b2^t)
//   w = w*(1 - lr*wd) - lr * m_hat / (sqrt(v_hat) + eps)   (decoupled weight decay)
// Matches torch.optim.AdamW.
//
// buffers: [0]=w [1]=g [2]=m [3]=v   (all length N, in-place rw)
// params : constant float* [6] = {lr, b1, b2, eps, wd, bias1}
//          constant float& bias2 = bias2  [buffer(5)]  -- note: bias1 lives in params[5]
// For simplicity we pass a 7-float array: [0]=lr [1]=b1 [2]=b2 [3]=eps [4]=wd [5]=bias1 [6]=bias2
// [4]=N (int) in buffer(4)
// ----------------------------------------------------------------
kernel void adamw_step(
    device float* w          [[buffer(0)]],
    device const float* g    [[buffer(1)]],
    device float* m          [[buffer(2)]],
    device float* v          [[buffer(3)]],
    constant int& N          [[buffer(4)]],
    constant float* hp       [[buffer(5)]],   /* [lr,b1,b2,eps,wd,bias1,bias2] */
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= N) return;
    float lr = hp[0], b1 = hp[1], b2 = hp[2], eps = hp[3], wd = hp[4];
    float bias1 = hp[5], bias2 = hp[6];
    float gi = g[gid];
    float m_new = b1 * m[gid] + (1.0f - b1) * gi;
    float v_new = b2 * v[gid] + (1.0f - b2) * gi * gi;
    m[gid] = m_new;
    v[gid] = v_new;
    float m_hat = m_new / bias1;
    float v_hat = v_new / bias2;
    w[gid] = w[gid] * (1.0f - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps);
}
