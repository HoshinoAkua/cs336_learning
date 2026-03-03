import triton
from triton import language as tl
import torch, math

@triton.jit
def _FA_forward_triton_kernel(q_ptr, k_ptr, v_ptr, 
                                o_ptr, max_logit_ptr, LSE_ptr,
                                stride_qb, stride_qh, stride_qq, stride_qd,
                                stride_kb, stride_kh, stride_kk, stride_kd,
                                stride_vb, stride_vh, stride_vk, stride_vd,
                                stride_ob, stride_oh, stride_oq, stride_od,
                                Sq, Sk, # q, k 的 token 数目
                                log2e:tl.constexpr,
                                Dqk:tl.constexpr, Dvo:tl.constexpr, # 我们假设 Dqk/Dvo 的性质很好, 都是2的幂次
                                BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr):
    
    # query:[B, H, Sq, D]
    # key: [B, H, Sk, D]
    # value; [B, H, Sk, D]

    '''
    grid 的形状是三维的. 我们让每一个 Program 处理 一个 Head 上的 Q[some rows, :] @ K^T. 这样在 Q 的行间并行
    '''

    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    B = tl.num_programs(0)
    H = tl.num_programs(1)

    # 读取恒定的Q_part 和 滑动的 K_part 
    # 在使用tensor.transpose的情况下, 不会重排stride, 可能会出现类似于 (M,1,N) 这种内存布局
    # 因此需要考虑如何正确地计算index和offsets, 保证在不同的形状下适配

    batch_idx = bid 
    head_idx = hid
    qo_row_idx = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:,None] # 我们读取的 query 的起始idx + idx offset
    qk_col_idx = tl.arange(0,Dqk)[None,:]
    
    # 读取 key 和 value 的对应的行
    kv_row_idx = 0 + tl.arange(0, BLOCK_SIZE_N)[:, None] 
    vo_col_idx = tl.arange(0, Dvo)[None,:]
    
    qo_mask = qo_row_idx < Sq
    q_ptrs = q_ptr + (
             batch_idx * stride_qb + 
             head_idx * stride_qh + 
             qo_row_idx * stride_qq + 
             qk_col_idx * stride_qd)
            
    q = tl.load(q_ptrs, qo_mask, other=0.0)

    o_ptrs = o_ptr + (
        batch_idx * stride_ob +
        head_idx * stride_oh +
        qo_row_idx * stride_oq +
        vo_col_idx * stride_od
    )
    # max_logits : [B, H, Sq]
    # LSE : [B, H, Sq]
    s0 = H * Sq
    LSE_ptrs = LSE_ptr + (tl.arange(0, BLOCK_SIZE_M) + mid * BLOCK_SIZE_M) + batch_idx * s0 + head_idx * Sq
    max_logit_ptrs = max_logit_ptr + tl.arange(0, BLOCK_SIZE_M) + mid * BLOCK_SIZE_M + batch_idx * s0 + head_idx * Sq
    lse_mask = (tl.arange(0, BLOCK_SIZE_M) + mid * BLOCK_SIZE_M) < Sq


    max_logit = tl.zeros(shape=(BLOCK_SIZE_M, ), dtype=tl.float32) + -1e9
    LSE = tl.zeros_like(max_logit)
    acc = tl.zeros(shape=(BLOCK_SIZE_M, Dvo),dtype=tl.float32)


    NUM_STEP = tl.cdiv((mid + 1) * BLOCK_SIZE_M , BLOCK_SIZE_N)
    low = (mid * BLOCK_SIZE_M) // BLOCK_SIZE_N

    for step in tl.range(0, NUM_STEP):
        
        kv_mask = kv_row_idx < Sk
        k_ptrs = k_ptr + (
            batch_idx * stride_kb +
            head_idx * stride_kh + 
            kv_row_idx * stride_kk +
            qk_col_idx * stride_kd
        )

        v_ptrs = v_ptr + (
            batch_idx * stride_vb + 
            head_idx * stride_vh +
            kv_row_idx * stride_vk +
            vo_col_idx * stride_vd
        )

        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        
        # 计算attention weight
        attn_weight = tl.dot(q, tl.trans(k))/math.sqrt(Dqk) * log2e
        
        if step >= low:
            # 要计算mask
            row_mask = tl.arange(0,BLOCK_SIZE_M)[:,None]
            col_mask = tl.arange(0,BLOCK_SIZE_N)[None,:]
            # 对于attn_weight的每一个元素的坐标(x,y), 如果 y > x , 那就是float(“-inf”), 否则保持原样
            # x = x0 + bar; y = y0 + step * BLOCK_SIZE_N
            # 其中 (x, y) 表示在整个attention中的坐标, (x0, y0)表示在这个attn_weight tile中的坐标
            # 那么计算公式等价于: y0 > x0 + mid * BLOCK_SIZE_M - step * BLOCK_SIZE_N
            mask = col_mask > row_mask + (mid * BLOCK_SIZE_M - step * BLOCK_SIZE_N)
            mask = tl.where(mask, -1e9, 0)
            attn_weight += mask
        

        new_max = tl.max(attn_weight, axis=-1) #(BLOCK_SIZE_M, )
        new_max = tl.maximum(new_max, max_logit)
        delta = new_max - max_logit
        max_logit = new_max
        
        max_logit_expend = max_logit[:,None] #(BLOCK_SIZE_M, 1)
        attn_weight -= max_logit_expend

        attn_score = tl.exp2(attn_weight).to(v.dtype) # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        o = tl.dot(attn_score, v) # (BLOCK_SIZE_M, Dvo)
        
        exp_delta = tl.exp2(-delta)
        acc = acc * exp_delta[:,None] + o
        LSE = LSE * exp_delta + tl.sum(attn_score, axis=-1)

        
        kv_row_idx += BLOCK_SIZE_N
    acc = acc / LSE[:,None]

    tl.store(o_ptrs, acc, mask=qo_mask)
    tl.store(LSE_ptrs, LSE, mask=lse_mask)
    tl.store(max_logit_ptrs, max_logit, mask=lse_mask)


@triton.jit
def _backward_preprocess(Delta_ptr, do_ptr, o_ptr, 
                         do_stride_B, do_stride_H, do_stride_S, do_stride_D, 
                         o_stride_B, o_stride_H, o_stride_S, o_stride_D, 
                         Sq, BLOCK_SIZE_M:tl.constexpr, Do:tl.constexpr):
    '''
    这个函数用来计算 rowsum(do \odot o) = rowsum(dp \odot p) 
    '''
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    # Delta : [B, H, Sq]
    H = tl.num_programs(1)
    do_ptrs = do_ptr + (
          bid * do_stride_B
        + hid * do_stride_H
        + (mid * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)[:,None]) * do_stride_S
        + tl.arange(0,Do)[None,:] * do_stride_D
    )
    do_mask = (mid * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)[:,None]) < Sq

    o_ptrs = o_ptr + (
          bid * o_stride_B
        + hid * o_stride_H
        + (mid * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)[:,None]) * o_stride_S
        + tl.arange(0,Do)[None,:] * o_stride_D
    )
    
    Delta_ptrs = Delta_ptr + (mid * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)) + bid * (H * Sq) + hid * Sq
    Delta_mask = (mid * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)) < Sq

    do = tl.load(do_ptrs, mask=do_mask, other=0.0)
    o = tl.load(o_ptrs, mask=do_mask, other=0.0)
    Delta = tl.sum(o * do, axis=-1)
    tl.store(Delta_ptrs, Delta, Delta_mask)



@triton.jit
def _FA_backward_triton_kernel(do_ptr, dq_ptr, dk_ptr, dv_ptr,
                               q_ptr, k_ptr, v_ptr, Delta_ptr,
                               
                               do_stride_B, do_stride_H, do_stride_S, do_stride_D, 
                               dq_stride_B, dq_stride_H, dq_stride_S, dq_stride_D,
                               dk_stride_B, dk_stride_H, dk_stride_S, dk_stride_D,
                               dv_stride_B, dv_stride_H, dv_stride_S, dv_stride_D,
                               # 这里我还是把dq dkv的stride写上, 主要是假如qkv是split出来的, 而dq = zeros_like(q)的话, 会出现dq和q的stride不同的情况
                               q_stride_B, q_stride_H, q_stride_S, q_stride_D,
                               k_stride_B, k_stride_H, k_stride_S, k_stride_D,
                               v_stride_B, v_stride_H, v_stride_S, v_stride_D,
                               
                               LSE_ptr, max_logits_ptr, 
                               Sq, Sk, # q, k 的 token 数目
                               log2e:tl.constexpr, Dqk:tl.constexpr, Dvo:tl.constexpr,
                               BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr):
    bid = tl.program_id(axis=0)
    hid = tl.program_id(axis=1)
    nid = tl.program_id(axis=2)
    H = tl.num_programs(1)
    
    # 在 backward 中我们反过来了, 外循环读取k、v、dk、dv, 内循环读取q、o、do、dq
    do_ptrs = do_ptr + (
          bid * do_stride_B 
        + hid * do_stride_H
        + (Sq - BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:,None]) * do_stride_S
        + tl.arange(0,Dvo)[None,:] * do_stride_D
    )
    q_ptrs = q_ptr + (
         bid * q_stride_B
       + hid * q_stride_H
       + (Sq - BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:,None]) * q_stride_S
       + tl.arange(0,Dqk)[None,:] * q_stride_D
    )

    dq_ptrs = dq_ptr + (
        bid * dq_stride_B
        + hid * dq_stride_H
        + (Sq - BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:,None]) * dq_stride_S
        + tl.arange(0, Dqk)[None,:] * dq_stride_D
    )

    k_ptrs = k_ptr + (
          bid * k_stride_B
        + hid * k_stride_H
        + (nid * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)[:,None]) * k_stride_S
        + tl.arange(0, Dqk)[None,:] * k_stride_D
    )

    v_ptrs = v_ptr + (
          bid * v_stride_B
        + hid * v_stride_H
        + (nid * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)[:,None]) * v_stride_S
        + tl.arange(0,Dvo)[None,:] * v_stride_D
    )
    
    kv_mask = (nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[:,None]) < Sk
    
    k = tl.load(k_ptrs, kv_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
    
    one_dim_tensor_offsets = (H * Sq) * bid + Sq * hid + (Sq - BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))


    num_steps = tl.cdiv(Sq - nid * BLOCK_SIZE_N, BLOCK_SIZE_M)
    low = (Sq - BLOCK_SIZE_N * (nid+1)) // BLOCK_SIZE_M
    
    dv_acc = tl.zeros(shape=(BLOCK_SIZE_N, Dvo), dtype=tl.float32)
    dk_acc = tl.zeros(shape=(BLOCK_SIZE_N, Dqk), dtype=tl.float32)
    
    for step in tl.range(num_steps):
        
        qo_mask = (Sq - BLOCK_SIZE_M * (step + 1) 
                   + tl.arange(0, BLOCK_SIZE_M)[:,None]) >= 0
        q = tl.load(q_ptrs, mask=qo_mask, other=0.0)
        do = tl.load(do_ptrs, mask=qo_mask, other=0.0)


        max_logits_ptrs = max_logits_ptr + one_dim_tensor_offsets
        LSE_ptrs = LSE_ptr + one_dim_tensor_offsets
        Delta_ptrs = Delta_ptr + one_dim_tensor_offsets
        
        one_dim_tensor_mask = (Sq - (BLOCK_SIZE_M) * (step + 1)
                               + tl.arange(0, BLOCK_SIZE_M)) >= 0
        max_logits = tl.load(max_logits_ptrs, mask=one_dim_tensor_mask, other=0.0)[:, None]
        LSE = tl.load(LSE_ptrs, mask=one_dim_tensor_mask, other=1e9)[:,None]
        Delta = tl.load(Delta_ptrs, mask=one_dim_tensor_mask, other=0.0)[:,None]
        
        
        S = tl.dot(q,tl.trans(k))
        if step >= low:
            abs_x = tl.arange(0, BLOCK_SIZE_M)[:,None] + Sq - ((step+1) * BLOCK_SIZE_M)
            abs_y = tl.arange(0, BLOCK_SIZE_N)[None,:] + nid * BLOCK_SIZE_N
            attn_mask = (abs_y > abs_x)
            attn_mask = tl.where(attn_mask, -1e9, 0)

            S += attn_mask
        S = S * log2e/math.sqrt(Dqk) - max_logits
        P = ( tl.exp2(S)/LSE )

        dP = tl.dot(do, tl.trans(v))
        dv_acc += tl.dot(tl.trans(P), do)
        dS = (P * (dP - Delta)/math.sqrt(Dqk)) #[BM, BN]
        # k: [BN, Dqk]
        dk_acc += tl.dot(tl.trans(dS), q)
        dq = tl.dot(dS, k)
        
        tl.atomic_add(dq_ptrs, dq, mask=qo_mask)
        # 循环完毕之后记得移动 q, dq, o, do, 以及 one_dim_tensors 的

        q_ptrs -= BLOCK_SIZE_M * q_stride_S
        dq_ptrs -= BLOCK_SIZE_M * dq_stride_S
        do_ptrs -= BLOCK_SIZE_M * do_stride_S
        one_dim_tensor_offsets -= BLOCK_SIZE_M 


    dk_ptrs = dk_ptr + (
          bid * dk_stride_B
        + hid * dk_stride_H
        + (nid * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)[:, None]) * dk_stride_S
        + tl.arange(0, Dqk)[None, :] * dk_stride_D
    )

    dv_ptrs = dv_ptr + (
          bid * dv_stride_B
        + hid * dv_stride_H
        + (nid * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)[:, None]) * dv_stride_S
        + tl.arange(0, Dvo)[None, :] * dv_stride_D
    )

    tl.store(dk_ptrs, dk_acc, mask=kv_mask)
    tl.store(dv_ptrs, dv_acc, mask=kv_mask)



class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:torch.Tensor
                , K:torch.Tensor
                , V:torch.Tensor):
        B, H, S, Dqk = Q.shape
        Dov = V.shape[-1]
        
        out = torch.zeros(size=(B, H, S, Dov)).to(DEVICE)
        LSE = torch.zeros(size=(B,H,S),dtype=torch.float32).to(DEVICE)
        max_logits = torch.zeros_like(LSE).to(DEVICE)

        grid = lambda meta:(B, H, triton.cdiv(S, meta["BLOCK_SIZE_M"]))

        _FA_forward_triton_kernel[grid](
            Q, K, V, out, max_logits,
            LSE, *Q.stride(), 
            *K.stride(),
            *V.stride(),
            *out.stride(),
            S, S, 1.4426950408889634,
            Dqk, Dov, 16, 16
        )

        ctx.save_for_backward(Q, K, V, out, LSE, max_logits)
        return out

    
    @staticmethod
    def backward(ctx, grad_outputs):
        Q, K, V, out, LSE, max_logits = ctx.saved_tensors
        B, H, S, Dqk = Q.shape
        Dvo = V.shape[-1]
        grid = lambda meta:(B, H, triton.cdiv(S, meta["BLOCK_SIZE_N"]))

        dq = torch.zeros(Q.shape, dtype=torch.float32).to(DEVICE)
        dk = torch.zeros(K.shape, dtype=torch.float32).to(DEVICE)
        dv = torch.zeros(V.shape, dtype=torch.float32).to(DEVICE)

        Delta = torch.zeros(size=(B,H,S), dtype=torch.float32).to(DEVICE)
        
        _backward_preprocess[(B, H, triton.cdiv(S, 32))](Delta, grad_outputs, out, 
                                       *grad_outputs.stride(),
                                       *out.stride(),
                                       S, 32, Dvo)
        
        # ref_delta = torch.sum(out * grad_outputs, dim=-1)
        # torch.testing.assert_close(ref_delta, Delta,rtol=0.01, atol=0.01)

        _FA_backward_triton_kernel[grid](   
            grad_outputs, dq, dk, dv, Q, K, V, Delta,
            *grad_outputs.stride(),
            *dq.stride(),
            *dk.stride(),
            *dv.stride(),
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            LSE,
            max_logits,
            S, S, 1.4426950408889634,
            Dqk,
            Dvo,
            32,32
        )
        return dq, dk, dv


DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

triton_attention = FlashAttention.apply
def test_flashattention_kernel(B, H, N, Dh, device=DEVICE, atol=5e-3):
    # create data
    q = torch.randn(size=(B, H, N, Dh), dtype=torch.float32).to(device)
    k = torch.randn(size=(B, H, N, Dh), dtype=torch.float32).to(device)
    v = torch.randn(size=(B, H, N, Dh), dtype=torch.float32).to(device)
    for _ in [q, k, v]:
        _.requires_grad = True
     # idk why I made scale a parameter to be passed in, whatever too late now
    # forward pass
    tri_out = triton_attention(q, k, v)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0) 
    print("passed fwd")

    # backward pass (triton)
    dLdout = 0.1 * torch.randn_like(q)
    tri_out.backward(dLdout, retain_graph=True)
    dLdq_tri, dLdk_tri, dLdv_tri = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None
    # backward pass (torch)
    ref_out.backward(dLdout, retain_graph=True)
    dLdq_ref, dLdk_ref, dLdv_ref = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None
    torch.testing.assert_close(dLdq_tri, dLdq_ref, atol=atol, rtol=0)
    torch.testing.assert_close(dLdk_tri, dLdk_ref, atol=atol, rtol=0)
    torch.testing.assert_close(dLdv_tri, dLdv_ref, atol=atol, rtol=0)
    print("Passed bwd")

if __name__ == "__main__":
    # always run unit-tests
    test_flashattention_kernel(1, 1, 189, 64) # without block masking
    test_flashattention_kernel(1, 1, 128, 64) # without block masking
    test_flashattention_kernel(1, 1, 128, 128) # without block masking
    test_flashattention_kernel(32, 8, 69, 128) # with block masking