attention计算公式: 
$$
\text{attn-weight} = \text{softmax}(\frac{QK^T}{\sqrt{d}})
$$
若是令 $QK^T=S$, 那么softmax($S$) 实际执行的操作如下:
```
S_max = rowmax(S)
S = S-S_max
out = exp(S)/(rowsum(exp(S)))
```
这一步涉及到求rowmax. 在flash attention的计算中, 他会迭代更新. 请注意, 在计算反向传播的过程中, 没必要对 rowmax(S) 这一步求导. 因为safe softmax的公式等价于:
$$
\frac{\exp(QK^T-M)}{\text{rowsum}(\exp(QK^T-M))}=\frac{\exp(QK^T)}{\text{rowsum}(\exp(QK^T))}
$$
