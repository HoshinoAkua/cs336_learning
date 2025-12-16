# from turtle import forward
import torch
from math import sqrt
from einops import einsum, rearrange
import einops
from jaxtyping import Float, Int
from typing import Union
from torch import Tensor, nn

class Linear(nn.Module):
    def __init__(self, 
                 d_in:int, 
                 d_out:int,
                 weight:Union[Float[Tensor, " d_out d_in"] | None] = None
                 ):
        super().__init__()
        if weight == None:
            sigma = sqrt(2/(d_in + d_out))
            weight = torch.empty(size = (d_out, d_in))
            torch.nn.init.trunc_normal_(weight, std=sigma, a=-3.0*sigma, b=3.0*sigma)

        self.weight = nn.Parameter(weight)
    
    def forward(self, in_features: Float[Tensor, " ... d_in"]):
        return torch.einsum("ji,...i-> ...j", self.weight, in_features)

class Embedding(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 weight: Float[Tensor, " vocab_size d_model"]=None):
        super().__init__()
        if weight == None:
            weight = torch.randn(size = (vocab_size, d_model))
        self.weight = nn.Parameter(weight)
    
    def forward(self, token_ids: Int[Tensor, "..."]):
        return self.weight[token_ids]

class SwiGLU(nn.Module):
    def __init__(self,  d_model: int, 
                        d_ff: int,
                        w1_weight:Union[Float[Tensor, "d_ff d_model"]|None],
                        w2_weight:Union[Float[Tensor, "d_model d_ff"]|None],
                        w3_weight:Union[Float[Tensor, "d_ff d_model"]|None]) -> None:
        super().__init__()

        
        self.weight_list = nn.ParameterList()
        for weight,shape in zip([w1_weight, w2_weight, w3_weight],
                        [(d_ff, d_model), (d_model, d_ff), (d_ff, d_model)]):
            if weight == None:
                self.weight_list.append(self.init_weight(*shape))

            else:
                self.weight_list.append(nn.Parameter(weight))


    def init_weight(self, out_dim, in_dim):
        sigma = sqrt(2/(out_dim + in_dim))
        weight = torch.randn(size=(out_dim, in_dim))
        torch.nn.init.trunc_normal_(weight, std=sigma, a=-3*sigma, b=3*sigma)
        return nn.Parameter(weight)
    
    def forward(self, in_feature:Float[Tensor, "... d_model"]):
        up_proj, down_proj, gate_proj = self.weight_list
        up_inter = torch.einsum("... i, ji->... j", in_feature, up_proj)
        gate_inter = nn.SiLU(torch.einsum("... i,ji->... j", in_feature, gate_proj))*up_inter
        return torch.einsum("... j,ij", gate_inter, down_proj)

class Softmax(nn.Module):
    def __init__(self, dim:int) -> None:
        super().__init__()
        self.dim = dim
    def forward(self, in_feature:Float[Tensor, "..."])->Float[Tensor, "..."]:
        return in_feature.softmax(dim=self.dim)


class SDPA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = Softmax(dim=-1)
    def forward(self, 
                Q:Float[Tensor, "... queries d_k"],
                K:Float[Tensor, "... keys d_k"],
                V:Float[Tensor, "... values d_v"],
                mask:Float[Tensor, "... queries keys"]):
        d_k = Q.shape[-1]
        attn_weight = torch.einsum("...qd,...kd->...qk", Q, K)/sqrt(d_k)+mask
        attn_logits = self.softmax(attn_weight)

        return torch.einsum("bhqk, bhkv->bhqv", attn_logits, V)

class MHA(nn.Module):
    def __init__(self, 
                d_k:int,
                d_model:int,
                num_heads:int,
                query_weight:Float[Tensor, "d_k d_model"]=None, 
                key_weight:Float[Tensor, "d_k d_model"]=None, 
                value_weight:Float[Tensor, "d_k d_model"]=None,
                out_weight:Float[Tensor, "d_model d_k"]=None) -> None:
        super().__init__()
        
        self.weight_list = nn.ParameterList()
        self.num_heads = num_heads
        self.attn = SDPA()
        self.d_k = d_k
        self.d_model = d_model

        for weight in [query_weight, key_weight, value_weight, out_weight]:
            if weight is None:
                self.weight_list.append(self.init_weight(d_k,d_model))
            else:
                self.weight_list.append(nn.Parameter(weight))



    def init_weight(self, d_k:int, d_model:int)->nn.Parameter:
        sigma = sqrt(2/(d_k+d_model))
        weight = torch.empty(size=(d_k, d_model))
        torch.nn.init.trunc_normal_(weight, std=sigma, a=-3*sigma, b=3*sigma)
        return nn.Parameter(weight)

    def forward(self, in_feature:Float[Tensor,"... seq d_model"]):
        q_proj, k_proj, v_proj, o_proj = self.weight_list
        def map_and_reshape(input:Float[Tensor, "bsz seq d_model"], weight:Float[Tensor, "d_k d_model"])->Float[Tensor, "bsz num_heads seq d_k"]:
            total_dim = input.shape[0]
            # input_inter = torch.einsum("bsm, km->bsk", input, weight)
            output = einops.einsum(input, weight, "b s m, (h d) m-> b h s d",h=self.num_heads, d=int(total_dim/self.num_heads))
            # bsz, seq, total_dim = input_inter.shape
            # d_k = int(total_dim/self.num_heads)
            # output:Float[Tensor, "bsz num_heads seq d_k"] = input_inter.view(bsz, seq, self.num_heads, d_k).transpose(1,2)
            
            return output
        Q = map_and_reshape(in_feature, q_proj)
        K = map_and_reshape(in_feature, k_proj)
        V = map_and_reshape(in_feature, v_proj)
        bsz, num_heads, seq, d_k = Q.shape
        mask = torch.triu(-1e9 * torch.ones(seq, seq),diagonal=1).to(Q.device)
        attn_output:Float[Tensor, "bsz seq d_k"] = self.attn(Q, K, V, mask).transpose(1,2).contiguous().view(bsz, seq, num_heads*d_k)
        attn_output = torch.einsum("... i, ji->...j", attn_output, o_proj)
        return attn_output

class RoPE(nn.Module):
    def __init__(self, theta: float, 
                       max_sequence_length: int, 
                       d_k: int) -> None:
        super().__init__()
        self.theta = theta
        self.max_sequence_length = max_sequence_length
        self.d_k = d_k
    
    def _set_cos_sin_cache(self, token_positions:int[Tensor, "... seq"]):
        if self.max_sequence_length == None:
            self.max_sequence_length = token_positions.shape[-1]
        self.







        



