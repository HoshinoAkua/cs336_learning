# from turtle import forward
from turtle import forward, position
from numpy import diag
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
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward( self, 
                 Q: Float[Tensor, "... queries d_k"]=None,
                 K: Float[Tensor, "... keys d_k"]=None,
                 V: Float[Tensor, "... values d_v"]=None,
                 mask: Float[Tensor, "... queries keys"]=None):
        d_k = Q.shape[-1]
        keys = K.shape[-2]
        values = V.shape[-2]
        assert keys == values
        attn_weight = torch.einsum("... qd,... kd->...qk", Q,K)/sqrt(d_k) + mask
        attn_weight:Float[Tensor, "... queries keys"] = self.softmax(attn_weight)
        return torch.einsum("... qk, ... kv->... qv", attn_weight, V)


class MHA(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_heads,
                 q_proj_weight: Float[Tensor, " d_k d_in"],
                 k_proj_weight: Float[Tensor, " d_k d_in"],
                 v_proj_weight: Float[Tensor, " d_v d_in"],
                 o_proj_weight: Float[Tensor, " d_model d_v"]) -> None:
        super().__init__()
        self.q_proj = nn.Parameter(q_proj_weight)
        self.k_proj = nn.Parameter(k_proj_weight)
        self.v_proj = nn.Parameter(v_proj_weight)
        self.o_proj = nn.Parameter(o_proj_weight)
        self.d_model = d_model
        self.num_heads = num_heads
        self.sdpa = SDPA()
    
    def forward(self, in_features: Float[Tensor, " ... seq d_in"]) -> Float[Tensor, "... seq d_model"]:
        v = self.v_proj.shape[0]
        def reshape_and_gemm(input:Float[Tensor, "... seq d_in"], weight:Float[Tensor, "d d_in"]):
            d = weight.shape[0]
            d_split = d//self.num_heads
            weight = einops.rearrange(weight, "(h d) i-> h d i", h=self.num_heads, d=d_split)
            return einops.einsum(input, weight, "... s i, h d i-> ... h s d")
        query = reshape_and_gemm(in_features, self.q_proj)
        key = reshape_and_gemm(in_features, self.k_proj)
        value = reshape_and_gemm(in_features, self.v_proj)
        seq = in_features.shape[-2]
        mask = torch.ones(size = (seq, seq),dtype=query.dtype, device = query.device) * (-1e8)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0) # 假设模型 1.没有past_key_value 2.一定是bsz head seq d 这种形状的
        attn_out:Float[Tensor, "bsz head seq d_v//head"] = einops.rearrange(self.sdpa(query, key, value, mask), "... h s d -> ... s (h d)")
        return einops.einsum(attn_out, self.o_proj, "... v, m v-> ... m")



class RoPE(nn.Module):
    def __init__(self, 
                theta: float, 
                max_sequence_length: int, 
                d_k: int) -> None:
        super().__init__()
        self.theta = theta #这个theta就是其他的模型中的base, 默认是10000
        self.max_sequence_length = max_sequence_length
        self.d_k = d_k
        self._get_inv_freq()
        if max_sequence_length != None:
            self._set_cos_sin_cache()
    
    def _get_inv_freq(self):
        assert self.d_k % 2 == 0
        idx = torch.arange(0, self.d_k, 2).float()
        inv_freq:Float[Tensor, self.d_k/2 - 1] = 1.0/(
            self.theta ** (idx/self.d_k)
        ) # 10000^{-2i/d}, 其中10000是theta
        self.register_buffer("inv_freq", inv_freq) #[theta_0, theta_1, ...]

    
    def _set_cos_sin_cache(self):            
        
        freq = torch.einsum("i,j->ij", torch.arange(self.max_sequence_length).float(), self.inv_freq) #[seq, d_k/2]
        freq = torch.cat([freq,freq],dim=-1) #[seq, d_k]
        self.register_buffer("cos", freq.cos())
        self.register_buffer("sin", freq.sin())
    
    def forward(self, token_positions: Int[Tensor, "... seq"]):
        max_pos = token_positions.max().item()
        if self.max_sequence_length == None or self.max_sequence_length <= max_pos:
            self.max_sequence_length = max_pos+1
            self._set_cos_sin_cache()
        return self.cos[token_positions], self.sin[token_positions]

def apply_rope(x:Float[Tensor, "... seq dim"], 
               cos:Float[Tensor, "... seq dim"], sin:Float[Tensor, "... seq dim"]):
    cos = cos.unsqueeze(1) # [bsz 1 seq dim]
    sin = sin.unsqueeze(1)
    x_embed = x * cos + rotate_half(x) * sin
    return x_embed

def rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    return torch.cat([-x2, x1], dim=-1)



class MHA_With_RoPE(nn.Module):
    def __init__(self, 
                d_model: int,
                num_heads: int,
                max_seq_len: int,
                theta: float,
                q_proj_weight: Float[Tensor, " d_k d_in"],
                k_proj_weight: Float[Tensor, " d_k d_in"],
                v_proj_weight: Float[Tensor, " d_v d_in"],
                o_proj_weight: Float[Tensor, " d_model d_v"],) -> None:
        super().__init__()
        # self.d_model = d_model
        self.q_proj = nn.Parameter(q_proj_weight)
        self.k_proj = nn.Parameter(k_proj_weight)
        self.v_proj = nn.Parameter(v_proj_weight)
        self.o_proj = nn.Parameter(o_proj_weight)
        if q_proj_weight == None:
            d_split = d_model//num_heads
        else:
            d_k = q_proj_weight.shape[0]
            d_split = d_k//num_heads

        self.rope = RoPE(theta, max_seq_len, d_split)
        self.sdpa = SDPA()
    
    def forward(self, 
                in_features:Float[Tensor, "... seq d_in"], 
                token_positions: Int[Tensor, "... seq"]):
        
        def map_and_gemm(in_features:Float[Tensor, "... seq d_in"], weight:Float[Tensor, "d_out d_in"]):
            d_out = weight.shape[0]
            d_split = d_out//self.num_heads
            weight = einops.rearrange(weight, "(h d) i->h d i", h=self.num_heads, d=d_split)
            return einops.einsum(in_features, weight, "... s i, h d i -> ... h s d")
        q = map_and_gemm(in_features, self.q_proj)
        k = map_and_gemm(in_features, self.k_proj)
        v = map_and_gemm(in_features, self.v_proj)

        # 绑定rope
        cos, sin = self.rope(token_positions)
        q_embed = apply_rope(q, cos, sin)
        k_embed = apply_rope(k, cos, sin)

        seq_len = token_positions.shape[-1]
        # 计算attn out
        mask = torch.ones(size=(seq_len, seq_len),dtype = torch.float, device = q_embed.device) * (-1e8)
        mask = torch.triu(mask, diagonal=1)
        attn_out = self.sdpa(q_embed, k_embed, v, mask)
        
        #计算最终输出
        attn_out = einops.rearrange(attn_out, "... h s v -> ... s (h v)")
        return einops.einsum(attn_out, self.o_proj, "... s v, m v -> ... s m")

class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model, 
                 eps,
                 weights) -> None:
        super().__init__()
        self.weights = weights
        self.eps = eps
        self.d_model = d_model
    def forward(self, in_features: Float[Tensor, " ... seq d_model"]):
        o_dtype = in_features.dtype
        in_features = in_features.to(torch.float32)
        var = torch.sum((in_features)**2,dim=-1,keepdim=True)/self.d_model
        in_inter = in_features*torch.rsqrt(var+self.eps).to(o_dtype)
        return in_inter * self.weights


class TransformerBlock(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float,
                weights: dict[str, Tensor]) -> None:
        super().__init__()

        self.attn_rmsnorm = RMSNorm(d_model, eps=1e-6, weights=weights["ln1.weight"])
        self.mlp_rmsnorm = RMSNorm(d_model, eps=1e-6, weights=weights["ln2.weight"])

        self.attn = MHA_With_RoPE(d_model,
                                  num_heads,
                                  max_seq_len,
                                  theta,
                                  weights["attn.q_proj.weight"],
                                  weights["attn.k_proj.weight"],
                                  weights["attn.v_proj.weight"],
                                  weights["attn.o_proj.weight"])
        self.mlp = SwiGLU(d_model,
                          d_ff,
                          weights["ffn.w1.weight"],
                          weights["ffn.w2.weight"],
                          weights["ffn.w3.weight"])
        

    def forward(self, in_features: Float[Tensor, "... seq d_model"]
                    ,token_positions: Int[Tensor, "... seq"] = None):

        res = in_features

        in_features = self.attn_rmsnorm(in_features)
        if token_positions == None:
            seq_len = in_features.shape[-2]
            token_positions = torch.arange(seq_len, device=res.device,dtype=torch.long)
            token_positions = token_positions.unsqueeze(0)
        attn_out = self.attn(in_features, token_positions) + res

        res = attn_out

        in_features = self.mlp_rmsnorm(in_features)
        mlp_out = self.mlp(in_features)

        return mlp_out + res

class DecoderLayer(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float,
                layer_idx: int,
                weights: dict[str, Tensor]) -> None:
        super().__init__()

        self.attn_rmsnorm = RMSNorm(d_model, eps=1e-6, weights=weights[f"layers.{layer_idx}.ln1.weight"])
        self.mlp_rmsnorm = RMSNorm(d_model, eps=1e-6, weights=weights[f"layers.{layer_idx}.ln2.weight"])

        self.attn = MHA_With_RoPE(d_model,
                                  num_heads,
                                  max_seq_len,
                                  theta,
                                  weights[f"layers.{layer_idx}.attn.q_proj.weight"],
                                  weights[f"layers.{layer_idx}.attn.k_proj.weight"],  
                                  weights[f"layers.{layer_idx}.attn.v_proj.weight"],
                                  weights[f"layers.{layer_idx}.attn.o_proj.weight"])
        self.mlp = SwiGLU(d_model,
                          d_ff,
                          weights[f"layers.{layer_idx}.ffn.w1.weight"],
                          weights[f"layers.{layer_idx}.ffn.w2.weight"],
                          weights[f"layers.{layer_idx}.ffn.w3.weight"])
        

    def forward(self, in_features: Float[Tensor, "... seq d_model"]
                    ,token_positions: Int[Tensor, "... seq"] = None):

        res = in_features

        in_features = self.attn_rmsnorm(in_features)
        if token_positions == None:
            seq_len = in_features.shape[-2]
            token_positions = torch.arange(seq_len, device=res.device,dtype=torch.long)
            token_positions = token_positions.unsqueeze(0)
        attn_out = self.attn(in_features, token_positions) + res

        res = attn_out

        in_features = self.mlp_rmsnorm(in_features)
        mlp_out = self.mlp(in_features)

        return mlp_out + res

class LmHead(nn.Module):
    def __init__(self, weight:Float[Tensor, "vocab_size d_model"], vocab_size:int, d_model:int) -> None:
        super().__init__()
        self.weight = weight
    
    def forward(self, in_features:Float[Tensor, "... seq d_model"]):
        return einops.einsum(in_features, self.weight, "... d, v d -> ... v")


class Model(nn.Module):
    def __init__(self, vocab_size:int,
                    max_seq_len:int,
                    d_model: int,
                    num_layers: int,
                    num_heads: int,
                    d_ff: int,
                    rope_theta: float,
                    weights: dict[str, Tensor]) -> None:
        super().__init__()
        self.embed_layer = Embedding(vocab_size, 
                                     d_model,
                                     weight = weights["token_embeddings.weight"])
        self.num_layers = num_layers
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model,
                         num_heads,
                         d_ff,
                         max_seq_len,
                         rope_theta,
                         layer_idx=i,
                         weights=weights) for i in range(num_layers)
        ])

        self.lm_head = LmHead(weights["lm_head.weight"], vocab_size, d_model)
        self.ln = RMSNorm(d_model, eps=1e-6, weights=weights["ln_final.weight"])
    
    def forward(self, in_indices:Int[Tensor, "bsz seq"]):
        in_features = self.embed_layer(in_indices)

        for idx in range(self.num_layers):
            in_features = self.decoder[idx](in_features)
        
        in_features = self.ln(in_features)

        return self.lm(in_features)



        


    




        



