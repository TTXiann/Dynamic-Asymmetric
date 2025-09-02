import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Extractor(nn.Module):
    def __init__(self, name, model_stage=3):
        super().__init__()

        if name == 'resnet18':
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT

        elif name == 'resnet50':
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT
        
        elif name == 'resnet101':
            from torchvision.models import ResNet101_Weights
            weights = ResNet101_Weights.DEFAULT

        cnn = getattr(torchvision.models, name)(weights=weights)

        if 'resnet' in name:
            layers = [
                cnn.conv1,
                cnn.bn1,
                cnn.relu,
                cnn.maxpool,
            ]
            for i in range(model_stage):
                name = 'layer%d' % (i + 1)
                layers.append(getattr(cnn, name))

        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.net(x)   
        y = self.net(y)      
        return x, y


class FFN(nn.Module):
    def __init__(self, dim, r, dropout, act):
        super().__init__()
        hidden_dim = int(dim * r)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):

        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous() # [B, C, H, W] -> [B, H, W, C]
            flag = True
        else:
            flag = False

        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x))

        if flag:
            x = x.permute(0, 3, 1, 2).contiguous() # [B, H, W, C] -> [B, C, H, W]

        return x


class MHA(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, q, kv, mask=None, return_attn=False):
        # mask [B, 1, 1, N]
        
        bs, nq = q.shape[:2]
        nkv = kv.shape[1]

        q = rearrange(self.q_proj(q), 'b n (h c)-> b h n c', b=bs, n=nq, h=self.num_heads, c=self.head_dim)
        k = rearrange(self.k_proj(kv), 'b n (h c)-> b h c n', b=bs, n=nkv, h=self.num_heads, c=self.head_dim)
        v = rearrange(self.v_proj(kv), 'b n (h c)-> b h n c', b=bs, n=nkv, h=self.num_heads, c=self.head_dim)

        attn = (q @ k) * self.scale # [B, H, N_q, N_k] 

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -torch.inf)
            attn = F.softmax(attn, dim=-1)
        else:       
            attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', b=bs, n=nq, h=self.num_heads, c=self.head_dim)
        out = self.o_proj(out) 
        out = self.dropout(out)
        if return_attn:
            return out, attn
        else:
            return out


class SelfMHA(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def init_buffer(self):
        self.time_step = 0 # ar

    def clear_buffer(self):
        self.buffer_key = None
        self.buffer_value = None
        self.time_step = None

    def repeat(self, n):
        self.buffer_key = self.buffer_key.repeat_interleave(n, dim=0)
        self.buffer_value = self.buffer_value.repeat_interleave(n, dim=0)

    def apply_to_states(self, fn):
        self.buffer_key = fn(self.buffer_key)
        self.buffer_value = fn(self.buffer_value)

    def forward(self, q, k, v, mask=None):
        bs, nq = q.shape[:2]
        _, nk = k.shape[:2]

        q = rearrange(self.q_proj(q), 'b n (h c) -> b h n c', b=bs, n=nq, h=self.num_heads, c=self.head_dim).contiguous()
        k = rearrange(self.k_proj(k), 'b n (h c) -> b h c n', b=bs, n=nk, h=self.num_heads, c=self.head_dim).contiguous()
        v = rearrange(self.v_proj(v), 'b n (h c) -> b h n c', b=bs, n=nk, h=self.num_heads, c=self.head_dim).contiguous()

        if self.time_step is not None: # 【AR self】
            if self.time_step == 0: 
                self.buffer_key = k
                self.buffer_value = v
            else:
                self.buffer_key = torch.cat([self.buffer_key, k], dim=-1)
                self.buffer_value = torch.cat([self.buffer_value, v], dim=2)

                k = self.buffer_key
                v = self.buffer_value 

            self.time_step += 1

        attn = (q @ k) * self.scale # [B, H, N_q, N_k] 

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -torch.inf)
            attn = F.softmax(attn, dim=-1)
        else:       
            attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', b=bs, n=nq, h=self.num_heads, c=self.head_dim).contiguous()
        out = self.o_proj(out)
        out = self.dropout(out)
        return out
 
 
class CrossMHA(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.time_step = None

    def init_buffer(self):
        self.time_step = 0 # ar
        self.buffer_key = None
        self.buffer_value = None

    def clear_buffer(self):
        self.time_step = None
        self.buffer_key = None
        self.buffer_value = None

    def forward(self, q, k, v, mask=None):
        bs, nq = q.shape[:2]
        nk = k.shape[1]

        q = rearrange(self.q_proj(q), 'b n (h c)-> b h n c', b=bs, n=nq, h=self.num_heads, c=self.head_dim).contiguous() 
        
        if self.time_step is not None: # 【AR cross】
           
            if self.time_step == 0:
                k = rearrange(self.k_proj(k), 'b n (h c)-> b h c n', b=bs, n=nk, h=self.num_heads, c=self.head_dim).contiguous()
                v = rearrange(self.v_proj(v), 'b n (h c)-> b h n c', b=bs, n=nk, h=self.num_heads, c=self.head_dim).contiguous()
                self.buffer_key = k
                self.buffer_value = v
            
            elif self.time_step == 1:
                if bs != self.buffer_key.shape[0]: # beam search
                    self.buffer_key = self.buffer_key.repeat_interleave(bs//self.buffer_key.shape[0], dim=0)
                    self.buffer_value = self.buffer_value.repeat_interleave(bs//self.buffer_value.shape[0], dim=0)
            
            k = self.buffer_key
            v = self.buffer_value
            self.time_step += 1

        else: # 【XE cross】
            k = rearrange(self.k_proj(k), 'b n (h c)-> b h c n', b=bs, n=nk, h=self.num_heads, c=self.head_dim)
            v = rearrange(self.v_proj(v), 'b n (h c)-> b h n c', b=bs, n=nk, h=self.num_heads, c=self.head_dim)

        attn = (q @ k) * self.scale # [B, H, N, M] 

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -torch.inf)
            attn = F.softmax(attn, dim=-1)
        else:       
            attn = F.softmax(attn, dim=-1)

        attn_ = self.dropout(attn)
        out = rearrange(attn_ @ v, 'b h n c -> b n (h c)', b=bs, n=nq, h=self.num_heads, c=self.head_dim).contiguous()
        out = self.o_proj(out)
        out = self.dropout(out)
        return out, attn


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, depth, dim, mlp_ratio, dropout, act=nn.GELU()):
        super().__init__()
        self.depth =  depth
        self.dim = dim
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            sublayer = TransformerDecoderLayer(dim, mlp_ratio, dropout, act)           
            self.layers.append(sublayer)
        
        self.word_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.generator = nn.Linear(dim, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.clear_buffer()

    def init_buffer(self):
        self.seq_len = 0
        for i, layer in enumerate(self.layers):
            layer.init_buffer()

    def clear_buffer(self):
        self.seq_len = None
        for i, layer in enumerate(self.layers):
            layer.clear_buffer()

    def repeat(self, n):
        for i, layer in enumerate(self.layers):
            layer.repeat(n)        

    def apply_to_states(self, fn):
        for i, layer in enumerate(self.layers):
            layer.apply_to_states(fn)

    def forward(self, seq, memory, seq_mask, memory_mask, return_attn_weights=False):    
        # seq [B, L]
        # seq_mask [1, 1, L, L]
        # memory [B, R, D]
        # memory_mask [1, 1, 1, R] / [1, 1, R, R]
        
        seq_len = seq.size(1)
        if self.seq_len is not None:
            pos_indx = torch.arange(self.seq_len, self.seq_len + 1, device=seq.device).view(1, -1)
            self.seq_len += 1
        else:
            pos_indx = torch.arange(0, seq_len, device=seq.device).view(1, -1) # [B, L]

        x = self.word_embed(seq) + self.pos_embed(pos_indx)
        x = self.dropout(x) # [B, L, D]

        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, memory, seq_mask, memory_mask)

        x = self.dropout(x)
        x = self.generator(x)
      
        if return_attn_weights:
            return x, attn_weights
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout, act):
        super().__init__()
        self.attn = SelfMHA(dim, dim//64, dropout)
        self.cross_attn = CrossMHA(dim, dim//64, dropout)
        self.ffn = FFN(dim, mlp_ratio, dropout, act)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def init_buffer(self):
        self.attn.init_buffer()
        self.cross_attn.init_buffer()

    def clear_buffer(self):
        self.attn.clear_buffer()
        self.cross_attn.clear_buffer()

    def repeat(self, n):
        self.attn.repeat(n)   
    
    def apply_to_states(self, fn):
        self.attn.apply_to_states(fn)

    def forward(self, x, memory, x_mask, memory_mask):
        
        # 【self】
        shortcut = x
        x = self.attn(x, x, x, x_mask)
        x = self.norm1(x + shortcut)

        # 【cross】
        shortcut = x
        x, attn_weights = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + shortcut)

        # 【ffn】
        shortcut = x
        x = self.ffn(x)
        x = self.norm3(x + shortcut)         

        return x, attn_weights
    

class LayerNorm2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous() # [B, C, H, W] -> [B, H, W, C]
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() # [B, H, W, C] -> [B, C, H, W]
        return x
