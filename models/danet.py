import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
from .module import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNeXt_Block(nn.Module):
    def __init__(self, dim, r, kernel_size, dw=True):
        super().__init__()
        if dw:
            groups = dim
        else:
            groups = 1

        self.layer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=groups, bias=False),
            LayerNorm2D(dim),
            nn.Conv2d(dim, dim*r, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim*r, dim, 1, 1, 0)
        )
        self.ln = LayerNorm2D(dim)

    def forward(self, x):
        x = self.layer(x) + x
        x = self.ln(x)
        return x


class MFA(nn.Module):
    def __init__(self, dim, depth, r):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvNeXt_Block(dim, r, kernel_size=2*(i_layer+1)+1, dw=True)
            for i_layer in range(depth)
        ])

    def forward(self, ds):

        out = None

        for i, d in enumerate(ds):

            d = rearrange(d, 'b (h w) d -> b d h w', b=d.shape[0], w=14, h=14).contiguous()

            if out is None:
                out = self.layers[i](d)
            else:
                out = self.layers[i](out + d)

        out = rearrange(out, 'b d h w -> b (h w) d', b=out.shape[0], w=14, h=14).contiguous()
        return out


class DDWC_Mul_Add(nn.Module): 
    def __init__(self,
                 dim,
                 kernel_size,
                 reduction_ratio,
                 num_groups):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."

        self.num_groups = num_groups
        self.K = kernel_size
        
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim//reduction_ratio, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim//reduction_ratio, dim*num_groups*2, 1, 1, 0),
        )

        self.weight = nn.Parameter(torch.zeros(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.beta_scale = nn.Parameter(torch.zeros(1, num_groups, dim, 1, 1), requires_grad=True) 

        self.pwconv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, s, r):
        B, C, H, W = s.shape

        scale = self.proj(F.adaptive_avg_pool2d(r, (1, 1))).reshape(B, self.num_groups, C, 2, 1, 1) # [B, G, C, 2, 1, 1]
        
        alpha = F.softmax(scale[:, :, :, 0], dim=1) # [B, G, C, 1, 1]

        beta = F.tanh(scale[:, :, :, 1] * self.beta_scale.exp() * 0.1) # a3

        # [B, G, C, 1, 1] * [1, G, C, K, K] + [B, G, C, 1, 1] -> [B, G, C, K, K]
        weights = alpha * self.weight.unsqueeze(dim=0) + beta 

        if self.training:
            out = None
            s = s.reshape(1, -1, H, W) # [1, B*C, H, W]
            for i in range(self.num_groups):
                weight = weights[:, i].reshape(-1, 1, self.K, self.K) # [B*C, 1, K, K]
                z = F.conv2d(
                    s,
                    weight=weight,
                    padding=self.K//2,
                    groups=B*C,
                    bias=None
                )
                
                if out is None:
                    out = z
                else:
                    out = out + z
        else:
            weight = torch.sum(weights, dim=1, keepdim=False) # [B, C, K, K]
            weight = weight.reshape(-1, 1, self.K, self.K)
            out = F.conv2d(
                s.reshape(1, -1, H, W),
                weight=weight,
                padding=self.K//2,
                groups=B*C,
                bias=None
            )

        out = F.gelu(out)
        out = out.reshape(B, C, H, W)
        out = self.pwconv(out)
        return out


class Conv2d(nn.Module):
    def __init__(self, in_dim, dim, kernel_size, dw):
        super().__init__()
        if dw:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size, 1, kernel_size//2, groups=in_dim, bias=False), 
                nn.Conv2d(in_dim, dim, 1, 1, 0), 
            )
        else:
            self.conv = nn.Conv2d(in_dim, dim, k, 1, kernel_size//2, groups=1, bias=True)

    def forward(self, x):
        
        if x.dim() == 3:
            x = rearrange(x, 'b (h w) d -> b d h w', b=x.shape[0], h=14, w=14).contiguous()
            flag = True
        else:
            flag = False

        z = self.conv(x)

        if flag:
            z = rearrange(z, 'b d h w -> b (h w) d', b=z.shape[0], h=14, w=14).contiguous()

        return z


class DGDC_Layer(nn.Module): 
    def __init__(self, dim, mlp_ratio, dropout, i_layer):
        super().__init__()
        self.unit = DDWC_Mul_Add(dim, kernel_size=2*(i_layer+1)+1, reduction_ratio=1, num_groups=8)
        self.ffn = FFN(dim=dim, r=mlp_ratio, dropout=dropout, act=nn.GELU())
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, y):
        x = rearrange(x, 'b (h w) d -> b d h w', b=x.shape[0], h=14, w=14).contiguous()
        y = rearrange(y, 'b (h w) d -> b d h w', b=y.shape[0], h=14, w=14).contiguous()    

        #### DGDC
        dx = x - y
        dy = y - x
        x = self.unit(x, dx) + x
        y = self.unit(y, dy) + y

        x = rearrange(x, 'b d h w -> b (h w) d', b=x.shape[0], w=14, h=14).contiguous()
        y = rearrange(y, 'b d h w -> b (h w) d', b=y.shape[0], w=14, h=14).contiguous()
        x = self.norm1(x)  
        y = self.norm1(y)  

        x = self.norm2(x + self.ffn(x))
        y = self.norm2(y + self.ffn(y))      
        return x, y


class DFS(nn.Module):
    def __init__(self, dim, num_groups):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim*num_groups, dim)) 
        self.fcs = nn.ModuleList([])
        for i in range(num_groups):
            self.fcs.append(nn.Linear(dim, dim))

    def forward(self, feas, i_layer):
        """
        feas [B, N, G, D]
        """
        fea_z = feas.mean(dim=1) # [B, 3, D]
        fea_z = fea_z.reshape(fea_z.shape[0], -1) # [B, 3*D]
        fea_z = self.fc(fea_z) # [B, D]
        attention_vectors = []
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z) # [B, D]
            attention_vectors.append(vector)
        attention_vectors = torch.stack(attention_vectors, dim=1) # [B, 3, D]
        # attention_vectors = attention_vectors * math.sqrt(i_layer+1) # only for fine-tuning
        attention_vectors = F.softmax(attention_vectors, dim=1) # [B, 3, D]
        fea_v = torch.sum(feas * attention_vectors.unsqueeze(dim=1), dim=2) # [B, N, 3, D] * [B, 1, 3, D] => [B, N, D]
        return fea_v, attention_vectors


class METI_Layer(nn.Module):  
    def __init__(self, dim, mlp_ratio, dropout, i_layer):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Linear(dim, dim*2),
            nn.LayerNorm(dim*2)
        )
        
        self.unit1 = MHA(dim, dim//64, dropout)
        self.unit2 = Conv2d(dim, dim, 2*(i_layer+1)+1, dw=True)
        self.dfs = DFS(dim, num_groups=3)
        self.ffn = FFN(dim=dim, r=mlp_ratio, dropout=dropout, act=nn.GELU())
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)        

    def forward(self, s, r, i_layer):
        shortcut = s
        z1 = self.unit1(s, r) # GIE 
        h = self.fc(torch.cat([s, r], dim=-1))
        h1, h2 = h.chunk(2, dim=-1)         
        z2 = self.unit2(h1) # RIE
        z3 = h2 # PIE
        z, path = self.dfs(torch.stack([z1, z2, z3], dim=2), i_layer) # [B, N, G, D]
        z = self.norm1(z + shortcut)
        z = self.norm2(z + self.ffn(z))    
        return z, path


class Encoder(nn.Module):
    def __init__(self, depth, ori_dim, dim, dropout):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        self.projection = nn.Sequential(nn.Linear(ori_dim, dim))

        r = 2

        # DGDC & FFN
        self.dgdc_layers = nn.ModuleList([
            DGDC_Layer(dim, mlp_ratio=r, dropout=dropout, i_layer=_)
            for _ in range(depth)
        ])

        # METI & FFN
        self.meti_layers = nn.ModuleList([
            METI_Layer(dim, mlp_ratio=r, dropout=dropout, i_layer=_)  
            for _ in range(depth)
        ])

        # MFA
        self.mfa = MFA(dim, depth, r=r)

    def forward(self, x, y):
        b, _, h, w = x.shape
        x = rearrange(x, 'b d h w -> b (h w) d', b=b, h=h, w=w).contiguous()
        y = rearrange(y, 'b d h w -> b (h w) d', b=b, h=h, w=w).contiguous()
        x = self.projection(x)
        y = self.projection(y)

        paths = []
        ds = []
        for i in range(self.depth):
            x, y = self.dgdc_layers[i](x, y)
            y, path = self.meti_layers[i](s=y, r=x, i_layer=i)
            paths.append(path)
            ds.append(y)

        out = self.mfa(ds)
        return out


class Changer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = Extractor('resnet101')
        ori_dim = 1024
        
        self.encoder = Encoder(args.n_enc, ori_dim, args.dim, args.enc_dropout)

        self.decoder = TransformerDecoder(
            vocab_size=len(args.vocab), 
            max_len=args.max_len,
            depth=args.n_dec, 
            dim=args.dim, 
            mlp_ratio=4,
            dropout=args.dec_dropout,
        )

    def visual_encoding(self, x, y):
        feats1, feats2 = self.backbone(x, y)
        feats = self.encoder(feats1, feats2) 
        return feats   

    def forward(self, x, y, seq):
        memory = self.visual_encoding(x, y)
        max_length = seq.size(1)
        seq_mask = torch.tril(torch.ones((1, 1, max_length, max_length), dtype=torch.int32, device=device), diagonal=0) # [1, 1, L, L]
        logits = self.decoder(seq, memory, seq_mask, None)

        scores = logits[:, :-1].reshape(-1, logits.shape[-1]).contiguous()
        targets = seq[:, 1:].reshape(-1).contiguous()
        loss = F.cross_entropy(scores, targets, ignore_index=0)   
        return loss

    def get_logprobs_state(self, wt, state, memory, memory_mask):
        if state is None:
            ys = wt
        else:
            ys = torch.cat([state, wt], dim=1) # [B, T]

        logits, attn_weights = self.decoder(wt, memory, None, memory_mask, return_attn_weights=True)

        return ys, logits, attn_weights

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s
        return fn

    def select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def decode_beam(self, image1, image2, beam_size, return_all=False):
        start_idx, end_idx, max_len = self.args.vocab['<start>'], self.args.vocab['<end>'], self.args.max_len
        X = image1
        Y = image2
        batch_size = X.size(0)

        memory = self.visual_encoding(X, Y)

        seq_logprob = torch.zeros((batch_size, 1, 1), device=device)
        log_probs = []
        seq_mask = torch.ones((batch_size, beam_size, 1), device=device)
        state = None
        wt = torch.empty((batch_size), dtype=torch.int32, device=device).fill_(start_idx)
        outputs = []

        self.decoder.init_buffer()

        for t in range(max_len):
            cur_beam_size = 1 if t == 0 else beam_size

            state, logits, attn_weights = self.get_logprobs_state(wt.unsqueeze(dim=-1), state, memory, None)
            logits = logits.squeeze(dim=1)
            word_logprob = F.log_softmax(logits, dim=-1)
         
            # [b*cur_beam_size, vocab_size] --> [b, cur_beam_size, vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)

            # sum of logprob
            # [b, cur_beam_size, vocab_size]
            candidate_logprob = seq_logprob + word_logprob # t=0: [b, 1, vocab_size] t>0: [b, beam_size, vocab_size]

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != end_idx).float().unsqueeze(-1) 
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)
            
            # [b, beam_size], [b, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='trunc')
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # update buffer
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))

            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
 
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                state = state.repeat_interleave(beam_size, dim=0)

            if seq_mask.sum() == 0:
                break

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1) # [B, Beam, L]
        log_probs = torch.cat(log_probs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, outputs.shape[-1])) # [B, Beam, L]
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, log_probs.shape[-1]))

        self.decoder.clear_buffer()

        if not return_all:
            outputs = outputs.contiguous()[:, 0] # [B, L]
            log_probs = log_probs.contiguous()[:, 0] # [B, L]            
            return outputs, log_probs
        else:
            outputs = outputs.contiguous() # [B, Beam, L]
            log_probs = log_probs.contiguous() # [B, Beam, L]                 
            return outputs, log_probs
    
    def set_finetune(self, finetune):
        for n, p in self.backbone.named_parameters():   
            p.requires_grad = finetune

    def set_finetune2(self, finetune):
        for n, p in self.backbone.named_parameters():   
            p.requires_grad = finetune

        for n, p in self.encoder.named_parameters():   
            if 'dfs' in n:
                p.requires_grad = True
            else:
                p.requires_grad = finetune
        
        for n, p in self.decoder.named_parameters():
            p.requires_grad = finetune

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for n, module in self.named_children():
            if 'backbone' in n:
                module.eval()
            else:
                module.train(mode)
        return self