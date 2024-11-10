import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    def attention(self, query, key, value, mask=None, att_mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        # print(f"{query.shape}, {key.shape}, {value.shape}, {mask.shape}")
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if att_mask is not None:
            # print(f"scores: {scores.shape}, att_mask:{att_mask.unsqueeze(1).unsqueeze(-1).shape}")
            scores = scores * att_mask.unsqueeze(1).unsqueeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, att_mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        # print(f"before: {query.shape}, {key.shape}")
        # print(f"{self.linears[0](query).shape}, {self.linears[1](key).shape}")
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        # print(f"after: {query.shape}, {key.shape}")
        x, _ = self.attention(query, key, value, mask, att_mask, dropout)
        # print(f"x0: {x.shape}")
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        # print(f"x1: {x.shape}")
        # print(f"x2: {self.linears[-1](x).shape}")
        # assert 0
        return self.linears[-1](x)


class mTAN_Enc(nn.Module):
 
    def __init__(self, input_dim, w, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
        super(mTAN_Enc, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.w = w # query = query.to(self.device)
        self.att = multiTimeAttention(input_dim, nhidden, embed_time, num_heads)
        # self.classifier = nn.Sequential(
        #     nn.Linear(nhidden, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 2))
        self.enc = nn.GRU(nhidden, nhidden)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    def transform_to_3d_ts(self, time_steps, val, mark, non_pad_mask, device='cpu'):
        """
        Given 2D time series with their time steps, values, mask, and corresponding mark (the dimension of the time series),
        transform them into 3D time series.
        """
        # Get dynamic dimensions
        B, L = time_steps.shape  # Batch size and time series length
        K = self.dim             # Maximum dimension (assuming mark is in range [1, K])
        non_pad_mask = non_pad_mask[:,:,0]

        # Create indices for batch and time dimensions
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(-1, L)
        time_indices = torch.arange(L, device=device).view(1, -1).expand(B, -1)

        # Convert mark to zero-based indexing for dimensions
        k_indices = mark - 1

        # Create output tensors initialized to zero
        time_steps_blk = torch.zeros((B, L, K), dtype=time_steps.dtype, device=device)
        val_blk = torch.zeros((B, L, K), dtype=val.dtype, device=device)
        non_pad_mask_blk = torch.zeros((B, L, K), dtype=non_pad_mask.dtype, device=device)

        # Use advanced indexing to place values where non_pad_mask is true
        mask_indices = non_pad_mask.bool()

        time_steps_blk[batch_indices[mask_indices], time_indices[mask_indices], k_indices[mask_indices]] = time_steps[mask_indices]
        val_blk[batch_indices[mask_indices], time_indices[mask_indices], k_indices[mask_indices]] = val[mask_indices]
        non_pad_mask_blk[batch_indices[mask_indices], time_indices[mask_indices], k_indices[mask_indices]] = non_pad_mask[mask_indices]

        return time_steps_blk, val_blk, non_pad_mask_blk
    
    # def transform_to_3d_ts(self, time_steps, val, mark, non_pad_mask, device='cpu'):
    #     """ 
    #     Given 2d time series with their time steps, values, mask and corresponding mark (the dimension of the time series), 
    #     transform them into 3d time series.
    #     """
    #     # Get dynamic dimensions
    #     B, L = time_steps.shape  # Batch size and time series length
    #     K = self.dim    # Maximum dimension (assuming mark is in range [1, K])

    #     # Initialize output tensors with shape [B, L, K] on the specified device
    #     time_steps_blk = torch.zeros((B, L, K), dtype=time_steps.dtype, device=device)
    #     val_blk = torch.zeros((B, L, K), dtype=val.dtype, device=device)
    #     non_pad_mask_blk = torch.zeros((B, L, K), dtype=non_pad_mask.dtype, device=device)
        
    #     # Iterate through each batch (B) and time step (L)
    #     for b in range(B):
    #         for l in range(L):
    #             if non_pad_mask[b, l] == 1:  # Only process non-padding values
    #                 k = mark[b, l] - 1  # Convert mark to zero-based index for dimensions
    #                 # Fill in the transformed tensors
    #                 time_steps_blk[b, l, k] = time_steps[b, l]
    #                 val_blk[b, l, k] = val[b, l]
    #                 non_pad_mask_blk[b, l, k] = non_pad_mask[b, l]
    
    #     return time_steps_blk, val_blk, non_pad_mask_blk
    
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = 48.*pos.unsqueeze(2).to(self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model)).to(self.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_subsequent_mask(self, seq, diag_offset=1):
        """ For masking out the subsequent info, i.e., masked self-attention. """
        if seq.dim() == 2:
            sz_b, len_s = seq.size()
        else:
            sz_b, len_s, dim = seq.size()
        subsequent_mask = torch.tril(
            torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=diag_offset)
        subsequent_mask = subsequent_mask.unsqueeze(
            0).expand(sz_b, -1, -1)  # b x ls x ls
        return subsequent_mask
    
    # def get_att_mask(self, qt, kt):
    #     '''
    #     generate attention mask to avoid seeing future events at reference points t.

    #     inputs:
    #         qt: reference points, shape (R, )
    #         kt: time steps, shape (B, L)

    #     output: 
    #         tril matrix, shape (B, R, L)
    #     '''
    #     qt = qt.unsqueeze(0)  # Shape: [nq, 1]
    #     return (qt.unsqueeze(2) >= kt.unsqueeze(1)).int()  # Shape: [B, nq, nk]
    
       
    def forward(self, data, mask, verbose=None):
        '''
        data: [time, val, mark]
        mask: non_pad_mask
        '''
        # ignore demo data here
        time_steps, val, mark, _ = data
        # transform mark into corresponding dimension
        _, val, mask = self.transform_to_3d_ts(time_steps, val, mark, mask, self.device)

        # print(f"original:{self.query.shape}, {time_steps.shape}")
        # time_steps = time_steps.cpu()
        # mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)

        # att_mask = self.get_att_mask(self.query, time_steps)
        att_mask = self.get_subsequent_mask(time_steps, self.w)
        
        out = self.att(key, key, val, mask, att_mask)
        out = out.permute(1, 0, 2)
        out, _ = self.enc(out) # shape: [L, B, hidden]
        out = out.permute(1, 0, 2)
        return out #[0, :, :] # shape: [B, L, hidden]