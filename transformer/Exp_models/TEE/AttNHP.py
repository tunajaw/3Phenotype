# Transformer Components Implementation Adapted from Annotated Transformer:
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
import math

import torch
from torch import nn


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # small change here -- we use "1" for masked element
        scores = scores.masked_fill(mask > 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_type_emb = d_model
        self.output_linear = output_linear

        if output_linear:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])
        #for i in range(len(self.linears)):
            #nn.init.xavier_uniform_(self.linears[i].weight)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            l(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            return self.linears[-1](x)
        else:
            return x


class SublayerConnection(nn.Module):
    # used for residual connnection
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_type_emb = d_model

    def forward(self, x, mask):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            return self.self_attn(x, x, x, mask)

class AttNHP(nn.Module):
    """Torch implementation of Attentive Neural Hawkes Process, ICLR 2022.
    https://arxiv.org/abs/2201.00044.
    Source code: https://github.com/yangalan123/anhp-andtt/blob/master/anhp/model/xfmr_nhp_fast.py
    """

    def __init__(self,
            n_marks,
            d_type_emb,

            time_enc, # default: "concat"
            d_time,

            d_inner,
            n_layers,
            n_head,
            dropout,

            device,

            diag_offset,
            
            use_norm):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(AttNHP, self).__init__()

        '''
        -- self.diag_offset = diag_offset
        ** self.n_marks = n_marks
        -- self.d_type_emb = d_type_emb

        -- self.time_enc = time_enc
        self.d_time = d_time  # will be set to 0 later if 'sum'

        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.dropout = dropout

        add: use_norm

        '''
        
        self.d_type_emb = d_type_emb
        self.use_norm = use_norm
        self.d_time = d_time # d_time

        self.div_term = torch.exp(torch.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time)).reshape(1, 1,
                                                                                                                -1)

        self.n_layers = n_layers
        self.n_head = n_head
        self.dropout = dropout
        self.device = device

        self.d_model = d_type_emb * self.n_head

        # event embedding
        self.layer_type_emb = nn.Linear(n_marks, d_type_emb, bias=True)

        self.heads = []
        for i in range(self.n_head):
            self.heads.append(
                nn.ModuleList(
                    [EncoderLayer(
                        self.d_type_emb + self.d_time,
                        MultiHeadAttention(1, self.d_type_emb + self.d_time, self.d_type_emb, self.dropout,
                                           output_linear=False),

                        use_residual=False,
                        dropout=self.dropout
                    )
                        for _ in range(self.n_layers)
                    ]
                )
            )
        self.heads = nn.ModuleList(self.heads)

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_type_emb)
        self.inten_linear = nn.Linear(self.d_type_emb * self.n_head, n_marks)
        self.softplus = nn.Softplus()
        self.layer_event_emb = nn.Linear(self.d_type_emb + self.d_time, self.d_type_emb)
        self.layer_intensity = nn.Sequential(self.inten_linear, self.softplus)
        self.eps = torch.finfo(torch.float32).eps

    def compute_temporal_embedding(self, time):
        """Compute the temporal embedding.

        Args:
            time (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, emb_size].
        """
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)

        return pe

    def forward_pass(self, init_cur_layer, time_emb, sample_time_emb, event_emb, combined_mask):
        """update the structure sequentially.

        Args:
            init_cur_layer (tensor): [batch_size, seq_len, hidden_size]
            time_emb (tensor): [batch_size, seq_len, hidden_size]
            sample_time_emb (tensor): [batch_size, seq_len, hidden_size]
            event_emb (tensor): [batch_size, seq_len, hidden_size]
            combined_mask (tensor): [batch_size, seq_len, hidden_size]

        Returns:
            tensor: [batch_size, seq_len, hidden_size*2]
        """
        cur_layers = []
        seq_len = event_emb.size(1)
        for head_i in range(self.n_head):
            # [batch_size, seq_len, hidden_size]
            cur_layer_ = init_cur_layer
            for layer_i in range(self.n_layers):
                # each layer concats the temporal emb
                # [batch_size, seq_len, hidden_size*2]
                layer_ = torch.cat([cur_layer_, sample_time_emb], dim=-1)
                # make combined input from event emb + layer emb
                # [batch_size, seq_len*2, hidden_size*2]
                _combined_input = torch.cat([event_emb, layer_], dim=1)
                enc_layer = self.heads[head_i][layer_i]
                # compute the output
                enc_output = enc_layer(_combined_input, combined_mask)

                # the layer output
                # [batch_size, seq_len, hidden_size]
                _cur_layer_ = enc_output[:, seq_len:, :]
                # add residual connection
                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_

                # event emb
                event_emb = torch.cat([enc_output[:, :seq_len, :], time_emb], dim=-1)

                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_

    def seq_encoding(self, time_seqs, event_seqs):
        """Encode the sequence.

        Args:
            time_seqs (tensor): time seqs input, [batch_size, seq_len].
            event_seqs (_type_): event type seqs input, [batch_size, seq_len].

        Returns:
            tuple: event embedding, time embedding and type embedding.
        """
        # [batch_size, seq_len, hidden_size]
        time_emb = self.compute_temporal_embedding(time_seqs)
        # [batch_size, seq_len, hidden_size]
        type_emb = torch.tanh(self.layer_type_emb(event_seqs.float()))
        # [batch_size, seq_len, hidden_size*2]
        event_emb = torch.cat([type_emb, time_emb], dim=-1)

        return event_emb, time_emb, type_emb

    def make_layer_mask(self, attention_mask):
        """Create a tensor to do masking on layers.

        Args:
            attention_mask (tensor): mask for attention operation, [batch_size, seq_len, seq_len]

        Returns:
            tensor: aim to keep the current layer, the same size of attention mask
            a diagonal matrix, [batch_size, seq_len, seq_len]
        """
        # [batch_size, seq_len, seq_len]
        layer_mask = (torch.eye(attention_mask.size(1)) < 1).unsqueeze(0).expand_as(attention_mask).to(attention_mask.device)
        return layer_mask

    def make_combined_att_mask(self, attention_mask, layer_mask):
        """Combined attention mask and layer mask.

        Args:
            attention_mask (tensor): mask for attention operation, [batch_size, seq_len, seq_len]
            layer_mask (tensor): mask for other layers, [batch_size, seq_len, seq_len]

        Returns:
            tensor: [batch_size, seq_len * 2, seq_len * 2]
        """
        # [batch_size, seq_len, seq_len * 2]
        combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # [batch_size, seq_len, seq_len * 2]
        contextual_mask = torch.cat([attention_mask, torch.ones_like(layer_mask)], dim=-1)
        # [batch_size, seq_len * 2, seq_len * 2]
        combined_mask = torch.cat([contextual_mask, combined_mask], dim=1)
        return combined_mask

    # self, event_type, event_time, non_pad_mask
    def forward(self, event_seqs, time_seqs, non_pad_mask, sample_times=None):
        """Call the model.

        Args:
            event_seqs (tensor): [batch_size, seq_len], sequences of event types.
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps.
            attention_mask (tensor): [batch_size, seq_len, seq_len], masks for event sequences.
            sample_times (tensor, optional): [batch_size, seq_len, num_samples]. Defaults to None.

        Returns:
            tensor: states at sampling times, [batch_size, seq_len, num_samples].
        """
        B, T, _ = non_pad_mask.shape
        attention_mask = torch.tile(torch.triu(torch.ones((T, T), dtype=bool))[None, :, :], (B, 1, 1),
                ).to(self.device)
        attention_mask = attention_mask * non_pad_mask

        event_emb, time_emb, type_emb = self.seq_encoding(time_seqs, event_seqs)
        init_cur_layer = torch.zeros_like(type_emb)
        layer_mask = self.make_layer_mask(attention_mask)
        if sample_times is None:
            sample_time_emb = time_emb
        else:
            sample_time_emb = self.compute_temporal_embedding(sample_times)
        combined_mask = self.make_combined_att_mask(attention_mask, layer_mask)
        cur_layer_ = self.forward_pass(init_cur_layer, time_emb, sample_time_emb, event_emb, combined_mask)

        return cur_layer_

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch
        # 1. compute event-loglik
        # the prediction of last event has no label, so we proceed to the last but one
        # att mask => diag is False, not mask.
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, 1:, :-1], time_seqs[:, 1:])
        # [batch_size, seq_len, num_event_types]
        lambda_at_event = self.layer_intensity(enc_out)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        temp_time = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # [batch_size, seq_len, num_sample]
        sample_times = temp_time + time_seqs[:, :-1].unsqueeze(-1)

        # 2.2 compute intensities at sampled times
        # [batch_size, seq_len = max_len - 1, num_sample, event_num]
        lambda_t_sample = self.compute_intensities_at_sample_times(time_seqs[:, :-1],
                                                                   time_delta_seqs[:, :-1],  # not used
                                                                   type_seqs[:, :-1],
                                                                   sample_times,
                                                                   attention_mask=attention_mask[:, 1:, :-1])

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        # return enc_inten to compute accuracy
        loss = - (event_ll - non_event_ll).sum()

        return loss, num_events

    def compute_states_at_sample_times(self,
                                       time_seqs,
                                       type_seqs,
                                       attention_mask,
                                       sample_times):
        """Compute the states at sampling times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps.
            time_delta_seqs (tensor): [batch_size, seq_len], sequences of delta times.
            type_seqs (tensor): [batch_size, seq_len], sequences of event types.
            attention_mask (tensor): [batch_size, seq_len, seq_len], masks for event sequences.
            sample_dtimes (tensor): delta times in sampling.

        Returns:
            tensor: hiddens states at sampling times.
        """
        batch_size = type_seqs.size(0)
        seq_len = type_seqs.size(1)
        num_samples = sample_times.size(-1)

        # [num_samples, batch_size, seq_len]
        sample_times = sample_times.permute((2, 0, 1))
        # [num_samples * batch_size, seq_len]
        _sample_time = sample_times.reshape(num_samples * batch_size, -1)
        # [num_samples * batch_size, seq_len]
        _types = type_seqs.expand(num_samples, -1, -1).reshape(num_samples * batch_size, -1)
        # [num_samples * batch_size, seq_len]
        _times = time_seqs.expand(num_samples, -1, -1).reshape(num_samples * batch_size, -1)
        # [num_samples * batch_size, seq_len]
        _attn_mask = attention_mask.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(num_samples * batch_size,
                                                                                         seq_len,
                                                                                         seq_len)
        # [num_samples * batch_size, seq_len, hidden_size]
        encoder_output = self.forward(_times,
                                      _types,
                                      _attn_mask,
                                      _sample_time)

        # [num_samples, batch_size, seq_len, hidden_size]
        encoder_output = encoder_output.reshape(num_samples, batch_size, seq_len, -1)
        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = encoder_output.permute((1, 2, 0, 3))
        return encoder_output

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_times, **kwargs):
        """Compute the intensity at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps.
            time_delta_seqs (tensor): [batch_size, seq_len], sequences of delta times.
            type_seqs (tensor): [batch_size, seq_len], sequences of event types.
            sampled_dtimes (tensor): [batch_size, seq_len, num_sample], sampled time delta sequence.

        Returns:
            tensor: intensities as sampled_dtimes, [batch_size, seq_len, num_samples, event_num].
        """
        attention_mask = kwargs.get('attention_mask', None)
        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool).to(time_seqs.device)

        if sample_times.size()[1] < time_seqs.size()[1]:
            # we pass sample_dtimes for last time step here
            # we do a temp solution
            # [batch_size, seq_len, num_samples]
            sample_times = time_seqs[:, :, None] + torch.tile(sample_times, [1, time_seqs.size()[1], 1])

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(time_seqs, type_seqs, attention_mask, sample_times)

        if compute_last_step_only:
            lambdas = self.layer_intensity(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.layer_intensity(encoder_output)
        return lambdas
    

