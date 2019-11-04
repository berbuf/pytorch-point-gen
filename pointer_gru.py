import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from seq2seq_pytorch.utils.cuda_helper import zeros, Tensor, LongTensor, cuda
from seq2seq_pytorch.utils.gru_helper import maskedSoftmax, generateMask
from seq2seq_pytorch.utils import Storage, SingleAttnGRU, zeros

class SingleAttnGRUPointerGen(SingleAttnGRU):
    def __init__(self, input_size, hidden_size, post_size,
                 coverage=False, initpara=True, gru_input_attn=False):
        super().__init__(input_size, hidden_size, post_size,
                         initpara=True, gru_input_attn=False)
        self.coverage = coverage
        self.pointer_gen = nn.Linear(post_size + input_size + hidden_size, 1)

    def forward(self, incoming, length, post, post_length, h_init=None):
        batch_size = incoming.shape[1]
        seqlen = incoming.shape[0]
        if h_init is None:
            h_init = self.getInitialParameter(batch_size)
        else:
            h_init = torch.unsqueeze(h_init, 0)
        h_now = h_init[0]
        hs, p_gen = [], []
        attn_weights, coverages = [], []
        context = zeros(batch_size, self.post_size)
        coverage = zeros(post.shape[0], batch_size)
        for i in range(seqlen):
            if self.gru_input_attn:
                h_now = self.cell_forward(torch.cat([incoming[i], context], last_dim=-1), h_now) \
                        * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)
            else:
                h_now = self.cell_forward(incoming[i], h_now) \
                        * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

            # decoder features
            query = self.attn_query(h_now)
            queried_post = (query.unsqueeze(0) * post).sum(-1)

            # add coverage
            if self.coverage:
                queried_post += coverage

            attn_weight = maskedSoftmax(queried_post, post_length)
            context = (attn_weight.unsqueeze(-1) * post).sum(0)
            coverage += attn_weight
            p_gen += [self.pointer_gen(torch.cat((context, incoming[i], h_now), dim=1)).sigmoid()]

            hs.append(torch.cat([h_now, context], dim=-1))
            attn_weights.append(torch.transpose(attn_weight, 0, 1))
            coverages.append(torch.transpose(coverage, 0, 1))

        return h_now, hs, attn_weights, p_gen, coverages

    def init_forward_3d(self, batch_size, top_k, post, post_length, h_init=None):
        if h_init is None:
            h_init = self.getInitialParameter(batch_size)
        else:
            h_init = torch.unsqueeze(h_init, 0)

        # batch_size * top_k * hidden_size
        h_now = h_init[0].unsqueeze(1).expand(-1, top_k, -1)
        context = zeros(batch_size, self.post_size)

        post = post.unsqueeze(-2)
        #post_length = np.tile(np.expand_dims(post_length, 1), (1, top_k, 1))

        def nextStep(incoming, stopmask, regroup=None):
            nonlocal h_now, post, post_length, context
            h_now = torch.gather(
                h_now, 1, regroup.unsqueeze(-1).repeat(1, 1, h_now.shape[-1]))

            if self.gru_input_attn:
                context = torch.gather(
                    context, 1, regroup.unsqueeze(-1).repeat(1, 1, context.shape[-1]))
                h_now = self.cell_forward(torch.cat([incoming, context], dim=-1), h_now) \
                    * (1 - stopmask).float().unsqueeze(-1)
            else:
                # batch_size * top_k * hidden_size                
                h_now = self.cell_forward(
                    incoming, h_now) * (1 - stopmask).float().unsqueeze(-1)

            # batch_size * top_k * post_size
            query = self.attn_query(h_now)
            queried_post = (query.unsqueeze(0) * post).sum(-1)

            mask = generateMask(post.shape[0], post_length).unsqueeze(-1)
            attn_weight = queried_post.masked_fill(mask == 0, -1e9).softmax(0)  # post_len * batch_size * top_k

            context = (attn_weight.unsqueeze(-1) * post).sum(0)

            p_gen = self.pointer_gen(torch.cat((context, incoming, h_now), dim=2)).sigmoid()

            return torch.cat([h_now, context], dim=-1), attn_weight, p_gen

        return nextStep

    def _freerun(self, inp, nextStep, wLinearLayerCallback, mode='max',
                 input_callback=None, no_unk=True, top_k=10):
        # inp contains: batch_size, dm, embLayer, max_sent_length, [init_h]
        # input_callback(i, embedding):   if you want to change word embedding at pos i, override this function
        # nextStep(embedding, flag):  pass embedding to RNN and get gru_h, flag indicates i th sentence is end when flag[i]==1
        # wLinearLayerCallback(gru_h): input gru_h and give a probability distribution on vocablist

        # output: w_o emb length

        start_id = inp.dm.go_id if no_unk else 0

        batch_size = inp.batch_size
        dm = inp.dm

        first_emb = inp.embLayer(LongTensor([dm.go_id])).repeat(batch_size, 1)

        gen = Storage()
        gen.w_pro = []
        gen.w_o = []
        gen.emb = []
        flag = zeros(batch_size).int()
        EOSmet = []

        next_emb = first_emb
        #nextStep = self.init_forward(batch_size, inp.get("init_h", None))

        for i in range(inp.max_sent_length):
            now = next_emb
            if input_callback:
                now = input_callback(i, now)

            gru_h = nextStep(now, flag)
            #if isinstance(gru_h, tuple):
            #    gru_h = gru_h[0]

            w = wLinearLayerCallback(gru_h, inp)
            gen.w_pro.append(w.softmax(dim=-1))
            # TODO: didn't consider copynet
            
            if mode == "max":
                w = torch.argmax(w[:, start_id:], dim=1) + start_id
                next_emb = inp.embLayer(w)
            elif mode == "gumbel" or mode == "sample":
                w_onehot = gumbel_max(w[:, start_id:])
                w = torch.argmax(w_onehot, dim=1) + start_id
                next_emb = torch.sum(torch.unsqueeze(
                    w_onehot, -1) * inp.embLayer.weight[start_id:], 1)
            elif mode == "samplek":
                _, index = w[:, start_id:].topk(
                    top_k, dim=-1, largest=True, sorted=True)  # batch_size, top_k

                mask = torch.zeros_like(
                    w[:, start_id:]).scatter_(-1, index, 1.0)
                w_onehot = gumbel_max_with_mask(w[:, start_id:], mask)
                w = torch.argmax(w_onehot, dim=1) + start_id
                next_emb = torch.sum(torch.unsqueeze(
                    w_onehot, -1) * inp.embLayer.weight[start_id:], 1)

            gen.w_o.append(w)
            gen.emb.append(next_emb)

            EOSmet.append(flag)
            flag = flag | (w == dm.eos_id).int()
            if torch.sum(flag).detach().cpu().numpy() == batch_size:
                break

        EOSmet = 1-torch.stack(EOSmet)
        gen.w_o = torch.stack(gen.w_o) * EOSmet.long()
        gen.emb = torch.stack(gen.emb) * EOSmet.float().unsqueeze(-1)
        gen.length = torch.sum(EOSmet, 0).detach().cpu().numpy()

        return gen

    def _beamsearch(self, inp, top_k, nextStep, wLinearLayerCallback, input_callback=None, no_unk=True, length_penalty=0.7):
        # inp contains: batch_size, dm, embLayer, max_sent_length, [init_h]
        # input_callback(i, embedding):   if you want to change word embedding at pos i, override this function
        # nextStep(embedding, flag):  pass embedding to RNN and get gru_h, flag indicates i th sentence is end when flag[i]==1
        # wLinearLayerCallback(gru_h): input gru_h and give logits on vocablist

        # output: w_o emb length

        #start_id = inp.dm.go_id if no_unk else 0

        batch_size = inp.batch_size
        dm = inp.dm
        first_emb = inp.embLayer(LongTensor(
            [dm.go_id])).repeat(batch_size, top_k, 1)
        w_pro = []
        w_o = []
        emb = []
        flag = zeros(batch_size, top_k).int()
        EOSmet = []
        score = zeros(batch_size, top_k)
        score[:, 1:] = -1e9
        now_length = zeros(batch_size, top_k)
        back_index = []
        regroup = LongTensor([i for i in range(top_k)]).repeat(batch_size, 1)

        next_emb = first_emb
        #nextStep = self.init_forward(batch_size, inp.get("init_h", None))

        for i in range(inp.max_sent_length):
            now = next_emb
            if input_callback:
                now = input_callback(i, now)

            # batch_size, top_k, hidden_size
            
            gru_h = nextStep(now, flag, regroup=regroup)
            w = wLinearLayerCallback(gru_h, inp)  # batch_size, top_k, vocab_size
            
            if no_unk:
                w[:, :, dm.unk_id] = -1e9
            w = w.log_softmax(dim=-1)
            w_pro.append(w.exp())

            new_score = (score.unsqueeze(-1) + w * (1-flag.float()).unsqueeze(-1)) / \
                ((now_length.float() + 1 - flag.float()).unsqueeze(-1) ** length_penalty)
            new_score[:, :, 1:] = new_score[:, :, 1:] - \
                flag.float().unsqueeze(-1) * 1e9
            _, index = new_score.reshape(
                batch_size, -1).topk(top_k, dim=-1, largest=True, sorted=True)  # batch_size, top_k
            

            new_score = (score.unsqueeze(-1) + w * (1-flag.float()
                                                    ).unsqueeze(-1)).reshape(batch_size, -1)
                                                                                                                             
            score = torch.gather(new_score, dim=1, index=index)

            vocab_size = w.shape[-1]
            regroup = index / vocab_size  # batch_size, top_k
            
            back_index.append(regroup)
            w = torch.fmod(index, vocab_size)  # batch_size, top_k
                                                                                                                             
            flag = torch.gather(flag, dim=1, index=regroup)
                                                                                                                             
            now_length = torch.gather(
                now_length, dim=1, index=regroup) + 1 - flag.float()

            w_x = w.clone()
            w_x[w_x >= dm.vocab_size] = dm.unk_id

            next_emb = inp.embLayer(w_x)
            w_o.append(w)
            emb.append(next_emb)

            EOSmet.append(flag)

            flag = flag | (w == dm.eos_id).int()
            if torch.sum(flag).detach().cpu().numpy() == batch_size * top_k:
                break

        # back tracking
        gen = Storage()
        back_EOSmet = []
        gen.w_o = []
        gen.emb = []
        now_index = LongTensor([i for i in range(top_k)]).repeat(batch_size, 1)

        for i, index in reversed(list(enumerate(back_index))):
            gen.w_o.append(torch.gather(w_o[i], dim=1, index=now_index))
            gen.emb.append(torch.gather(
                emb[i], dim=1, index=now_index.unsqueeze(-1).expand_as(emb[i])))
            back_EOSmet.append(torch.gather(EOSmet[i], dim=1, index=now_index))
            now_index = torch.gather(index, dim=1, index=now_index)

        back_EOSmet = 1-torch.stack(list(reversed(back_EOSmet)))
        gen.w_o = torch.stack(list(reversed(gen.w_o))) * back_EOSmet.long()
        gen.emb = torch.stack(list(reversed(gen.emb))) * \
            back_EOSmet.float().unsqueeze(-1)
        gen.length = torch.sum(back_EOSmet, 0).detach().cpu().numpy()

        return gen
