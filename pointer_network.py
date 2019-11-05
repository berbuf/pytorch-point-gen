import logging

import torch
from torch import nn

from seq2seq_pytorch.utils import cuda, zeros, LongTensor, BaseNetwork, MyGRU, Storage, gumbel_max, flattenSequence, SingleAttnGRU, SequenceBatchNorm
from seq2seq_pytorch.network import Network, EmbeddingLayer, PostEncoder, ConnectLayer

from pointer_gru import SingleAttnGRUPointerGen


class PointerNetwork(BaseNetwork):
    def __init__(self, param):
        super().__init__(param)
        self.embLayer = PointerEmbeddingLayer(param)
        self.postEncoder = PostEncoder(param)
        self.connectLayer = ConnectLayer(param)
        self.genNetwork = PointerGenNetwork(param)

    def forward(self, incoming):
        incoming.result = Storage()
        self.embLayer.forward(incoming)
        self.postEncoder.forward(incoming)
        self.connectLayer.forward(incoming)
        self.genNetwork.forward(incoming)
        incoming.result.loss = incoming.result.word_loss
        if torch.isnan(incoming.result.loss).detach().cpu().numpy() > 0:
            logging.info("Nan detected")
            logging.info(incoming.result)
            raise FloatingPointError("Nan detected")

    def detail_forward(self, incoming):
        incoming.result = Storage()
        self.embLayer.forward(incoming)
        self.postEncoder.forward(incoming)
        self.connectLayer.forward(incoming)
        self.genNetwork.detail_forward(incoming)

class PointerEmbeddingLayer(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.args = args = param.args
        self.param = param
        volatile = param.volatile
        # embedding layer not initialized with pretrained embedding
        self.embLayer = nn.Embedding(
            volatile.dm.vocab_size, args.embedding_size)

    def forward(self, incoming):
        incoming.post = Storage()
        incoming.post.embedding = self.embLayer(incoming.data.post)
        incoming.resp = Storage()
        incoming.resp.embedding = self.embLayer(incoming.data.resp)
        incoming.resp.embLayer = self.embLayer

class PointerGenNetwork(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.args = args = param.args
        self.param = param

        self.GRULayer = SingleAttnGRUPointerGen(
            args.embedding_size, args.dh_size, args.eh_size * 2,
            coverage=args.coverage, initpara=False, gru_input_attn=True)
        self.wLinearLayer = nn.Linear(
            args.dh_size + args.eh_size * 2, param.volatile.dm.vocab_size)
        self.lossCE = nn.CrossEntropyLoss()
        self.start_generate_id = param.volatile.dm.go_id
        self.drop = nn.Dropout(args.droprate)

    def sum_on_voc_index(self, att, voc, idx):
        # reduce text attention to a sum over vocabulary
        # with scatter_method
        # resp_length, batch size, 1, post_length
        idx = torch.transpose(idx, 0, 1)
        idx = torch.cat([ idx.reshape(1, idx.shape[0], 1, idx.shape[1]) ] * att.shape[0])
        x = att.reshape(att.shape[0], att.shape[1], 1, att.shape[2])
        # batch size, voc_length, post_length
        shape = cuda(torch.zeros(att.shape[1], voc.shape[2], att.shape[2]))
        # iterate on resp_length
        ret = [ shape.scatter_(1, a, b).sum(2) for a, b in zip(idx, x) ]
        return torch.stack(ret, dim=0)

    def teacherForcing(self, inp, gen):
        embedding = inp.embedding
        embedding = self.drop(embedding)
        _, gen.h, att_weights, p_gen, coverages = self.GRULayer.forward(
            embedding, inp.resp_length-1, inp.post, inp.post_length, h_init=inp.init_h)
        gen.h = torch.stack(gen.h, dim=0)
        gen.h = self.drop(gen.h)
        gen.att = torch.stack(att_weights, dim=0)
        gen.p = torch.stack(p_gen, dim=0)
        gen.cov = torch.stack(coverages, dim=0)

    def forward(self, incoming):
        inp = Storage()
        inp.resp_length = incoming.data.resp_length
        inp.embedding = incoming.resp.embedding
        inp.post = incoming.hidden.h
        inp.post_length = incoming.data.post_length
        inp.init_h = incoming.conn.init_h
        incoming.gen = gen = Storage()
        self.teacherForcing(inp, gen)
        gen.w = self.wLinearLayer(gen.h)
        if self.args.pointer_gen:
            # calc distribution on voc with pointer gen
            voc_att = self.sum_on_voc_index(gen.att, gen.w, incoming.data.post)
            gen.w = gen.p * gen.w + (1 - gen.p) * voc_att
        w_o_f = flattenSequence(gen.w, incoming.data.resp_length-1)
        data_f = flattenSequence(
            incoming.data.resp[1:], incoming.data.resp_length-1)
        incoming.result.word_loss = self.lossCE(w_o_f, data_f)
        if self.args.coverage:
            cov_loss = flattenSequence(torch.min(gen.cov, gen.att).sum(2),
                                       incoming.data.resp_length-1) * self.args.cov_loss_wt
            incoming.result.word_loss += cov_loss.mean()
        incoming.result.perplexity = torch.exp(incoming.result.word_loss)

    def detail_forward(self, incoming):
        inp = Storage()
        batch_size = inp.batch_size = incoming.data.batch_size
        inp.init_h = incoming.conn.init_h
        inp.post = incoming.hidden.h
        inp.post_length = incoming.data.post_length
        inp.embLayer = incoming.resp.embLayer
        inp.dm = self.param.volatile.dm
        inp.max_sent_length = self.args.max_sent_length
        inp.data = incoming.data
        incoming.gen = gen = Storage()
        self.freerun(inp, gen)
        dm = self.param.volatile.dm
        w_o = gen.w_o.detach().cpu().numpy()
        incoming.result.resp_str = resp_str = \
            [" ".join(dm.convert_ids_to_tokens(w_o[:, i].tolist()))
             for i in range(batch_size)]
        incoming.result.golden_str = golden_str = \
            [" ".join(dm.convert_ids_to_tokens(incoming.data.resp[:, i].detach().cpu().numpy().tolist()))
             for i in range(batch_size)]
        incoming.result.post_str = post_str = \
            [" ".join(dm.convert_ids_to_tokens(incoming.data.post[:, i].detach().cpu().numpy().tolist()))
             for i in range(batch_size)]
        incoming.result.show_str = "\n".join(["post: " + a + "\n" + "resp: " + b + "\n" +
                                              "golden: " + c + "\n"
                                              for a, b, c in zip(post_str, resp_str, golden_str)])

    def freerun(self, inp, gen):
        # mode: beam = beamsearch; max = choose max;
        # sample = random_sampling; sample10 = sample from max 10

        def wLinearLayerCallback(gen, inp):
            gru_h, att, p_gen = gen
            gru_h = self.drop(gru_h)
            w = self.wLinearLayer(gru_h)

            if self.args.pointer_gen:
                # calc distribution on voc with pointer gen
                w = torch.transpose(w, 0, 1)
                att = torch.transpose(att, 0, 2)
                p_gen = torch.transpose(p_gen, 0, 1)
                voc_att = self.sum_on_voc_index(att, w, inp.data.post)
                w = p_gen * w + (1 - p_gen) * voc_att
                w = torch.transpose(w, 0, 1)

            return w

        def input_callback(i, now):
            return self.drop(now)

        if self.args.decode_mode == "beam":
            new_gen = self.GRULayer.beamsearch(inp, self.args.top_k,
                                               wLinearLayerCallback,
                                               input_callback=input_callback,
                                               no_unk=True,
                                               length_penalty=self.args.length_penalty)

            w_o = []
            length = []
            for i in range(inp.batch_size):
                w_o.append(new_gen.w_o[:, i, 0])
                length.append(new_gen.length[i][0])
            gen.w_o = torch.stack(w_o).transpose(0, 1)
            gen.length = length

        else:
            new_gen = self.GRULayer.freerun(inp, wLinearLayerCallback,
                                            self.args.decode_mode,
                                            input_callback=input_callback,
                                            no_unk=True, top_k=self.args.top_k)
            gen.w_o = new_gen.w_o
            gen.length = new_gen.length
