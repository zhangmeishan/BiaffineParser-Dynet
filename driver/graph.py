from __future__ import division
from driver.layer import *
from data.vocab import *
import dynet as dy
import numpy as np


def dynet_flatten_numpy(ndarray):
    return np.reshape(ndarray, (-1,), 'F')

class ParserGraph(object):
    def __init__(self, vocab, config, pretrained_embedding):
        pc = dy.ParameterCollection()
        self.config = config
        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        self.word_embs = pc.lookup_parameters_from_numpy(word_init)
        self.pret_word_embs = pc.lookup_parameters_from_numpy(pretrained_embedding)
        tag_init = 2 * np.random.rand(vocab.tag_size, config.tag_dims).astype(np.float32)  - 1.0;
        tag_init = tag_init / np.sqrt(config.tag_dims)
        self.tag_embs = pc.lookup_parameters_from_numpy(tag_init)
        self.rel_size = vocab.rel_size

        self.LSTM_builders = []
        f = orthonormal_VanillaLSTMBuilder(1, config.word_dims + config.tag_dims, config.lstm_hiddens, pc)
        b = orthonormal_VanillaLSTMBuilder(1, config.word_dims + config.tag_dims, config.lstm_hiddens, pc)
        self.LSTM_builders.append((f, b))
        for i in range(config.lstm_layers - 1):
            f = orthonormal_VanillaLSTMBuilder(1, 2 * config.lstm_hiddens, config.lstm_hiddens, pc)
            b = orthonormal_VanillaLSTMBuilder(1, 2 * config.lstm_hiddens, config.lstm_hiddens, pc)
            self.LSTM_builders.append((f, b))

        self.dropout_lstm_input = config.dropout_lstm_input
        self.dropout_lstm_hidden = config.dropout_lstm_hidden

        mlp_size = config.mlp_arc_size + config.mlp_rel_size
        W = orthonormal_initializer(mlp_size, 2 * config.lstm_hiddens)
        self.mlp_dep_W = pc.parameters_from_numpy(W)
        self.mlp_head_W = pc.parameters_from_numpy(W)
        self.mlp_dep_b = pc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_head_b = pc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_arc_size = config.mlp_arc_size
        self.mlp_rel_size = config.mlp_rel_size
        self.dropout_mlp = config.dropout_mlp

        self.arc_W = pc.add_parameters((config.mlp_arc_size, config.mlp_arc_size + 1), init=dy.ConstInitializer(0.))
        self.rel_W = pc.add_parameters((vocab.rel_size * (config.mlp_rel_size + 1), config.mlp_rel_size + 1),
                                       init=dy.ConstInitializer(0.))

        self._pc = pc

        def _emb_mask_generator(seq_len, batch_size):
            ret = []
            for i in range(seq_len):
                word_mask = np.random.binomial(1, 1. - config.dropout_emb, batch_size).astype(np.float32)
                tag_mask = np.random.binomial(1, 1. - config.dropout_emb, batch_size).astype(np.float32)
                scale = 3. / (2. * word_mask + tag_mask + 1e-12)
                word_mask *= scale
                tag_mask *= scale
                word_mask = dy.inputTensor(word_mask, batched=True)
                tag_mask = dy.inputTensor(tag_mask, batched=True)
                ret.append((word_mask, tag_mask))
            return ret

        self.generate_emb_mask = _emb_mask_generator

    @property
    def parameter_collection(self):
        return self._pc

    def forward(self, words, extwords, tags, isTrain):
        dy.renew_cg()
        # inputs, targets: seq_len x batch_size
        batch_size = words.shape[1]
        seq_len = words.shape[0]

        dynamic_embs = [dy.lookup_batch(self.word_embs, w) for w in words]
        static_embs = [dy.lookup_batch(self.pret_word_embs, w, update=False) for w in extwords]
        word_embs = [dynamic_emb + static_emb for dynamic_emb, static_emb in zip(dynamic_embs, static_embs)]
        tag_embs = [dy.lookup_batch(self.tag_embs, pos) for pos in tags]

        if isTrain:
            emb_masks = self.generate_emb_mask(seq_len, batch_size)
            emb_inputs = [dy.concatenate([dy.cmult(word, wordm), dy.cmult(pos, posm)])
                          for word, pos, (wordm, posm) in zip(word_embs, tag_embs, emb_masks)]
        else:
            emb_inputs = [dy.concatenate([word, pos]) for word, pos in zip(word_embs, tag_embs)]

        bilstm_out = dy.concatenate_cols(
            biLSTM(self.LSTM_builders, emb_inputs, batch_size, self.dropout_lstm_input if isTrain else 0.,
                   self.dropout_lstm_hidden if isTrain else 0.))

        if isTrain:
            bilstm_out = dy.dropout_dim(bilstm_out, 1, self.dropout_mlp)

        W_dep, b_dep = dy.parameter(self.mlp_dep_W), dy.parameter(self.mlp_dep_b)
        dep = leaky_relu(dy.affine_transform([b_dep, W_dep, bilstm_out]))

        W_head, b_head = dy.parameter(self.mlp_head_W), dy.parameter(self.mlp_head_b)
        head = leaky_relu(dy.affine_transform([b_head, W_head, bilstm_out]))

        if isTrain:
            dep, head = dy.dropout_dim(dep, 1, self.dropout_mlp), dy.dropout_dim(head, 1, self.dropout_mlp)

        dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
        head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

        W_arc = dy.parameter(self.arc_W)
        arc_logits = bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size, seq_len, batch_size, num_outputs=1,
                              bias_x=True, bias_y=False)
        # (#head x #dep) x batch_size


        W_rel = dy.parameter(self.rel_W)
        rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len, batch_size,
                              num_outputs=self.rel_size, bias_x=True, bias_y=True)
        # (#head x rel_size x #dep) x batch_size

        return arc_logits, rel_logits