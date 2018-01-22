from __future__ import division
from driver.layer import *
import dynet as dy
from driver.mst import *



def dynet_flatten_numpy(ndarray):
    return np.reshape(ndarray, (-1,), 'F')

class ParserGraph(object):
    def __init__(self, vocab, config, pretrained_embedding):
        pc = dy.ParameterCollection()
        self.config = config
        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        self.word_embs = pc.lookup_parameters_from_numpy(word_init)
        self.pret_word_embs = pc.lookup_parameters_from_numpy(pretrained_embedding)
        tag_init = np.random.randn(vocab.tag_size, config.tag_dims).astype(np.float32);
        #tag_init = tag_init / np.sqrt(config.tag_dims)
        self.tag_embs = pc.lookup_parameters_from_numpy(tag_init)
        self.dropout_emb = config.dropout_emb
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


    @property
    def parameter_collection(self):
        return self._pc
        
    def save(self, save_path):
        self._pc.save(save_path)

    def load(self, load_path):
        self._pc.populate(load_path)
        
    def forward(self, words, extwords, tags, isTrain):
        # inputs, targets: seq_le
        seq_len = len(words)

        dynamic_embs = [dy.lookup(self.word_embs, w) for w in words]
        static_embs = [dy.lookup(self.pret_word_embs, w, update=False) for w in extwords]
        word_embs = [dynamic_emb + static_emb for dynamic_emb, static_emb in zip(dynamic_embs, static_embs)]
        tag_embs = [dy.lookup(self.tag_embs, pos) for pos in tags]

        if isTrain:
            word_masks = np.random.binomial(1, 1. - self.dropout_emb, seq_len).astype(np.float32)
            tag_masks = np.random.binomial(1, 1. - self.dropout_emb, seq_len).astype(np.float32)
            scale = 3. / (2. * word_masks + tag_masks + 1e-12)
            word_masks *= scale
            tag_masks *= scale
            word_embs = [dy.cmult(word_emb, dy.inputVector([word_mask])) \
                         for word_emb, word_mask in zip(word_embs, word_masks)]
            tag_embs = [dy.cmult(tag_emb, dy.inputVector([tag_mask])) \
                         for tag_emb, tag_mask in zip(tag_embs, tag_masks)]


        emb_inputs = [ dy.concatenate([word_emb, pos_emb]) \
                      for word_emb, pos_emb in zip(word_embs, tag_embs)]

        # (2 * lstm_hiddens) * seq_len
        bilstm_out = dy.concatenate_cols(
            biLSTM(self.LSTM_builders, emb_inputs, self.dropout_lstm_input if isTrain else 0.,
                   self.dropout_lstm_hidden if isTrain else 0.))

        if isTrain:
            bilstm_out = dy.dropout_dim(bilstm_out, 1, self.dropout_mlp)

        # (mlp_arc_size + mlp_rel_size) * seq_len
        W_dep, b_dep = dy.parameter(self.mlp_dep_W), dy.parameter(self.mlp_dep_b)
        dep = leaky_relu(dy.affine_transform([b_dep, W_dep, bilstm_out]))

        W_head, b_head = dy.parameter(self.mlp_head_W), dy.parameter(self.mlp_head_b)
        head = leaky_relu(dy.affine_transform([b_head, W_head, bilstm_out]))

        if isTrain:
            dep, head = dy.dropout_dim(dep, 1, self.dropout_mlp), dy.dropout_dim(head, 1, self.dropout_mlp)

        # mlp_arc_size * seq_len,  mlp_rel_size * seq_len
        dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
        head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

        # (#head x #dep)
        W_arc = dy.parameter(self.arc_W)
        arc_logits = bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size, seq_len, num_outputs=1,
                              bias_x=True, bias_y=False)

        # (#head x rel_size x #dep)
        W_rel = dy.parameter(self.rel_W)
        rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len,
                              num_outputs=self.rel_size, bias_x=True, bias_y=True)

        return arc_logits, rel_logits

    def compute_loss(self, words, extwords, tags, true_arcs, true_labels):
        arc_logits, rel_logits = self.forward(words, extwords, tags, True)
        seq_len = len(true_arcs)
        targets_1D = dynet_flatten_numpy(true_arcs)
        flat_arc_logits = dy.reshape(arc_logits, (seq_len,), seq_len)
        losses = dy.pickneglogsoftmax_batch(flat_arc_logits, targets_1D)
        arc_loss = dy.sum_batches(losses)

        flat_rel_logits = dy.reshape(rel_logits, (seq_len, self.rel_size), seq_len)
        partial_rel_logits = dy.pick_batch(flat_rel_logits, targets_1D)
        targets_rel1D = dynet_flatten_numpy(true_labels)
        losses = dy.pickneglogsoftmax_batch(partial_rel_logits, targets_rel1D)
        rel_loss = dy.sum_batches(losses)

        loss = arc_loss + rel_loss

        return loss

    def parse(self, words, extwords, tags):
        arc_logits, rel_logits = self.forward(words, extwords, tags, False)
        seq_len = len(words)
        flat_arc_logits = dy.reshape(arc_logits, (seq_len,), seq_len)
        arc_probs = dy.softmax(flat_arc_logits)
        flat_rel_logits = dy.reshape(rel_logits, (seq_len, self.rel_size), seq_len)
        rel_probs = dy.softmax(dy.transpose(flat_rel_logits))

        return arc_probs, rel_probs