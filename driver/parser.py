import sys
sys.path.extend(["../../","../","./"])
from driver.graph import *
from driver.mst import *
from data.dataloader import *
from driver.config import *
import time
import numpy as np
#import random
import dynet as dy

'''
def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)


def _model_var(graph, x):
    p = next(filter(lambda p: p.requires_grad, graph.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)
'''

class BiaffineParser(object):
    def __init__(self, graph, root_id):
        self.graph = graph
        self.root = root_id
        self.rel_size = graph.rel_size

    def forward(self, words, extwords, tags, isTrain):
        # arc_logits: heads * deps * batch
        # rel_logits: heads * labels * deps * batch
        self.arc_logits, self.rel_logits = self.graph.forward(words, extwords, tags, isTrain)

        self.batch_size = words.shape[1]
        self.seq_len = words.shape[0]
        self.mask = np.greater(words, self.root).astype(np.float32)
        self.num_tokens = int(np.sum(self.mask))




    def compute_loss(self, true_arcs, true_labels):
        # seq_len x batch_size
        mask_1D = dynet_flatten_numpy(self.mask)
        mask_1D_tensor = dy.inputTensor(mask_1D, batched=True)

        # seq_len x batch_size
        arc_preds = self.arc_logits.npvalue().argmax(0)
        arc_correct = np.equal(arc_preds, true_arcs).astype(np.float32) * self.mask
        arc_accuracy = np.sum(arc_correct) / self.num_tokens

        targets_1D = dynet_flatten_numpy(true_arcs)
        flat_arc_logits = dy.reshape(self.arc_logits, (self.seq_len,), self.seq_len * self.batch_size)
        losses = dy.pickneglogsoftmax_batch(flat_arc_logits, targets_1D)

        arc_loss = dy.sum_batches(losses * mask_1D_tensor) / self.num_tokens

        # (#head x rel_size) x (#dep x batch_size)
        flat_rel_logits = dy.reshape(self.rel_logits,  \
                                     (self.seq_len, self.rel_size), self.seq_len * self.batch_size)

        # (rel_size) x (#dep x batch_size)
        partial_rel_logits = dy.pick_batch(flat_rel_logits, targets_1D)

        # seq_len x batch_size
        rel_preds = partial_rel_logits.npvalue().argmax(0)
        targets_1D = dynet_flatten_numpy(true_labels)
        rel_correct = np.equal(rel_preds, targets_1D).astype(np.float32) * mask_1D
        rel_accuracy = np.sum(rel_correct) / self.num_tokens
        losses = dy.pickneglogsoftmax_batch(partial_rel_logits, targets_1D)

        rel_loss = dy.sum_batches(losses * mask_1D_tensor) / self.num_tokens

        loss = arc_loss + rel_loss
        correct = rel_correct * dynet_flatten_numpy(arc_correct)

        overall_accuracy = np.sum(correct) / self.num_tokens


        return arc_accuracy, rel_accuracy, overall_accuracy, loss



    def parse(self, words, extwords, tags, lengths):
        batch_size = words.shape[1]
        seq_len = words.shape[0]
        if batch_size == 0 or seq_len == 0: return

        self.forward(words, extwords, tags, False)
        ROOT = self.root
        pred_heads = -1 * np.ones((seq_len, batch_size), dtype=np.int32)
        pred_rels = -1 * np.ones((seq_len, batch_size), dtype=np.int32)


        flat_arc_logits = dy.reshape(self.arc_logits, (self.seq_len,), self.seq_len * self.batch_size)

        # batch_size x #dep x #head
        arc_probs = np.transpose(np.reshape(dy.softmax(flat_arc_logits).npvalue(), \
                                        (self.seq_len, self.seq_len, self.batch_size), 'F'))

        # (#head x rel_size) x (#dep x batch_size)
        flat_rel_logits = dy.reshape(self.rel_logits, \
                                     (self.seq_len, self.rel_size), self.seq_len * self.batch_size)
        # batch_size x #dep x #head x rel_size
        rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(), \
                        (self.rel_size, self.seq_len, self.seq_len, self.batch_size), 'F'))

        for arc_logit, label_logit, length in zip(arc_probs, rel_probs, lengths):
            arcs, arc_probs = mst(arc_logit, length)
            label_probs = softmax2d(label_logit[np.arange(length), arcs], length, label_logit.shape[2])
            labels = np.argmax(label_probs, axis=1)

            tokens = np.arange(1, length)
            roots = np.where(labels[tokens] == ROOT)[0] + 1
            if len(roots) < 1:
                root_arc = np.where(arcs[tokens] == 0)[0] + 1
                labels[root_arc] = ROOT
            elif len(roots) > 1:
                label_probs[roots, ROOT] = 0
                new_labels = np.argmax(label_probs[roots], axis=1)
                root_arc = np.where(arcs[tokens] == 0)[0] + 1
                labels[roots] = new_labels
                labels[root_arc] = ROOT
            arcs[0] = -1
            labels[0] = ROOT

            for index in range(length):
                pred_heads[index] = arcs[index]
                pred_rels[index] = labels[index]

        return pred_heads, pred_rels


def train(data, dev_data, test_data, graph, parser, vocab, config):
    pc = graph.parameter_collection
    trainer = dy.AdamTrainer(pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)

    global_step = 0

    def update_parameters():
        trainer.learning_rate = config.learning_rate * config.decay ** (global_step / config.decay_steps)
        trainer.update()

    best_UAS = 0.

    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        for onebatch in data_iter(data, config.train_batch_size, True):
            # optimizer.zero_grad()
            words, extwords, tags, heads, rels, lengths = batch_data_variable(onebatch, vocab, True)
            sumLength = sum(lengths)

            parser.forward(words, extwords, tags, True)
            arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.compute_loss(heads, rels)
            loss = loss / config.update_every
            loss_value = loss.scalar_value()
            loss.backward()

            print('batch: ' + str(batch_iter)  + ', arc: ' + str(arc_accuracy) + \
                  ', rel: ' + str(rel_accuracy) + ', accuracy: ' + str(overall_accuracy) + \
                  ', length: ' + str(sumLength)  + ', loss: ' + str(loss_value))

            if (batch_iter%config.update_every == 0):
                update_parameters()
                global_step += 1
            
            batch_iter += 1

            if batch_iter % config.validate_every == 0:
                print('dev')
                uas, las = evaluate(dev_data, parser, vocab, config.dev_file + '.' + str(iter) + '-' + str(batch_iter))
                print('test')
                evaluate(test_data, parser, vocab, config.test_file + '.' + str(iter) + '-' + str(batch_iter))
                if uas > best_UAS:
                    print('Exceed best uas: history=' + str(best_UAS) + ', current=' + str(uas))
                    best_UAS = uas
                    #if iter > config.save_after: parser.save(config.save_model_path)

        print('dev')
        uas, las = evaluate(dev_data, parser, vocab, config.dev_file + '.' + str(iter) + '-' + str(batch_iter))
        print('test')
        evaluate(test_data, parser, vocab, config.test_file + '.' + str(iter) + '-' + str(batch_iter))
        if uas > best_UAS:
            print('Exceed best uas: history=' + str(best_UAS) + ', current=' + str(uas))
            best_UAS = uas
        print('iter: ', iter, ' train: ', time.time() - start_time)

def evaluate(data, parser, vocab, outputFile):
    start = time.time()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, label_total_test, label_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, tags, heads, rels, lengths = batch_data_variable(onebatch, vocab, False)
        count = 0
        pred_heads, pred_rels = parser.parse(words, extwords, tags, lengths)
        for tree in batch_variable_depTree(words, tags, pred_heads, pred_rels, lengths, vocab):
            printDepTree(output, tree)
            arc_total, arc_correct, label_total, label_correct = evalDepTree(tree, onebatch[count])
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            label_total_test += label_total
            label_correct_test += label_correct
            count += 1

    output.close()

    uas = arc_correct_test * 1.0 / arc_total_test
    las = label_correct_test * 1.0 / label_total_test

    print('UAS: ' + str(uas))
    print('LAS: ' + str(las))

    end = time.time()
    print('time: ', end - start)

    return uas, las


if __name__ == '__main__':
    np.random.seed(666)


    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--graph', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    word, vec = load_all_pretrained_embeddings(config.pretrained_embeddings_file)
    vocab = creatVocab(config.train_file, word)

    graph = ParserGraph(vocab, config, vec)
        
    parser = BiaffineParser(graph, vocab.ROOT) 

    data = read_corpus(config.train_file, vocab)
    dev_data = read_corpus(config.dev_file, vocab)
    test_data = read_corpus(config.test_file, vocab)

    train(data, dev_data, test_data, graph, parser, vocab, config)