import sys
sys.path.extend(["../../","../","./"])
from driver.graph import *
from data.dataloader import *
from driver.config import *
import time
import numpy as np
import dynet as dy
import pickle


def evaluate(data, graph, vocab, outputFile):
    start = time.time()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False, False):
        dy.renew_cg()
        batch_arc_probs, batch_rel_probs = [], []
        for words, extwords, tags, heads, rels in sentences_numberize(onebatch, vocab):
            arc_probs, rel_probs = graph.parse(words, extwords, tags)
            batch_arc_probs.append(arc_probs)
            batch_rel_probs.append(rel_probs)

        dy.forward(batch_arc_probs + batch_rel_probs)

        batch_size = len(onebatch)
        for index in range(batch_size):
            seq_len = len(onebatch[index])
            arc_probs = batch_arc_probs[index].npvalue()
            arc_probs = np.transpose(np.reshape(arc_probs, (seq_len, seq_len), 'F'))
            rel_probs = batch_rel_probs[index].npvalue()
            rel_probs = np.transpose(np.reshape(rel_probs, (vocab.rel_size, seq_len, seq_len), 'F'))
            arc_pred = arc_argmax(arc_probs, seq_len)
            rel_probs = rel_probs[np.arange(len(arc_pred)), arc_pred]
            rel_pred = rel_argmax(rel_probs, seq_len, vocab.ROOT)

            tree = append2Tree(arc_pred, rel_pred, vocab, onebatch[index])
            printDepTree(output, tree)
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, onebatch[index])
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct

    output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test

    end = time.time()
    print('sentence num:' + str(len(data)) + ', parse time: ', end - start)

    return uas, las


if __name__ == '__main__':
    np.random.seed(666)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    vec = vocab.create_pretrained_embs(config.pretrained_embeddings_file)
    graph = ParserGraph(vocab, config, vec)
    graph.load(config.load_model_path)

    test_data = read_corpus(config.test_file, vocab)

    test_uas, test_las = evaluate(test_data, graph, vocab, config.test_file)
    print("Test: uas = %.2f, las = %.2f" % (test_uas, test_las))


