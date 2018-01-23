import sys
sys.path.extend(["../../","../","./"])
from driver.graph import *
from data.dataloader import *
from driver.config import *
import time
import numpy as np
import dynet as dy
import pickle


def train(data, dev_data, test_data, graph, vocab, config):
    pc = graph.parameter_collection
    trainer = dy.AdamTrainer(pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)
    trainer.set_clip_threshold(config.clip)


    global_step = 0

    def update_parameters():
        trainer.learning_rate = config.learning_rate * config.decay ** int(global_step / config.decay_steps)
        trainer.update()

    best_UAS = 0.

    for iter in range(config.train_iters):
        start_time = time.time()
        batch_iter = 0
        for onebatch in data_iter(data, config.train_batch_size, True, True):
            dy.renew_cg()
            losses = []
            sumLength = 0
            for words, extwords, tags, heads, rels in sentences_numberize(onebatch, vocab):
                curloss = graph.compute_loss(words, extwords, tags, heads, rels)
                losses.append(curloss)
                sumLength += len(words)

            loss = dy.esum(losses)
            loss = loss / sumLength
            loss = loss / config.update_every
            loss_value = loss.scalar_value()
            loss.backward()

            print("Step %d: LR: %.4f, Iter: %d, batch: %d, length %d, loss %.2f" \
                  %(global_step, trainer.learning_rate, iter, batch_iter,  sumLength, loss_value))

            batch_iter += 1
            if (batch_iter%config.update_every == 0):
                update_parameters()
                global_step += 1
            

            if batch_iter % config.validate_every == 0:
                arc_correct, rel_correct, arc_total, dev_uas, dev_las = evaluate(dev_data, \
                    graph, vocab, config.dev_file + '.' + str(iter) + '-' + str(batch_iter))
                print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))
                arc_correct, rel_correct, arc_total, test_uas, test_las = evaluate(test_data, graph, vocab, \
                                config.test_file + '.' + str(iter) + '-' + str(batch_iter))
                print("Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, test_uas, rel_correct, arc_total, test_las))
                if dev_uas > best_UAS:
                    print("Exceed best uas: history = %.2f, current = %.2f" %(best_UAS, dev_uas))
                    best_UAS = dev_uas
                    if config.save_after > 0 and iter > config.save_after:
                        graph.save(config.save_model_path)

        arc_correct, rel_correct, arc_total, dev_uas, dev_las = evaluate(dev_data, graph,\
                        vocab, config.dev_file + '.' + str(iter) + '-' + str(batch_iter))
        print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
              (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))
        arc_correct, rel_correct, arc_total, test_uas, test_las = evaluate(test_data, graph, \
                        vocab, config.test_file + '.' + str(iter) + '-' + str(batch_iter))
        print("Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
              (arc_correct, arc_total, test_uas, rel_correct, arc_total, test_las))
        if dev_uas > best_UAS:
            print("Exceed best uas: history = %.2f, current = %.2f" % (best_UAS, dev_uas))
            best_UAS = dev_uas
            if config.save_after > 0 and iter > config.save_after:
                graph.save(config.save_model_path)
        print('iter: ', iter, ' train: ', time.time() - start_time)


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

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las


if __name__ == '__main__':
    np.random.seed(666)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = creatVocab(config.train_file, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))
    graph = ParserGraph(vocab, config, vec)

    data = read_corpus(config.train_file, vocab)
    dev_data = read_corpus(config.dev_file, vocab)
    test_data = read_corpus(config.test_file, vocab)

    train(data, dev_data, test_data, graph, vocab, config)