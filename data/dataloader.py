from data.vocab import *
import numpy as np

def read_corpus(file_path, vocab=None):
    data = []
    with open(file_path, 'r') as infile:
        for sentence in readDepTree(infile, vocab):
            data.append(sentence)
    return data

def sentences_numberize(sentences, vocab):
    for sentence in sentences:
        yield sentence2id(sentence, vocab)

def sentence2id(sentence, vocab):
    words, extwords, tags, arcs, rels = [], [], [], [], []
    for dep in sentence:
        words.append(vocab.word2id(dep.form))
        extwords.append(vocab.extword2id(dep.form))
        tags.append(vocab.tag2id(dep.tag))
        arcs.append(dep.head)
        rels.append(vocab.rel2id(dep.rel))

    return words, extwords, tags, arcs, rels


def batch_slice(data, batch_size, sort=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]
        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(sentences[src_id]), reverse=True)
            sentences = [sentences[src_id] for src_id in src_ids]

        yield sentences


def data_iter(data, batch_size, shuffle=True, sort=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size, sort)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def append2Tree(heads, rels, vocab, goldTree):
    length = len(goldTree)
    sentence = []
    for idx in range(length):
        sentence.append(Dependency(idx, goldTree[idx].form, goldTree[idx].tag, \
                            heads[idx], vocab.id2rel(rels[idx])))
    return sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input file',
                      default='examples/dev.ptb.dep')
    parser.add_argument('--emb', help='emb file',
                      default='examples/glove.6B.100d.txt')
    parser.add_argument('--output', help='output file',
                  default='examples/dev.ptb.batchout')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')

    args = parser.parse_args()

    data = read_corpus(args.input)
    file = open(args.output + '.txt', 'w')
    count = 0
    for onebatch in data_iter(data, args.batch_size, False):
        file.write('batch ' + str(count) + ': ' + str(len(onebatch)) + '\n')
        for sentence in onebatch:
            file.write(str(len(sentence)) + '\n')
            for dep in sentence:
                values = [dep.form, dep.tag, str(dep.head), dep.rel]
                file.write('\t'.join(values) + '\n')
            file.write('\n')
        count += 1
    file.close()
    print(count)

    vocab = creatVocab(args.input)
    vec = vocab.load_pretrained_embs(args.emb)

    data = read_corpus(args.input)
    file = open(args.output + '.id', 'w')
    count = 0
    for iter in range(10):
        file.write('Iteration: ' + str(iter) + '\n')
        for onebatch in data_iter(data, args.batch_size, False):
            file.write('batch ' + str(count) + ': '  + str(len(onebatch))+ '\n')
            for sentence in sentences_numberize(onebatch, vocab, iter!=0):
                file.write(str(len(sentence)) + '\n')
                for dep in sentence:
                    values = [str(dep[0]), str(dep[1]), str(dep[2]), str(dep[3]), str(dep[4])]
                    file.write('\t'.join(values) + '\n')
                file.write('\n')
            count += 1
        print(count)
    file.close()
    print(count)


