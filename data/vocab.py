from collections import Counter
from data.dependency import *
from data.pretrained_embedding import *
#import torch.nn as nn

class Vocab(object):
    PAD, ROOT, UNK = 0, 1, 2
    def __init__(self, word_counter, extword_list, tag_counter, rel_counter, relroot='root'):
        self._root = relroot
        self._id2word = ['<pad>', relroot.lower(), '<unk>']
        self._wordid2freq = [10000, 10000, 10000]
        self._id2extword = ['<pad>']
        self._id2tag = ['<pad>', relroot]
        self._id2rel = ['<pad>', relroot]
        for word, count in word_counter.most_common():
            self._id2word.append(word)
            self._wordid2freq.append(count)

        for tag, count in tag_counter.most_common():
            self._id2tag.append(tag)

        for rel, count in rel_counter.most_common():
            if rel != relroot: self._id2rel.append(rel)

        self._id2extword += extword_list

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._extword2id = reverse(self._id2extword)
        self._tag2id = reverse(self._id2tag)
        self._rel2id = reverse(self._id2rel)



        print("Vocab info: #words %d,  #extwords %d, #tags %d #rels %d" % (self.vocab_size, self.extvocab_size, self.tag_size, self.rel_size))

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x) for x in xs]
        return self._tag2id.get(xs)

    def id2tag(self, xs):
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    def save(self, savefile):
        with open(savefile, 'w') as file:
            file.write('rel: ' + str(len(self._id2rel)) + '\n')
            for strRel in self._id2rel:
                file.write(strRel + '\n')
            file.write('tag: ' + str(len(self._id2tag)) + '\n')
            for strTag in self._id2tag:
                file.write(strTag + '\n')
            file.write('word: ' + str(len(self._id2word)) + '\n')
            for strWord, wordFreq in zip(self._id2word, self._wordid2freq):
                file.write(strWord + '\t' + str(wordFreq) + '\n')
            file.write('extword: ' + str(len(self._id2extword)) + '\n')
            for strExtWord in self._id2extword:
                file.write(strExtWord + '\n')

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def rel_size(self):
        return len(self._id2rel)

def creatVocab(corpusFile, extWord):
    word_counter = Counter()
    tag_counter = Counter()
    rel_counter = Counter()
    root = ''

    with open(corpusFile, 'r') as infile:
        for sentence in readDepTree(infile):
            for dep in sentence:
                word_counter[dep.form] += 1
                tag_counter[dep.tag] += 1
                if dep.head != 0:
                    rel_counter[dep.rel] += 1
                elif root == '':
                    root = dep.rel
                    rel_counter[dep.rel] += 1
                elif root != dep.rel:
                    print('root = ' + root + ', rel for root = ' + dep.rel)


    return Vocab(word_counter, extWord, tag_counter, rel_counter, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input file',
                      default='examples/ptb/dev.ptb.dep')
    parser.add_argument('--emb', help='emb file',
                      default='examples/glove.6B.100d.txt')
    parser.add_argument('--output', help='output file',
                  default='examples/ptb/dev.ptb.vocab')

    args = parser.parse_args()

    word, vec = load_all_pretrained_embeddings(args.emb)

    vob = creatVocab(args.input, word)
    vob.save(args.output)
