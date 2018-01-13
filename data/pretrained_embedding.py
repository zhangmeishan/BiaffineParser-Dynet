import numpy as np

def load_all_pretrained_embeddings(embfile):
    embedding_dim = -1
    word_count = 0
    with open(embfile, encoding='utf-8') as f:
        for line in f.readlines():
            if word_count < 1:
                values = line.split()
                embedding_dim = len(values) - 1
            word_count += 1
    print('Total words: ' + str(word_count) + '\n')
    print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')
    words = []
    embeddings = np.zeros((word_count + 1, embedding_dim))
    index = 0
    padding_idx = 0
    with open(embfile, encoding='utf-8') as f:
        for line in f.readlines():
            if index == padding_idx:
                words.append('<pad>')
                index += 1
            values = line.split()
            words.append(values[0])
            vector = np.array(values[1:], dtype='float32')
            embeddings[index] = vector
            index += 1

    embeddings = embeddings / np.std(embeddings)


    return words, embeddings


def load_chosen_pretrained_embeddings(embfile, word2idx):
    embedding_dim = -1
    with open(embfile, encoding='utf-8') as f:
        for line in f:
            if embedding_dim < 1:
                values = line.split()
                embedding_dim = len(values) - 1
            else:
                break
    word_count = len(word2idx)
    print('Total words: ' + str(len(word2idx)) + '\n')
    print('The dim of pretrained embeddings: ' + embedding_dim + '\n')
    embeddings = np.zeros((word_count, embedding_dim))
    with open(embfile, encoding='utf-8') as f:
        for line in f.readlines():
            values = line.split()
            index = word2idx.get(values[0])
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector

    embeddings = embeddings / np.std(embeddings)

    return embeddings