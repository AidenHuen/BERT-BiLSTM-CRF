# encoding:utf-8

import numpy
from collections import Counter

from keras import Sequential
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
import config
import random
import pickle
from skimage import io, transform
import codecs
from skimage.viewer import ImageViewer
from bert_serving.client import BertClient

para = config.para

def get_char2id(train_x, id2id, maxlen):
    char_l = []
    lose = [0]*maxlen
    for sentence in train_x:
        sent_l = []
        for word_id in sentence:
            try:
                # print(id2id[word_id])
                sent_l.append(id2id[word_id])
            except Exception, e:
                sent_l.append(lose)
        char_l.append(sent_l)
    return char_l

def cross_validation(X,Y,fold):
    val_X = []
    val_Y = []
    train_X = []
    train_Y = []
    step = int(X.__len__() / fold)
    for i in range(fold):
        if i != fold - 1:
            val_X.append(X[step * i:step * (i + 1)])
            val_Y.append(Y[step * i:step * (i + 1)])
        else:
            val_X.append(X[step * i:])
            val_Y.append(Y[step * i:])
    for i in range(fold):
        X_list = []
        Y_list = []
        for j in range(val_X.__len__()):
            if j != i:
                X_list.append(val_X[j])
                Y_list.append(val_Y[j])
        train_X.append(numpy.concatenate(X_list, axis=0))
        train_Y.append(numpy.concatenate(Y_list, axis=0))
    return train_X, train_Y, val_X, val_Y

def train_test_dev_preprocess():
    train = _parse_data(codecs.open(para["train_path"], 'r'), sep=para["sep"])
    test = _parse_data(codecs.open(para["test_path"], 'r'), sep=para["sep"])
    dev = _parse_data(codecs.open(para["dev_path"], 'r'), sep=para["sep"])
    # train_len = train.__len__()
    print("Load dataset finish!!")
    dataset = train+test+dev
    tags = get_tag(dataset)
    print(tags)
    print(train.__len__(), test.__len__(), dev.__len__(), dataset.__len__())
    word_counts = Counter(row[0].lower() for sample in dataset for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 1]
    word2id = dict((w, i + 1) for i, w in enumerate(vocab))

    train_X, train_Y = process_data(train, word2id, tags)
    dev_X, dev_Y = process_data(dev, word2id, tags)
    test_X, test_Y = process_data(test, word2id, tags)
    pickle.dump((train_X, train_Y,  test_X, test_Y, dev_X,dev_Y, word2id, tags), open(para["data_pk_path"], "wb"))

def train_test_set_preprocess():
    train = _parse_data(codecs.open(para["train_path"], 'r'), sep=para["sep"])
    test = _parse_data(codecs.open(para["test_path"], 'r'), sep=para["sep"])

    # train = dic+train
    print("Load trainset,dataset finish!!")
    dataset = train+test
    tags = get_tag(dataset)
    print(tags)
    print(train.__len__(),test.__len__(),dataset.__len__())
    word_counts = Counter(row[0].lower() for sample in dataset for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 1]
    word2id = dict((w, i + 1) for i, w in enumerate(vocab))
    # print(word2id)
    train_X, train_Y = process_data(train, word2id, tags)
    print(tags)
    test_X, test_Y = process_data(test, word2id, tags)
    print(train_X.shape,train_Y.shape)
    pickle.dump((train_X, train_Y,  test_X, test_Y, word2id, tags), open(para["data_pk_path"], "wb"))

def load_bert_repre():
    train = _parse_data(codecs.open(para["train_path"], 'r'), sep=para["sep"])
    test = _parse_data(codecs.open(para["test_path"], 'r'), sep=para["sep"])
    train = [[items[0] for items in sent] for sent in train]
    test = [[items[0] for items in sent] for sent in test]

    train_x = numpy.zeros(shape=(train.__len__(),para["max_len"],768),dtype="float32")
    test_x = numpy.zeros(shape=(test.__len__(),para["max_len"],768),dtype="float32")

    bc = BertClient()

    step = int(train.__len__()/256)+1
    for i in range(step):
        if i != step-1:
            x = bc.encode(train[i*256:(i+1)*256], is_tokenized=True)
            x = x[:,1:para["max_len"]+1]
            train_x[i*256:((i+1)*256)] = x
            # print(train_x[i*256:(i+1)*256])
        else:
            x = bc.encode(train[i*256:],is_tokenized=True)
            x = x[:,1:para["max_len"]+1]
            train_x[i*256:] = x
            # print(train_x[i*256:])

    step = int(test.__len__() / 256) + 1
    # print(step)
    for i in range(step):
        if i != step - 1:
            x = bc.encode(test[i * 256:(i + 1) * 256], is_tokenized=True)
            x = x[:, 1:para["max_len"]+1]
            test_x[i * 256:((i + 1) * 256)] = x
            print(test_x[i * 256:(i + 1) * 256])
        else:
            x = bc.encode(test[i * 256:], is_tokenized=True)
            x = x[:, 1:para["max_len"]+1]
            test_x[i * 256:] = x
            # print(test_x[i * 256:])
    return train_x, test_x

def load_path_bert(path,sep="\t"):

    test = _parse_data(codecs.open(path, 'r'), sep=sep)
    test = [[items[0] for items in sent] for sent in test]
    test_x = numpy.zeros(shape=(test.__len__(), para["max_len"], 768),dtype="float32")
    bc = BertClient()

    step = int(test.__len__() / 256) + 1
    print(step)
    for i in range(step):
        if i != step - 1:
            x = bc.encode(test[i * 256:(i + 1) * 256], is_tokenized=True)
            x = x[:, 1:para["max_len"]+1]
            test_x[i * 256:((i + 1) * 256)] = x
            # print(test_x[i * 256:(i + 1) * 256])
        else:
            x = bc.encode(test[i * 256:], is_tokenized=True)
            x = x[:, 1:para["max_len"]+1]
            test_x[i * 256:] = x
            # print(test_x[i * 256:])
    # pickle.dump(test_x, open("./data/bert-pku-seg.pk", "wb"))
    return test_x

def get_tag(data):
    tag = []
    for words in data:
        for word_tag in words:
            if word_tag[1] not in tag:
                tag.append(word_tag[1])
    return tag


def _parse_data(file_input,sep="\t"):
    rows = file_input.readlines()
    rows[0] = rows[0].replace('\xef\xbb\xbf', '')
    items = [row.strip().split(sep) for row in rows]
    # print(items)
    max_len = 0
    sents = []
    sent = []
    n = 0
    for item in items:

        if item.__len__() != 1:
            sent.append(item)
        else:
            if sent.__len__() > para["max_len"]:
                n += 1
                split_sent = []
                for i, item in enumerate(sent):
                    if item[0] in ["。",",","，","!","！","?","？", "、", "；"] and split_sent.__len__()>50:
                        split_sent.append(item)
                        if split_sent.__len__() < para["max_len"]:
                            # for item in split_sent:
                            #     if item[1] != "O":
                            #         sents.append(split_sent[:])
                            #         break
                            # print(" ".join([item[0] for item in split_sent]))
                            sents.append(split_sent[:])
                        # else:
                        #     for item in split_sent:
                        #         print item[0],
                        #     print ""
                        split_sent = []
                    else:
                        split_sent.append(item)

                    # if i == sent.__len__()-1 and split_sent.__len__() < config.max_len:
                    #     for item in split_sent:
                    #         sents[sents.__len__()-1].append(item)
                    #     split_sent = []
                # continue
            else:
                if sent.__len__() > 1:
                    sents.append(sent[:])
            sent = []
    print  ("over_maxlen_sentence_num:", n)
    return sents


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i+1) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    x = pad_sequences(x, maxlen, padding='post', truncating='post')  # left padding
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1, padding='post', truncating='post')
    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
        # print(y_chunk)
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk, word2idx

def process_data(data,word2idx,chunk_tags,onehot=False):
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    x = pad_sequences(x, para["max_len"],padding='post', truncating='post')  # left padding
    y_chunk = pad_sequences(y_chunk, para["max_len"], value=-1,padding='post', truncating='post')

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
        # print(y_chunk)
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def get_lengths(X):
    lengths = []
    for i in range(len(X)):
        length = 0
        for dim in X[i]:
            # print(dim)
            if dim != 0:
                length += 1
            else:
                break
        # print(length)
        lengths.append(length)

    return lengths

def create_bool_matrex(repre_dim,x):
    bool_x = numpy.zeros(shape=(x.shape[0], x.shape[1],repre_dim))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] != 0:
                bool_x[i,j,:] = 1.
    return bool_x

def load_embed_weight(word2id):
    embed_weight = numpy.zeros(shape=(len(word2id.keys())+1, para["embed_dim"]))
    char2vec = {}
    with open(para["embed_path"], "r") as f:
        rows = f.readlines()
        for row in rows:
            item = row.strip().split(" ", 1)
            char = item[0]
            # print(item)
            vec_str = item[1].split(" ")
            vec = [float(i) for i in vec_str]
            char2vec[char] = vec
    for word in word2id.keys():
        # print(word)
        vec = char2vec[word]
        embed_weight[word2id[word]] = numpy.array(vec)
    print(embed_weight)
    return embed_weight

def get_simple2traditional():
    simple2traditional = {}
    with open(config.traditional_dict_path,"r") as f:
        rows = f.readlines()
        for row in rows:
            item = row.strip().split("	")
            simple2traditional[item[0]] = item[1]
    return simple2traditional



if __name__ == "__main__":
    para["data_pk_path"] = "./cache/nlpcc-pos.pk"
    para["train_path"] = "./data/Pos_train.txt"
    para["test_path"] =  "./data/Pos_test.txt"
    train_test_set_preprocess()

