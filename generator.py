from keras_preprocessing.sequence import pad_sequences
import pickle
from preprocess import _parse_data,get_tag,Counter
import codecs
import config
from bert_serving.client import BertClient
import numpy as np

para = config.para

def make_batches( size, batch_size):

    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1) * batch_size)) for i in range(0, nb_batch)]

def bert_generator(batch_size,train_path,sep,y,Shuffle = True):
    index_array = np.arange(y.shape[0])
    if Shuffle:
        np.random.shuffle(index_array)

    data = _parse_data(codecs.open(train_path, 'r'), sep=sep)
    data = [[items[0] for items in sent] for sent in data]
    bc = BertClient()  # ip address of the GPU machine
    # step = int(data.__len__() / batch_size) + 1
    batches = make_batches(y.shape[0] - 1, batch_size)
    while 1:
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            data_batch = [data[id] for id in batch_ids]
            # print(data_batch)
            x_batch = bc.encode(data_batch,is_tokenized=True)
            x_batch = x_batch[:, 1:para["max_len"] + 1]
            y_batch = y[batch_ids]
            yield (x_batch,y_batch)

if __name__ == "__main__":
    # train_path = config.fold_path + "PKU/train.txt"
    # test_path = config.fold_path + "PKU/test.txt"
    # sep = "    "
    # train_x, train_y, val_x, val_y, word2id, tags, img_embed = pickle.load(open("./data/pku-seg.pk", 'rb'))
    # for x,y in bert_generator(64, train_path, sep, train_y):
    #     print x.shape,y.shape
    data = make_batches(10000,64)
    print(data)