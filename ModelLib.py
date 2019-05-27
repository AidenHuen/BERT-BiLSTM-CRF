import keras
import numpy
from keras import Model
from keras.layers import Embedding, Bidirectional, LSTM, \
    BatchNormalization, Dropout, Reshape, Conv2D, \
    Masking, MaxPooling2D, MaxPooling1D
from keras.layers import Input, Dense, Concatenate, TimeDistributed,Permute,RepeatVector, Multiply
from keras_contrib.layers import CRF
import MyLayer
from keras.layers.core import *

def BERT_MODEL(para):
    # for key in para:
    #     print key,para[key]
    bert_input = Input(shape=(para["max_len"], 768,), dtype='float32', name='bert_input')
    mask = Masking()(bert_input)
    repre = Dropout(para["char_dropout"])(mask)
    repre = Dense(300, activation="relu")(repre)
    repre = Bidirectional(LSTM(para["lstm_unit"], return_sequences=True, dropout=para["rnn_dropout"]))(repre)
    crf = CRF(para["tag_num"], sparse_target=True)
    crf_output = crf(repre)
    model = Model(input=bert_input, output=crf_output)
    model.summary()
    # adam_0 = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile("adam", loss=crf.loss_function, metrics=[crf.accuracy])
    return model


if __name__ == "__main__":
    data = numpy.ones((10, 10, 10), dtype='float32')
    # data_2 = numpy.ones((10, 100),dtype='float32')
    #
    # data_input = Input(shape=(100,))
    # data_reshape = Reshape(target_shape=(10, 10))(data_input)
    # weight_input = Input(shape=(10, 10))
    # # model.add(Embedding(input_dim=3,output_dim=5,weights=[weight],mask_zero=True))
    # output = Multiply(output_dim=(10,10))([data_reshape, weight_input])
    #
    # model = Model(input=[data_input,weight_input], output=output)
    # result = model.predict([data_2, data], batch_size=2)
    # model.summary()

    # train_x, train_y, val_x, val_y, word2id, tags, img_voc = pickle.load(open(config.data_pk, 'rb'))
    #
    # img_embed = load_img_embed(word2id)
    # x = numpy.array([1.0, 2.0, 3.0])
    # # x = x.astype(numpy.int64)
    # # result = tf.nn.embedding_lookup(img_embed, x)
    # # sess = tf.Session()
    # # result = sess.run(result)
    # # print(result)
    #
    # x_input = Input(shape=(train_x[0].shape[1],), dtype="int64")
    # y = MyLayer.ImageEmbeding(img_weight=img_embed, output_dim=(50, 50, 1))(x_input)
    # model = Model(input=x_input, output=y)
    # model.summary()
    # result = model.predict(train_x[0], batch_size=64)
    # print(result.shape)
