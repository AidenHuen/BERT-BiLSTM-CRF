# encoding:utf-8
from keras.callbacks import ModelCheckpoint
from preprocess import *
from generator import bert_generator
import ModelLib
import config

para = config.para
train_x, train_y, val_x, val_y, word2id, tags = pickle.load(open(para["data_pk_path"], 'rb'))

def train_bert_model(para, use_generator = False):
    para['tag_num'] = len(tags)
    model = ModelLib.BERT_MODEL(para)
    checkpoint = ModelCheckpoint(para["model_path"], monitor='val_viterbi_acc', verbose=1,
                                 save_best_only=True, mode='max')

    if use_generator:
        val_bert = load_path_bert(para["test_path"], para["sep"])
        model.fit_generator(bert_generator(para["batch_size"], para["train_path"], para["sep"], train_y,Shuffle=True), steps_per_epoch=int(train_y.shape[0]/para["batch_size"])+1, callbacks=[checkpoint],
                    validation_data=(val_bert, val_y), epochs=para["EPOCHS"],verbose=1)
    else:
        train_bert, val_bert = load_bert_repre()
        model.fit(train_bert, train_y, batch_size=para["batch_size"], epochs=para["EPOCHS"], callbacks=[checkpoint],
                  validation_data=(val_bert, val_y), shuffle=True,verbose=1)


if __name__ == "__main__":

    para["char_dropout"] = 0.5
    para["rnn_dropout"] = 0.5
    para["model_path"] = "./model/bert-lstm-crf"
    train_bert_model(para, use_generator=False)
