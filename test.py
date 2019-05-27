
import numpy
from preprocess import *
from preprocess import get_lengths
import ModelLib
import config
import pickle
import datetime

para = config.para
train_x, train_y, val_x, val_y, word2id, tags = pickle.load(open(para["data_pk_path"], 'rb'))

def predict_bert(para):
    para['tag_num'] = len(tags)
    model = ModelLib.BERT_MODEL(para)
    model.load_weights(filepath=para["model_path"])
    bert_val =load_path_bert(para["test_path"],sep=para["sep"])
    lengths = get_lengths(val_x)

    pred_y = model.predict(bert_val)

    tag_pred_y = []
    tag_val_y = []
    for i, y in enumerate(pred_y):
        y = [numpy.argmax(dim) for dim in y]
        print(lengths[i])
        p_y = y[:lengths[i]]
        print(p_y)
        v_y = val_y[i][:lengths[i]].flatten()
        print(v_y)
        p_y = [tags[dim] for dim in p_y]
        v_y = [tags[dim] for dim in v_y]
        tag_pred_y.append(p_y)
        tag_val_y.append(v_y)
    return tag_pred_y,tag_val_y

def char_seg_acc(tag_pred_y, tag_val_y):
    acc = 0.0
    num = 0.0
    for j in range(len(tag_pred_y)):
        for z in range(len(tag_pred_y[j])):
            if tag_pred_y[j][z] == tag_val_y[j][z]:
                acc+=1
            num += 1
    print("test acc:"+str(acc/num))

def word_seg_F1(y_pred,y):
    c = 0
    true = 0
    pos = 0
    for i in xrange(len(y)):
        start = 0
        for j in xrange(len(y[i])):
            if y_pred[i][j] == 'E' or y_pred[i][j] == 'S':
                pos += 1
            if y[i][j] == 'E' or y[i][j] == 'S':
                flag = True
                if y_pred[i][j] != y[i][j]:
                    flag = False
                if flag:
                    for k in range(start, j):
                        if y_pred[i][k] != y[i][k]:
                            flag = False
                            break
                    if flag:
                        c += 1
                true += 1
                start = j+1

    P = c/float(pos)
    R = c/float(true)
    F = 2*P*R/(P+R)
    return P,R,F

def pos_F1(y_pred, y):
    c = 0
    true = 0
    pos = 0
    for i in xrange(len(y)):
        start = 0
        for j in xrange(len(y[i])):
            # print y_pred[i][j]
            if y_pred[i][j][0] == 'E' or y_pred[i][j][0] == 'S':
                pos += 1
            if y[i][j][0] == 'E' or y[i][j][0] == 'S':
                flag = True
                if y_pred[i][j] != y[i][j]:
                    flag = False
                if flag:
                    for k in range(start, j):
                        if y_pred[i][k] != y[i][k]:
                            flag = False
                            break
                    if flag:
                        c += 1
                true += 1
                start = j+1
    try:
        P = c/float(pos)
        # print pos
        R = c/float(true)
        # print true
        F = 2*P*R/(P+R)
    except Exception, e:
        print e
    return P, R, F

if __name__ == "__main__":
    para["char_dropout"] = 0.5
    para["rnn_dropout"] = 0.5

    para["model_path"] = "./model/lstm-crf-bert"
    pred_y, val_y = predict_bert(para)
    # pred_y, val_y = predict_normal(para, use_embed=False,feature="")
    P,R,F = pos_F1(pred_y,val_y)
    # P, R, F = word_seg_F1(pred_y,val_y)
    print("P:"+str(P))
    print("R:"+str(R))
    print("F1:"+str(F))
