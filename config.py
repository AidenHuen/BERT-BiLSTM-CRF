
para = {}

para["data_pk_path"] = "./data/pku-seg.pk"
para["fea_pk_path"] = "./data/pku-seg-fea.pk"

para["data_pk_path"] = "./cache/nlpcc-pos.pk"
para["train_path"] = "./data/Pos_train.txt"
para["test_path"] = "./data/Pos_test.txt"

para["model_path"] = "./model/pku/lstm-crf-embed-bert"


para["img_w"] = 50
para["img_h"] = 50
para["embed_dim"] = 200
para["unit_num"] = 200
para["split_seed"] = 2018
para["max_len"] = 142
para["EPOCHS"] = 40
para["batch_size"] = 20

para["traditional_chinese"] = False
para["sep"] = "\t"
para["char_dropout"] = 0.5
para["rnn_dropout"] = 0.5
para["lstm_unit"] = 300
para["REPRE_NUM"] = 128

para["fea_dropout"] = 0.3
para["fea_lstm_unit"] = 32
para["fea_dim"] = 20
para["radical_max"] = 7
para["pinyin_max"] = 8
para["rad_max"] = 1



