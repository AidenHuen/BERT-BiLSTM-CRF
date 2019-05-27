# BERT-BiLSTM-CRF
BERT-BiLSTM-CRF的Keras版实现

## BERT配置
 1. 首先需要下载Pre-trained的BERT模型，本文用的是Google开源的中文BERT模型：
- https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
 2. 安装BERT客户端和服务器 pip install bert-serving-server pip install bert-serving-client，源项目如下：
- https://github.com/hanxiao/bert-as-service
 3. 打开服务器，在BERT根目录下，打开终端，输入命令：
- bert-serving-start -pooling_strategy NONE -max_seq_len 144 -mask_cls_sep -model_dir chinese_L-12_H-768_A-12/  -num_worker 1

## DEMO数据
- 2015词性标注数据集

## 文件描述
- preprocess.py 数据预处理，产生模型输入的pickle文件
- train.py 通过训练集，训练模型
- test.py 计算模型在测试集中的F1值
- Modellib.py 模型位置
- config.py 参数配置
## 模型训练
配置BERT->>执行preprocess.py->>执行train.py

## 配置
- python 2.7
- tensorflow-gpu 1.10.0
- Keras 2.2.4
- keras-contrib 2.0.8
