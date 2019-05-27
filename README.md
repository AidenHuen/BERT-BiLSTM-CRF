# BERT-BiLSTM-CRF
BERT-BiLSTM-CRF的Keras版实现

## DEMO数据
- 2015词性标注数据集

## 文件描述
- preprocess.py 数据预处理，产生模型输入的pickle文件
- train.py 通过训练集，训练模型
- test.py 计算模型在测试集中的F1值

## 配置
- python 2.7
- tensorflow 1.10.0
- Keras 2.2.4
- keras-contrib 2.0.8
