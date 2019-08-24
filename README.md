# DIB-PEB-Sequential-RS
A tensorflow implementation of the  paper "Dynamic Item Block and Prediction Enhancing Block for Sequential Recommendation" in IJCAI 2019

需求环境：
---------
python3.6，tensorflow

1.paper原文
------
  在Github中直接可以下载，文件名为Dynamic Item Block and Prediction Enhancing Block for Sequential Recommendation.pdf
  
2.如何获取训练数据集？
------
  自带了ml-100k数据集，存放在dataset/processed_datasets/ml-100k<br>
  还可以通过以下方式获得其余三个数据集<br>
  百度网盘地址：https://pan.baidu.com/s/1X7G3dVEJ0JN7yppcKLY4pA<br>
  下载后解压在dataset/processed_datasets目录下即可<br>


3.如何训练模型
------
  直接执行RUM-Ksoft-mulcha-test.py即可<br>
  以下文件：<br>
  caser-test.py <br>
  fpmc-test.py <br>
  gru4rec-test.py <br>
  rum-i-test.py<br>
  为paper中所选baseline，也可以直接运行<br>
  
4参数说明
------
'learnRate': 0.002,             学习速率<br>
'maxIter': 2000,                迭代次数<br>
'trainBatchSize': 512,          训练batch_size<br>
'testBatchSize': 512,           测试batch_size <br>
'numFactor': 128,               item和user embedding的size<br>
'cell_numbers': 128,            如果选用GRU为中间网络结构，cell的数量<br>
'topN': 10,                     test时选TopN进行评测<br>
'gru_model': False,             设置为True则中间网络结构为GRU，设置为False则为memory network<br>
'decrease soft': True,          设置为False，损失函数为softmax + cross entropy，设置为True，则采用其他loss<br>
'loss_type': 'PEB',             在decrease soft为True情况下有效，可选择top1，bpr，neg，PEB<br>
'negative_numbers': 25,         负样本数量<br>
'eval_item_num': 1000,          train阶段抽取多少item作为一个评价子集，当数据集为ml-100k时，设置为500左右比较合适<br>
'numK': 15,                     PEB总共计算了K次概率分布<br>
'save_path': 'saved_model',<br>
'save_model': True,<br>
'load_model': False,<br>

for fileName in ['newkin-seq']:   括号中为训练所用数据集，可设置为'newkin-seq','ml-100k','cd-seq','movies_tv-seq'
