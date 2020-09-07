# Match_sum
NLP摘要生成，复现Extractive Summarization as Text Matching  

对ACL2020 论文 MatchSum的简单的复现，参考了原论文开源的一些代码实现，自定义了一些部分 

最后在DailyMail数据集上进行测试，Rouge值44.42

模型的使用方法：在对应的目录下直接运行MatchSum文件，在Linux下执行python3 Match_sum.py

Model.py中包含模型的实现，Dataloader则是对应数据的读取和准备
