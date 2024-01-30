##### README

所依赖的库

```bash
chardet==4.0.0
chardet==5.2.0
numpy==1.21.5
numpy==1.24.3
pandas==1.4.4
pandas==1.5.3
Pillow==9.2.0
Pillow==10.2.0
scikit_learn==1.0.2
torch==2.1.0
torch==2.0.1
torchvision==0.16.0
torchvision==0.16.1
tqdm==4.64.1
transformers==4.36.1

```

```bash
pip install -r requirments.txt
```

可以执行上述指令安装

```bash
│  README.md
│  requirements.txt
│  result.txt
│  tree.txt
│  实验五——多模态融合.md
│  实验五——多模态融合.pdf
│  
├─code
│  │  data_process.py
│  │  main.py
│  │  result.py
│  │  result_data.xlsx
│  │  test.csv
│  │  test_without_label.txt
│  │  train.csv
│  │  train.txt
│  │  train_train.csv
│  │  train_val.csv
│  │  
│  ├─bert-base-uncased    
│  ├─data
│  │     
│  └─results
│          result.csv
│          result.txt     
└─prtsc
```



代码保存在code里面

其中有train_val.csv，train_train.csv，test.csv，train.csv文件的基础下直接运行main.py程序即可

若没有上述csv文件则先运行data_process.py后再运行main.py

最后得到的结果保存在results文件夹中，为result.csv文件，运行result.py可以生成最后的result.txt文件