# BriKQA: Bridging the Gap between Knowledge Graphs and LLMs for Multi-hop Question Answering

This is our PyTorch implementation for 'Bridging the Gap between Knowledge Graphs and LLMs for Multi-hop Question Answering' accepted in CIKM 2025.



## Environment Requirement

The code has been tested running under Python 3.10.16 on Linux. 

The required packages are as follows:

```
datasets==3.6.0
peft==0.15.2
torch==2.7.0
tqdm==4.67.1
transformers==4.51.3
wandb==0.19.11
modelscope==1.26.0
```



## Download Models

```
cd ./llm
python llm_downloader.py
```



## KG Embedding

```
cd ./encoder
python main.py --data MovieQA --model TransE --max_steps 5000 --batch_size 512
```



## Run Model

```
python main.py --data MovieQA --num_epoches 30 --lora_rank 64 --batch_size 8 --train_encoder_model TransE --train_encoder_steps 5000 
```





Data Preprocess is in ./preprocess, please read ./preprocess/README.md to continue.





## Citation

```
@inproceedings{luo2025brikqa,
  title={Bridging the Gap between Knowledge Graphs and LLMs for Multi-hop Question Answering},
  author={Luo, Shijie and Lu, Xinyuan and Zhao, Qinpei and Rao, Weixiong},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25')},
  publisher={Association for Computing Machinery},
  address={Seoul, Republic of Korea},
  pages={5006--5010},
  year={2025}
}
```



