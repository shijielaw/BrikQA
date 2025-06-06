# Data Preprocess



## Environment Requirement

The required packages are as follows:

```
langchain==0.3.25
langchain_openai==0.3.19
modelscope==1.26.0
numpy==1.23.1
sentence_transformers==3.2.1
```



## Download PLM

```
cd ./plm
python plm_downloader.py
```



## Process Data

```
python question_atomizer.py
python ent_re_matcher.py
python subgraph_retrieval.py
```


