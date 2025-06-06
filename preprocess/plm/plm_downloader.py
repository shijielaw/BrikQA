# !/usr/bin/env python
# -*- coding: UTF-8 -*-


from modelscope import snapshot_download


bert_dir = snapshot_download(model_id='deepset/sentence_bert', cache_dir='./')

print(bert_dir)
