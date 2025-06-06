# !/usr/bin/env python
# -*- coding: UTF-8 -*-


from modelscope import snapshot_download


llm_dir = snapshot_download(model_id='Qwen/Qwen3-8B-Base', cache_dir='./')

print(llm_dir)
