# ColBERT
环境
```shell
#pip install -r requirements.txt
conda env create -f environment.yaml [-p /data/anaconda/env/vir_base] 
```
1训练
```shell
./eval.sh train
```
2. 语料库编码
```shell
./eval.sh index 
```
3. faiss建立ivfpq索引
```shell
./eval.sh faiss
```
4. 开启稠密检索服务
```shell
./eval.sh server 
```
5. 检索验证
```shell
./eval.sh eval
```
以上过程针对dureader中文数据集。\
模型非训练使用参数位于 **proj_conf/dense.yaml** \
加载参数代码位于 **colbert/utils/dense_conf.py** \
训练验证文件格式
```json
[{
  "question": "...",
  "positive_ctxs": [
    "...",
    "..."
  ],
  "hard_negative_ctxs": [
    "...",
    "..."
  ]
}]
```
数据集格式为
```json
[
  "...", 
  "..."
]
```
加载数据集代码在 **colbert/proj_utils/dureader_utils.py->get_dureader_ori_corpus()**，可以修改为自己的数据集

## 使用multi-view版本
* 参见论文 [Multi-View Document Representation Learning for Open-Domain Dense Retrieval
,  ACL 2022](https://arxiv.org/abs/2203.08372), 这里不使用lce loss
* 设置 **proj_conf/dense.yaml** 中 **enable_multiview=true**

