# This README explains how to apply the External Knowledge Code Gen model to the Concode dataset

See the main `README.md` for general information on External Knowledge Code Gen (EKCG).


## Prepare Environment
Use a python virtual Environment and initialize it with `requirements.txt`.

Some key dependencies and their versions are:
- astor=0.7.1 (This is very important)
- clang==14.0

Use python 3.7. Install with `pip install -e .`

## Preparing dataset and training


Look into `scripts/concode` and `data/concode`.


Run
```bash
python src/datasets/concode/dataset.py
```
This will transform the Concode dataset already in Conala format into the
binary format used for training.

By default things should be preprocessed and saved to `data/concode`. Check out
those `.bin` files.

############# Original README below. To adapt to Concode

### Pretraining

Check out the script `scripts/conala/train_retrieved_distsmpl.sh` for our best performing strategy. Under the directory you could find scripts for other strategies compared in the experiments as well.

Basically, you have to specify number of mined pairs (50k or 100k), retrieval method (`snippet_count100k_topk1_temp2`, etc.):
```
scripts/conala/train_retrieved_distsmpl.sh 100000 snippet_count100k_topk1_temp2
``` 
If anything goes wrong, make sure you have already preprocessed the corresponding dataset/strategy in the previous step.

The best model will be saved to `saved_models/conala`

### Finetuning

Check out the script `scripts/conala/finetune_retrieved_distsmpl.sh` for best performing finetuning on CoNaLa training dataset (clean).
The parameters are similar as above, number of mined pairs (50k or 100k), retrieval method (`snippet_count100k_topk1_temp2`, etc.), and additionally, the previous pretrained model path:
```
scripts/conala/finetune_retrieved_distsmpl.sh 100000 snippet_count100k_topk1_temp2 saved_models/conala/retdistsmpl.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.vocab.src_freq3.code_freq3.mined_100000.goldmine_snippet_count100k_topk1_temp2.bin.pre_100000_goldmine_snippet_count100k_topk1_temp2.bin.seed0.bin
``` 
For other strategies, modify accordingly and refer to other `finetune_xxx.sh` scripts.
The best model will also be saved to `saved_models/conala`.

### Reranking
Reranking is not the core part of this paper, please refer to [this branch](https://github.com/pcyin/tranX/tree/rerank) and [the paper](https://www.aclweb.org/anthology/P19-1447.pdf).
This is an orthogonal post-processing step.

In general, you will first need to obtain the decoded hypothesis list after beam-search of the train/dev/test set in CoNaLA, and train the reranking weight on it.

To obtain decodes, run `scripts/conala/decode.sh <train/dev/test_data_file> <model_file>`.
The outputs will be saved at `decodes/conala`

Then, train the reranker by `scripts/conala/rerank.sh <decode_file_prefix>.dev.bin.decode/.test.decode`

For easy use, we provide our trained reranker at `best_pretrained_models/reranker.conala.vocab.src_freq3.code_freq3.mined_100000.intent_count100k_topk1_temp5.bin`

### Test
This is easy, just run `scripts/conala/test.sh saved_models/conala/<model_name>.bin`

## Provided State-of-the-art Model
The best models are provided at `best_pretrained_models/` directories, including the neural model as well as trained reranker weights.

First, checkout our [online demo](http://moto.clab.cs.cmu.edu:8081/).

Second, we also provide an easy to use HTTP API for code generation.
### Web Server/HTTP API
To start the web server with our state-of-the-art model, simply run:

```
conda activate tranx
python server/app.py --config_file config/config_conala.json
```

The config file contains the path to our best models under `best_pretrained_models`.

This will start a web server at port 8081.

**HTTP API** To programmically query the model to get semantic parsing results, send your HTTP GET request to

```
http://<IP Address>:8081/parse/conala/<utterance>

# e.g., http://localhost:8081/parse/conala/reverse a list
```



## Reference
```
@inproceedings{xu20aclcodegen,
    title = {Incorporating External Knowledge through Pre-training for Natural Language to Code Generation},
    author = {Frank F. Xu and Zhengbao Jiang and Pengcheng Yin and Graham Neubig},
    booktitle = {Annual Conference of the Association for Computational Linguistics},
    year = {2020}
}
```


## Thanks
Most of the code for the underlying neural model is adapted from [TranX](https://github.com/pcyin/tranx) software, and the [CoNaLa challenge dataset](https://conala-corpus.github.io/).

We are also grateful to the following previous papers that inspire this work :P
```
@inproceedings{yin18emnlpdemo,
    title = {{TRANX}: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation},
    author = {Pengcheng Yin and Graham Neubig},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP) Demo Track},
    year = {2018}
}

@inproceedings{yin18acl,
    title = {Struct{VAE}: Tree-structured Latent Variable Models for Semi-supervised Semantic Parsing},
    author = {Pengcheng Yin and Chunting Zhou and Junxian He and Graham Neubig},
    booktitle = {The 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
    url = {https://arxiv.org/abs/1806.07832v1},
    year = {2018}
}

Abstract Syntax Networks for Code Generation and Semantic Parsing.
Maxim Rabinovich, Mitchell Stern, Dan Klein.
in Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2017

The Zephyr Abstract Syntax Description Language.
Daniel C. Wang, Andrew W. Appel, Jeff L. Korn, and Christopher S. Serra.
in Proceedings of the Conference on Domain-Specific Languages, 1997
```
