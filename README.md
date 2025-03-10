# uRank_uMart
Listwise Learning to Rank by Exploring Unique Ratings (accepted at ICDM 2020)

## Configuration
Set a system variable `RAW_RANK_DATA` that stores the folder that contains the raw ranking data

Set a system variable `TF_RANK_DATA` that stores the folder that contains the serialized ranking data

Convert raw ranking data to tf records
```bash
python prepare_data.py
```

Your `TF_RANK_DATA` should now contain the serialized tf records for running the experiments.


## params
src/experiments/base_model/params.json contains all of the hyper-parameters.

For instance, `num_learners` is used to control how many weak learners are used in uBoost and urBoost. 

`mlp_sizes` controls the number of MLP layers and the number of neurons per layer.

`residual_mlp_sizes` controls the MLP network parameters for the gradient boosting step.

`pooling` and `rnn` are used for urRank and urBoost only. `"pooling": "MP"` denotes max-pooling, and `"pooling": "AP"` denotes average-pooling. `"rnn": "C1"` generally works better than `"rnn": "C2"`.

`batch_size` needs to be 1, which indicates one query per batch or we put say all query-document features and labels that belong to the same query in the same batch.

## uRank
Set `num_learners` to 1. Change the GPU number accordingly.

```bash
./run_OHSUMED_uRank.sh
```

## uBoost
Set `num_learners` to an integer larger than 1.

```bash
./run_OHSUMED_uRank.sh
```

## urRank
This model was not mentioned in the paper. It is uRank + RNN without boosting.

```bash
./run_OHSUMED_urRank.sh
```

## urBoost
Set `num_learners` to an integer larger than 1.

```bash
./run_OHSUMED_urBoost.sh
```

## uMart
Set a system variable `LIGHTGBM_DATA` that stores the folder that contains the LightGBM ranking data

Convert the raw learning-to-rank data sets to LightGBM inputs.

```bash
python msltr2libsvm.py
```

Replace the rank_objective.hpp in this repo with the same file in LightGBM (https://github.com/microsoft/LightGBM) and compile to obtain the binary for training uMart.

## Notice
Please keep in mind that the default NDCG calculation in LightGBM takes queries with all 0 labels as NDCG any position 1, i.e., NDCG@1=1,  NDCG@3=1, NDCG@5=1, NDCG@10=1. 

The original NDCG script used for MQ2007 and MQ2008 data sets take such case as 0 (see mslr-eval-score-mslr-original.pl). 

Yahoo learning to rank dataset takes such case as 0.5. 

We remove queries with all 0 labels from all of the data sets to avoid this confusion. This was done during `python prepare_data.py`. 

We fixed the logic and added ERR calculation based on the perl script (see mslr-eval-score-mslr.pl).

We also implemented ranknet, listnet, listmle in this repo, however, it might not as efficient as TF-Ranking https://github.com/tensorflow/ranking. Other implementations can be found at https://github.com/microsoft/LightGBM and https://sourceforge.net/p/lemur/wiki/RankLib/.

## Citation
Please kindly cite our work if you would like to use our code.

Xiaofeng Zhu and Diego Klabjan. 2020. Listwise Learning to Rank by Explor-ing Unique Ratings. InThe Thirteenth ACM International Conference on WebSearch and Data Mining (WSDM ’20), February 3–7, 2020, Houston, TX, USA.WSDM, Houston, TX, USA.