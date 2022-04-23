# DAPS-Rec

Deformable-Attentive Sequential Recommendation with Progressive Sampling

This is the pytorch implementation of our work DAPS-Rec.


# Dependencies
* NVIDIA GPU + CUDA 11.4
* Python 3.7 (Recommend to use Anaconda)
* PyTorch == 1.7.1
* tqdm==4.36.1
* pandas==0.25.0
* scipy==1.3.2
* numpy

## Datasets
We use four datasets in our paper.
All the datasets together with the corresponding experimental records have been uploaded to [Google Drive](https:)
 and [Baidu Netdisk](https:).

The downloaded dataset should be placed in the `Data` folder.

The downloaded records should be placed in the `experiments` folder for testing process.


## Train
Run `main.py` with arguments to train our model on with different templates. 
There are predefined templates for all models in `templates.py`.

```bash
python main.py --template DAPS_ml1m
python main.py --template DAPS_ml20m
python main.py --template DAPS_steam
python main.py --template DAPS_beauty
```

## Test
Run `test.py` to test our model with the provided checkpoints in `experiments`.

```bash
python test.py --template DAPS_ml1m
python test.py --template DAPS_ml20m
python test.py --template DAPS_steam
python test.py --template DAPS_beauty
```



## Output
```
Beauty:
{'Recall@100': 0.9967118275316456, 'NDCG@100': 0.310053890264487, 'Recall@50': 0.6794722634780256, 'NDCG@50': 0.25828952002751676, 'Recall@20': 0.3886166116859339, 'NDCG@20': 0.20157802736834635, 'Recall@10': 0.28032942814163014, 'NDCG@10': 0.17439645764571202, 'Recall@5': 0.2036500022381167, 'NDCG@5': 0.14976259985867935, 'Recall@1': 0.09393760486494136, 'NDCG@1': 0.09393760486494136}

ML-20m:
{'Recall@100': 0.9990035813308688, 'NDCG@100': 0.5820478228456212, 'Recall@50': 0.9505898603229558, 'NDCG@50': 0.5741052727271801, 'Recall@20': 0.8380074504342846, 'NDCG@20': 0.5515289520930892, 'Recall@10': 0.7324209475473203, 'NDCG@10': 0.524734989111613, 'Recall@5': 0.6188127235431988, 'NDCG@5': 0.48788511185681316, 'Recall@1': 0.34172146008547927, 'NDCG@1': 0.34172146008547927}

Steam:
{'Recall@100': 0.9917578125, 'NDCG@100': 0.3266360049356114, 'Recall@50': 0.7792767519300634, 'NDCG@50': 0.2920716289227659, 'Recall@20': 0.5121638257936998, 'NDCG@20': 0.23910763189196588, 'Recall@10': 0.353510044650598, 'NDCG@10': 0.1991739582405849, 'Recall@5': 0.23678672890771518, 'NDCG@5': 0.16161883136088198, 'Recall@1': 0.08565154897218401, 'NDCG@1': 0.08565154897218401}

ML-1m:
{'Recall@100': 0.998046875, 'NDCG@100': 0.54584393526117, 'Recall@50': 0.9119466145833334, 'NDCG@50': 0.5318786489466826, 'Recall@20': 0.7933485247194767, 'NDCG@20': 0.5081504378467798, 'Recall@10': 0.7047526041666666, 'NDCG@10': 0.48579067178070545, 'Recall@5': 0.5870768229166666, 'NDCG@5': 0.447698558991154, 'Recall@1': 0.2920464413861434, 'NDCG@1': 0.2920464413861434}
```



## License

The codes are currently for presentation to the reviewers only.
Please do not use the codes for other purposes.
<!-- ## MovieLens-1m
<img src=Images/ML1m-results.png> -->
