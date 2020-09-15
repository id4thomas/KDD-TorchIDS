# KDD TorchIDS
Unsupervised Intrusion Detection System PyTorch Implementation of KDD Dataset (10pct)
Forked from https://github.com/kabbi159/NSL-KDD-ADS

## Data Preprocessing
Dataset is processed with script from https://github.com/id4thomas/KDD-IDS

## Autoencoder Based Models
* AE (Autoencoder)
* VAE (Variational Autoencoder)
* DSEBM (Deep Structed Energy Based Model)
* DAGMM
* MEMAE (Memory-augmented Deep Autoencoder)

### DSEBM (Deep Structed Energy Based Models for Anomaly Detection) [[paper](https://arxiv.org/abs/1605.07717)] - ICML 2016   
To be implemented

### DAGMM (Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection) [[paper](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)] - ICLR 2018
I followed the experimental details and hyperparmeters from the paper (KDDCup-Rev setting) except the input shape of the encoder and the output shape of the encoder. (It depends on the data preprocessing)   
I referenced re-implementation of DAGMM with PyTorch ([code](https://github.com/danieltan07/dagmm)). It really helps me to re-implement. Thanks!   
In my attempt, NSL-KDD dataset shows weak performance than KDDCup99. I think I need to find best hyperparmeters for this dataset.

### MEMAE (Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection) [[paper](https://arxiv.org/abs/1904.02639)] - ICCV 2019    
I followed the experimental details and hyperparmeters from the paper in **4.3 Experiments on Cybersecurity Data** except the input shape of the encoder and the output shape of the decoder. (It depends on the data preprocessing)   
Also, I referenced author's code ([3d-conv MEMAE](https://github.com/donggong1/memae-anomaly-detection)). This implementation is about fc-MEMAE.   


## Usage
I edited my code so that multiple models (now, only two models) can be used as "main.py".   
model list : dagmm, memae   
You can run this code:
```bash
python main.py \
--batch_size=64 \
--seed=42 \
--lr=0.0001 \
--epoch=5 \
--model=memae
```

