# EACL2021
This is the repository to reproduce results in the paper
"Hierarchical Multi-head Attentive Network for Evidence-aware Fake News Detection", EACL 2021.  
# Hierarchical Multi-head Attentive Network
![alt text](https://github.com/nguyenvo09/EACL2021/blob/main/examples/mac.png)

# Usage
## 1. Install required packages
We use Pytorch 0.4.1 and python 3.5. 
```
pip install requirements.txt
```
## 2.1 Running experiment for Snopes dataset (Table 2)
```
mkdir logs
python Masters/master_mac.py --dataset="Snopes" \
                             --cuda=1 \
                             --fixed_length_left=30 \
                             --fixed_length_right=100 \
                             --log="logs/mac" \
                             --loss_type="cross_entropy" \
                             --batch_size=32 \
                             --num_folds=5 \
                             --use_claim_source=0 \
                             --use_article_source=1 \
                             --num_att_heads_for_words=5 \
                             --num_att_heads_for_evds=2 \
                             --path="../formatted_data/declare/" \
                             --hidden_size=300 \
                             --epochs=100
```

## 2.2 Running experiment for PolitiFact dataset (Table 3)
```
mkdir logs
python Masters/master_mac.py --dataset="Politifact" \
                             --cuda=1 \
                             --fixed_length_left=30 \
                             --fixed_length_right=100 \
                             --log="logs/mac" \
                             --loss_type="cross_entropy" \
                             --batch_size=32 \
                             --num_folds=5 \
                             --use_claim_source=1 \
                             --use_article_source=1 \
                             --num_att_heads_for_words=3 \
                             --num_att_heads_for_evds=1 \
                             --path="../formatted_data/declare/" \
                             --hidden_size=300 \
                             --epochs=100
```

# Citation
If you feel our paper and resources are useful, please cite our work as follows:

```
@inproceedings{vo2021multihead,
	title={Hierarchical Multi-head Attentive Network for Evidence-aware Fake News Detection},
	author={Vo, Nguyen and Lee, Kyumin},
	booktitle={The 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021)},
	year={2021}
}
```