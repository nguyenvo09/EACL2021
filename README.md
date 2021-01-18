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
python MasterFC/master_mac.py --dataset="Snopes" \
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
or you can simply run the following script in your terminal
```
./run.sh
```

You can get following results in log:
```
|Epoch 000 | Train time: 16.1(s) | Train loss: 67.254| Val F1_macro = 0.659 | Vad AUC = 0.74845 | Val F1 = 0.49091 | Val F1_micro = 0.741 | Validation time: 05.7(s)
|Epoch 001 | Train time: 15.9(s) | Train loss: 44.168| Val F1_macro = 0.621 | Vad AUC = 0.80257 | Val F1 = 0.37975 | Val F1_micro = 0.774 | Validation time: 05.4(s)
|Epoch 002 | Train time: 15.9(s) | Train loss: 36.379| Val F1_macro = 0.735 | Vad AUC = 0.84374 | Val F1 = 0.60714 | Val F1_micro = 0.797 | Validation time: 05.6(s)
|Epoch 003 | Train time: 15.9(s) | Train loss: 29.171| Val F1_macro = 0.727 | Vad AUC = 0.85348 | Val F1 = 0.58768 | Val F1_micro = 0.799 | Validation time: 05.5(s)
|Epoch 004 | Train time: 15.8(s) | Train loss: 21.600| Val F1_macro = 0.744 | Vad AUC = 0.87069 | Val F1 = 0.61321 | Val F1_micro = 0.811 | Validation time: 05.4(s)
|Epoch 005 | Test AUC = 0.88417 | Testing time: 10.1(s)
|Epoch 005 | Train time: 15.9(s) | Train loss: 16.448| Val F1_macro = 0.746 | Vad AUC = 0.87330 | Val F1 = 0.60317 | Val F1_micro = 0.827 | Validation time: 05.5(s)
|Epoch 006 | Train time: 15.9(s) | Train loss: 13.176| Val F1_macro = 0.773 | Vad AUC = 0.86827 | Val F1 = 0.65728 | Val F1_micro = 0.831 | Validation time: 05.5(s)
|Epoch 007 | Train time: 16.0(s) | Train loss: 11.274| Val F1_macro = 0.766 | Vad AUC = 0.86449 | Val F1 = 0.64815 | Val F1_micro = 0.824 | Validation time: 05.5(s)
|Epoch 008 | Train time: 15.9(s) | Train loss: 10.475| Val F1_macro = 0.767 | Vad AUC = 0.87248 | Val F1 = 0.64390 | Val F1_micro = 0.831 | Validation time: 05.6(s)
|Epoch 009 | Train time: 15.8(s) | Train loss: 9.484| Val F1_macro = 0.708 | Vad AUC = 0.85913 | Val F1 = 0.54255 | Val F1_micro = 0.801 | Validation time: 05.6(s)
```
## 2.2 Running experiment for PolitiFact dataset (Table 3)
```
mkdir logs
python MasterFC/master_mac.py --dataset="Politifact" \
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

# An example of attention heads
![alt text](https://github.com/nguyenvo09/EACL2021/blob/main/examples/attention_heads.png)

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