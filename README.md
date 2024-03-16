## Single-head Single-Layer Retentive Network Encorder Block

```
cd Memoristive_RetNet

python main.py --model transformer --dataset mnist --pos True

python main.py --model retnet --dataset mnist
```

Hyper parameters:

RetNet(num_layer = 1, num_head = 1, num_sequence = 64, num_feature = 48, args)

Transformer(num_layer = 1, num_head = 1, num_sequence = 64, num_feature = 48, args)
