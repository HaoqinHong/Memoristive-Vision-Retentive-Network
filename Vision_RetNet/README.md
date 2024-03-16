# Retentive Vision Transformer

Run commands: `<br>`

python main.py --dset mnist `<br>`

python main.py --dset mnist  --pos True --pooling True  `<br>`
python main_retnet.py --dset fmnist `<br>`

python main_retnet.py --dset cifar10 `<br>`

python main_retnet.py --dset mnist `<br>`
python main_retnet.py --dset fmnist `<br>`

python main_retnet.py --dset cifar10 `<br>`

`<br><br>`
Transformer Config:

| `<!-- -->`       | `<!-- -->` |
| ------------------ | ------------ |
| Input Size         | 28           |
| Patch Size         | 4            |
| Sequence Length    | 7*7 = 49     |
| Embedding Size     | 96           |
| Num of Layers      | 6            |
| Num of Heads       | 4            |
| Forward Multiplier | 2            |
