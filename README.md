# No more ImageNet pre-trained

## file

## Train
Simclr pre-trained model: run ```pre_train.py``` with the following args:
- ```--root```: the root of the dataset
- ```--data```: CheXpert with "chest", MURA with "mura", stl10 with "stl10"
- ```--arch```: the model we trained, we have "resnet18", "resnet50", and "resnet18_imgnet" three different models
- ```--lr```: laerning rate defult 3e-4
- ```--batch_size```: batch size defult 64

*** pre_train_simclr_v2.py #L22 need to change the path. the output folder have .log, .pth, and .event.
example:
```python3 pre_train.py --data="chest" --lr=1e-4```
