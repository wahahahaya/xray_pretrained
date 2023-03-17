# No more ImageNet pre-trained

## file

## Train
Simclr pre-trained model: run ```pre_train.py``` with the following args:
- ```--root```: the root of the dataset
- ```--data```: CheXpert with "chest", MURA with "mura", stl10 with "stl10"
- ```--arch```: the model we trained, we have "resnet18", "resnet50", and "resnet18_imgnet" three different models
- ```--lr```: laerning rate default 3e-4
- ```--batch_size```: batch size default 64

*** pre_train_simclr_v2.py #L22 need to change the path. the output folder have .log, .pth, and .event.
example:

```bash
python3 pre_train.py --data="chest" --lr=1e-4
```

Classification: run ```classifier.py``` with the following args:
```--epochs```: default 1000
```--batch_size```: default 32
```--lr```: default=3e-4
```--weight_decay```: default=8e-4
```--model```: "scratch", "ImageNet", and "simclr"
```--root```: the root of the dataset
```--data```: "mura", and "stl10

*** classifier.py #L75~L109 the default set is freeze all layer but the last

example:

```bash
python3 classifier.py --data="mura" --model="scratch"
```