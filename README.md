# No more ImageNet pre-trained

## TODO
- [] add lr step (stable the output scalar)
- [] increase x-aray dataset (without shoulder)
- [] add labeled information in the simclr pre-trained

argument
- 100K unlabeled image vs Imagenet labeled image (100K labeled natural image will be better than x-ray pretrained?)
- freeze layer vs fine tune (make sure the training necessary: freeze, regulize, weight decay…)
    - data distribution [model embedding] (chest x-ray, shoulder x-ray, nature image)


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