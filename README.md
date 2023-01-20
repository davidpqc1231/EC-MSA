# EC-MSA
Implementation for Enhancing Multi-Source Consistency for Cross-Domain Learning

Here we use the Office-Home dataset.

#### Compatibility:
```
python: 3.7.7
pytorch: 1.5.1
torchvision: 0.6.1
numpy: 1.18.1
pandas: 1.0.5
matplotlib: 3.2.2
argparse: 1.1

GPU type: Nvidia GeForce GTX 1080
driver: 440.82
cuda: 10.2.89

```

#### Temporary code instructions:

Note: We are still working on improving this codes' implementation.

To run EC-MSA with the same settings as ours:

```
python main.py --cuda True --source1 <S1FILE> --source2 <S2FILE> --source3 <S3FILE> --target <TFILE>
```
S1FILE contains source1 dataset. S2FILE contains source2 dataset. S3FILE contains source3 dataset. TFILE contains target dataset. 

Thesis available on https://hammer.purdue.edu/articles/thesis/MULTI-SOURCE_AND_SOURCE-PRIVATE_CROSS-DOMAIN_LEARNING_FOR_VISUAL_RECOGNITION/19610010/1



