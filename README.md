# README

## NOW-I-DO
Now, there are four *.py files under tesing, which are:
1. cifar10.py, use VGG11 to train the CIFAR-10 datasets from the beginning. Can plot the fig, but can't re-train the model. The training process is really slow, it costs nearly 600s with 4 GPUs, 8 Cores, and 64 batches.
2. fiinetune_bee.py, use the pre-trained model of SqueezeNet in the hymenoptera dataset. Just re-train the last Linear Layer, so it's really fast.
3. simple_cnn.py, use the basic net structure in CIFAR-10.
4. complex_cnn.py, use a comple model to train CIFAR-10, but got stuck, and the loss didn't drop.


## File structure
By default, each task hold one single folder:
> TASK_NAME ______ TaskName_Detail.py
>            |____ ./data   :load the datasets here.
>            |____ ./fig    :save the figs for visulization.
>            |____ ./model  :save the *.pth file, and save the information of the model, such as EPOCH, BATCH, correct rate, and optimizer, etc.
>            |____ other files, README.md, and useful *.py files.

Moreover, the *.py files can be seperated by its function, such as load model, save model, save statistics, draw fig, define CNN model ...

All the files should be update by Git, so I can use the files both the Server and the Mac. Use Github is better in the beginning.


