# README

## NOW-I-DO
Now, there are four *.py files under tesing, which are:
1. cifar10.py, use VGG11 to train the CIFAR-10 datasets from the beginning. Can plot the fig, but can't re-train the model. The training process is really slow, it costs nearly 600s with 4 GPUs, 8 Cores, and 64 batches.
2. fiinetune_bee.py, use the pre-trained model of SqueezeNet in the hymenoptera dataset. Just re-train the last Linear Layer, so it's really fast.
3. simple_cnn.py, use the basic net structure in CIFAR-10.
4. complex_cnn.py, use a comple model to train CIFAR-10, but got stuck, and the loss didn't drop.



the simple models only get 60%, and the VGG11 only get 89%.

fine-tune with the SqueezeNet trained in ImageNet, but only modify the last layer, it soon gets the best correct rate and remains still.

The fine-tuned model of SqueezeNet only has 5M, but the one of VGG11 has nearly 500M. 



How to use the optimizer? Adam or SGD, how to adjust the parameters? Find the connection between parameters adjustment and the accuracy variety.



- [ ] try to get 95%+ accuracy in CIFAR-10 dataset
- [ ] try to train in MNIST dataset
- [ ] Re-arrange the codes
- [ ] Analyze the models saved as *\*.pth*




## File structure
By default, each task hold one single folder:
> TASK_NAME \_\_\_\_\_ TaskName_Detail.py
> ​	   |\_\_\_\_\_\_\_\_\_\_ ./data   :load the datasets here.
> ​           |\_\_\_\_\_\_\_\_\_\_ ./fig    :save the figs for visulization.
> ​           |\_\_\_\_\_\_\_\_\_\_ ./model  :save the *.pth file, and save the information of the model, such as EPOCH, BATCH, correct rate, and optimizer, etc.
> ​           |\_\_\_\_\_\_\_\_\_\_ other files, README.md, and useful *.py files.

Moreover, the *.py files can be seperated by its function, such as load model, save model, save statistics, draw fig, define CNN model ...

All the files should be update by Git, so I can use the files both the Server and the Mac. Use Github is better in the beginning. And test the .gitignore file to filter the useless folders.

> main  \_\_\_\_\_\_  network.py
>
> ​    |\_\_\_\_\_\_\_\_\_  train & test
>
> ​    |\_\_\_\_\_\_\_\_\_  visualization & save
>
> main file, named as `TASK_MODEL_DETAIL`, for example, `cifar10_vgg11_finetune`

To save the whole work, need to save the model, the parameters and the log. The log should describe the details of this work, such as optimizer information, best accuracy, epoch num and other things, a .txt file is enough.



## Attention

DO NOT use Chinese characters in this repo.

use torch.load( .pth ) to load the model and get a dict about this model. The keys are like this,

```txt
features.0.weight
features.0.bias
features.3.squeeze.weight
features.3.squeeze.bias
features.3.expand1x1.weight
features.3.expand1x1.bias
```

so, we just to make sure we save and load with the same structure.

To save the work, we should both save the models and the parameters.