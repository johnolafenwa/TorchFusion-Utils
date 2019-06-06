# TorchFusion-Utils
A pytorch helper library for Mixed Precision Training, Metrics and More Utilities to simplify training of deep learning models.

TorchFusion Utils was built to enable pytorch programmers easily take advantage of advanced training techniques without having to use any specific trainer framework. It is very transparent and can be easily plugged in to existing pytorch code bases.

# Core Features

**Mixed Precision Training**

In just two lines of code, you can speed up training of your pytorch models, reduce memory usage on any GPU and fit in larger batch sizes than was previously possible on your GPU.

<pre>#convert your model and optimizer to mixed precision mode
model, optim = convertToFP16(model,optim)

#in your batch loop, replace loss.backward with optim.backward(loss)
optim.backward(loss)
</pre>

**Initialization**

A very simple api to easily initialize your model parameters with fine grained control over the type of layers and type of weights to be initialized.

<pre>
kaiming_normal_init(model,types=[nn.Conv2d],category="weight")
</pre>

**Metrics**

An extensible metric package that makes it easy to easily compute accuracy of your models. A number of metrics are provided out of the box and you can extend to add yours.

<pre>
top5_acc = Accuracy(topk=5)

#sample evaluation loop
for i,(x,y) in enumerate(data_loader):
    predictions = model(x)
    top5_acc.update(predictions,y)

print("Top 5 Acc: ",top5_accc.getValue())
</pre>

**Model Utilities**

Simple functions to easily analyse, load and save your pytorch models in an error free way.


**Installation**

TorchFusion Utils is extremely light with no other dependency other than pytorch itself. 
You can install from pypi

<pre> pip3 install torchfusion-utils </pre>



