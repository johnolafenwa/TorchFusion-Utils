# TorchFusion-Utils
A pytorch helper library for Mixed Precision Training, Metrics and More Utilities to simplify training of deep learning models.

TorchFusion Utils was built to enable pytorch programmers easily take advantage of advanced training techniques without having to use any specific trainer framework. It is very transparent and can be easily plugged in to existing [Pytorch](https://pytorch.org) code bases

# Installation

TorchFusion Utils is extremely light with no other dependency other than [Pytorch](https://pytorch.org) itself. 
You can install from pypi

<pre> pip3 install torchfusion-utils --upgrade </pre>


# Core Features

**Mixed Precision Training**

In just two lines of code, you can speed up training of your [Pytorch](https://pytorch.org) models, reduce memory usage on any GPU and fit in larger batch sizes than was previously possible on your GPU.

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

Simple functions to easily analyse, load and save your [Pytorch](https://pytorch.org) models in an error free way.

# Documentation

Find tutorials and extensive documentation on using TorchFusion Utils from [https://utils.torchfusion.org](https://utils.torchfusion.org)


# About The TorchFusion Project

The TorchFusion project is a set of [Pytorch](https://pytorch.org) based deep learning libraries aimed at making making research easier and more productive. We believe anyone can be a great researcher with the right tools, thats why we build!

TorchFusion is an initiative of [DeepQuest AI](https://deepquestai.com), founded by John Olafenwa & Moses Olafenwa.

</pre>


<h3><b><u>Contact Developers</u></b></h3>
 <p>
  <br>
      <b>John Olafenwa</b> <br>
    <i>Email: </i>    <a style="text-decoration: none;"  href="mailto:johnolafenwa@gmail.com"> johnolafenwa@gmail.com</a> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://john.aicommons.science"> https://john.aicommons.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/johnolafenwa"> @johnolafenwa</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@johnolafenwa"> @johnolafenwa</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" href="https://facebook.com/olafenwajohn"> olafenwajohn</a> <br>

<br>
  <b>Moses Olafenwa</b> <br>
    <i>Email: </i>    <a style="text-decoration: none;"  href="mailto:guymodscientist@gmail.com"> guymodscientist@gmail.com</a> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://moses.aicommons.science"> https://moses.aicommons.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/OlafenwaMoses"> @OlafenwaMoses</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@guymodscientist"> @guymodscientist</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" target="_blank" href="https://facebook.com/moses.olafenwa"> moses.olafenwa</a> <br>
<br>
 </p>

 <br>
