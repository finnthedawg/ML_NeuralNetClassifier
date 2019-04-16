# Multiclass Classification of 5000 handwritten digits.

**1. Creating and training a 2 layer ANN in pytorch**

**2. Training the network with Pre-trained features**

**3. Comparison with a Multiclass regularized logistic regression strategy**


---
Build instructions:
```
git clone
jupyter notebook ANNclassifier.ipynb
```

## Creating and training a 2 layer ANN in pytorch.

It was determined from the data that the dataset consists of handwritten (MNIST) data. This consists of `5000` `20x20` images with a few examples shown below:

<p align="center">
  <img width="600"  src="./digits.png">
</p>

With this dataset and classification task, I used a small two layer ANN consisting of:
```
fc1 = nn.Linear(400,25)
fc2 = nn.Linear(25,10)

#A fully connected layer using Sigmoid activication
x = Sigmoid(fc1(x))
#Another fully connected layer and apply Softmax
x = Softmax(fc2(x))
```

Since the total size of the Dataset is small, I used a Stochastic Gradient Descentover each image for each epoch rather than for a batch of images. The parameters that were used:

```
epochs = 20
test_size = 30%
SGD learning rate = 0.2
momentum = 0
dampening = 0
loss = MSE loss
```

Before training, the accuracy of the network was `10.34%` which is as accurate as random guessing. After training, over `20 epochs`, our accuracy reached `92.2%` which is resonable considering the dataset contains numbers that even humans have difficulty detecting e.g the 2 in figure 1. Below are figures of our training process across epochs.

<p align="center">
  <img width="350"  src="./accuracyANN.png"><img width="350"  src="./lossANN.png">
</p>
