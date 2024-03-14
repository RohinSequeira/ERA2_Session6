# Part 1: Backpropagation in Microsoft Excel
Backpropagation of a Dense Network, implemented in MS Excel

## Fully Connected Network: The fundamental form of Artificial Neural Network
Artificial Neural Networks (ANN) are a class of computing models that are designed to mimic the working of a human brain. The most fundamental implementation of ANN is a Fully Connected Network. The FC network consists of multiple layers viz. a. Input layer - where the the independent variables required for modelling are passed as input, b. Hidden layers (one or more) which work on the input variables to extract features, and finally c. Output layer that gives the prediction of the model.

Each layer consists of a set of nodes, known as "Neurons". Each neurons in any given layer is connected to the entire set of neurons in the following layer and hence the name - "Fully Connected Network". Consider a layer containing "n" neurons - each of the "n" neurons in this layer establishes connections to all the "m" neurons in the next layer, resulting in a total of "n * m" connections. Each connections is assigned a weight, that decides how much of the output from the neuron in the preceeding layer (Layer "l-1") should pass to the next layer (Layer "l"). All the incoming connections to a neuron are then summed up, resulting in a weighted sum of the inputs. An optional bias term is added to this weighted sum for each neuron, but bias term is not used in this example. An activation is applied on this weighted sum, which is called the "Activation Output" of the neuron. The activations that are most commonly used are ReLU (Rectified Linear Unit), Sigmoid and Tanh. The activation function adds non-linearity to the netowrk which otherwise will just be a linear combination of weights and inputs, and the whole network will effective collapse to a single linear function without any activation.

Figure 1 below illustrates a simple Fully Connected Network, which has 1 input layer, 1 hidden layer and 1 output layer. There are 2 inputs (i1, i2) in the input layer, which is then connected to a hidden layer with 2 neurons. Each of these 2 neurons in the hidden layer (h1, h2) receive a weighted sum of the input values from the input layer. These values undergo activation (a_h1, a_h2) and are then further fully connected as weighted sum to the neurons in the output layer (o1, o2). The output layer then applies an activation (Sigmoid, in this case) and gives out the final predictions (a_o1, a_o2).

h1 = w1 * i1 + w2 * i2  
h2 = w3 * i1 + w4 * i2  
a_h1 = σ(h1) = 1/(1+exp(-h1))  
a_h2 = σ(h2) = 1/(1+exp(-h2))  
o1 = w5 * a_h1 + w6 * a_h2  
o2 = w7 * a_h1 + w8 * a_h2  
a_o1 = σ(o1) = 1/(1+exp(-o1))  
a_o2 = σ(o2) = 1/(1+exp(-o2))


![image](https://user-images.githubusercontent.com/71654199/119672619-5fe25500-be58-11eb-82da-f705dd3f9eab.png)
> Figure 1: An example of a Fully Connected Network

## Training the Neural Network

THe objective of training a neural network is to find out the set of weights that will minimize the error in the final prediction as much as possible for the given set of input values. 

The predictions a_o1, a_o2 of the Neural Network for the given input values i1, i2 are compared with the actual target values t1, t2. In this example, we use the "Mean Squared Error" as the function to quantify the loss (difference between actual and predicted values).

Loss / "Mean Squared Error" from output a_o1 = E1 = (1/2) * (t1 - a_o1)^2  
Loss / "Mean Squared Error" from output a_o2 = E2 = (1/2) * (t2 - a_o2)^2

Total Loss = E1 + E2

The weights of the network connections are randomly initialized as a first step of the training process. The initial error and loss are calculated based on the final output given by the network using the randomly initialized weights. The training process involves making gradual updates to the weights and finiding the best combination of values that give an output that is as close as possible to the actual output. In other words, we need to find the values for the weights such that global minima of the Loss Function, which is now a function of the weights of the network, is achieved.

![image](https://user-images.githubusercontent.com/71654199/120009815-044fcd00-bffa-11eb-9a4a-f4e8e6642777.png)
> Figure 2: A sample plot of the loss function with respect to the weights

## Backpropagation

The updates to the weights is done through a process called Backpropagation where in we calculate the partial derivative of the total error is calculated with respect to each weight. The partial derivative gives the gradient or slope at the current point of the loss. This acts as an indicator as how much to increase or decrease the weight to move towards the global minima.

To calculate the gradinet of the total error (E) with respect to a weight (say w5), chain rule can be applied i.e. the gradient of "E" w.r.t. w5 is calculated by multiplying the gradients that are at each step of the network, as illustrated below.

∂E_total/∂w5 = ∂(E1+E2)/∂w5 = ∂E1/∂w5 (w5 has no effect of E2)
∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5  
∂E1/∂a_o1 = ∂((1/2) * (t1 - a_o1)2)/∂a_o1 = (a_o1 - t1)  
∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = σ(o1) * (1- σ(o1)) = a_o1 * (1 - a_o1)
∂o1/∂w5 = ∂(w5 * a_h1 + w6 * a_h2)/∂w5 = a_h1  

Hence,  
∂E_total/∂w5 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h1  

Similarly,  
∂E_total/∂w6 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h2  
∂E_total/∂w7 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h1  
∂E_total/∂w8 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h2

Now to calculate the gradients in the previous layer, we apply the chain rule further -

∂E_total/∂w1 = ∂(E1+E2)/∂w1 =  = ∂E1/∂w1+∂E2/∂w1  

∂E1/∂w1 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1  
∂o1/∂a_h1 =  ∂(w5 * a_h1 + w6 * a_h2)/∂a_h1 = w5  
∂a_h1/∂h1 = ∂(σ(h1))/∂h1 =  σ(h1) * (1- σ(h1)) = a_h1 * (1 - a_h1)  
∂h1/∂w1 = ∂(w1 * i1 + w2 * i2)/∂w1 = i1  

Hence,  
∂E1/∂w1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5 * a_h1 * (1 - a_h1) * i1  
and,  
∂E2/∂w1 = (a_o2 - t2) * a_o2 * (1 - a_o2) * w7 * a_h1 * (1 - a_h1) * i1

Therefore,  
∂E_total/∂w1 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1  

Similarly,  
∂E_total/∂w2 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2  
∂E_total/∂w3 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1  
∂E_total/∂w4 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2

As the gradients are propagated from the last layer backwards until the first layer, the process is named "Backpropagation".

Once we calculate the gradients, these need to be subtracted from the initial weights and next set of predictions as well as the errors are calculated. But the updates to the weights can't be applied directly. The gradients are factored by a hyperparameter called Learning Rate and then reduced from the initial weights.

w1_updated = w1 - (learning rate) * ∂E_total/∂w1  
w2_updated = w2 - (learning rate) * ∂E_total/∂w2  
w3_updated = w3 - (learning rate) * ∂E_total/∂w3  
w4_updated = w4 - (learning rate) * ∂E_total/∂w4  
w5_updated = w5 - (learning rate) * ∂E_total/∂w5  
w6_updated = w6 - (learning rate) * ∂E_total/∂w6  
w7_updated = w7 - (learning rate) * ∂E_total/∂w7  
w8_updated = w8 - (learning rate) * ∂E_total/∂w8



![image](https://user-images.githubusercontent.com/71654199/119672983-a59f1d80-be58-11eb-939a-58219cd3fbea.png)

![image](https://user-images.githubusercontent.com/71654199/119671879-c31fb780-be57-11eb-9c09-4479afd09594.png)


