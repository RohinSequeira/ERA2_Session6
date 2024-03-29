
# Neural Network Feedforward and Backpropagation

Our objective here is to understand how back propagation works with Neural Networks, by implementing them in MS Excel. Read on to know more.

### Introduction:
Neural Network consists of neurons and connections between these neurons called weights and some biases connected to each neuron.
We move forward through the network called the forward pass, we iteratively use a formula to calculate each neuron in the next layer.

The network goes forward until we get an output. In this case we have one input layer, one hidden layer and last the output layer.
Our aim is to optimize the cost function. This is done by adjusting the weights based on errors generated by them through a backward pass.
To update the network, we calculate gradients w.r.t. each weights in every layer.

### Network Architechture:
There are two inputs: i1 and i2  
Targets: t1 and t2  
Hidden layer nodes: h1 and h2  
Activated nodes: a_h1 and a_h2  
Outputs: o1 and o2  
Activated outputs: a_o1 and a_o2  
Errors: E1 and E2  
Weights: w1, w2, w3, w4, w5, w6, w7 and w8  

![feedforward](https://user-images.githubusercontent.com/65554220/119848614-94721180-bf29-11eb-91e7-989e09bb1d98.JPG)

### Feed Forward Calculations:
#### Hidden layer:
h1 is getting input from i1 and i2 connected with w1 and w2 respectively. The hidden layer is calculated as below:  
h1 = w1 * i1 + w2 * i2  
Similarly, h2 is calculated as:  
h2 = w3 * i1 + w4 * i2  

#### Activation function at hidden layer:
In this example we are using sigmoid as the activation function.  
a_h1 = 𝜎(h1) = 1/(1 + exp(-h1))  
Similarly, a_h2 is calculated as:  
a_h2 = 𝜎(h2)  

#### Output layer:
o1 is getting input from a_h1 and a_h2 connected with w5 and w6 respectively. The output layer is calculated as below:  
o1 = w5 * a_h1 + w6 * a_h2  
Similarly, o2 is calculated as:  
o2 = w7 * a_h1 + w8 * a_h2  

#### Activation function at output layer:
In this example we are using sigmoid as the activation function.  
a_o1 = 𝜎(o1) = 1/(1 + exp(-o1))  
Similarly, a_o2 is calculated as:  
a_o2 = 𝜎(o2)  

#### Error calculations:
E1 = 1/2 * (t1 - a_o1)^2  
E2 = 1/2 * (t2 - a_o2)^2  
E_total = E1 + E2

### Back Propagation Calculations:

#### Back pass from output to hidden layer:
We have to calculate gradients of cost function E_Total w.r.t. all weights from output to hidden layer we have four weights w5, w6, w7 and w8.  

𝜕E_Total/𝜕w5 = 𝜕(E1 + E2)/𝜕w5  
Since, E2 is not connected to w5 the gradient across w5 will be zero.   
𝜕E_Total/𝜕w5 = 𝜕E1/𝜕w5  

Tracing the path moving backward, E1 is connected to a_o1, a_o1 connected to o1 and o1 connected to w5.   
𝜕E_Total/𝜕w5 = 𝜕E1/𝜕w5 = 𝜕E1/𝜕a_o1 * 𝜕a_o1/𝜕o1 * 𝜕o1/𝜕w5  
𝜕E1/𝜕a_o1 = 𝜕(1/2 * (t1 - a_o1)^2)/𝜕a_o1 = a_o1 - t1  
𝜕a_o1/𝜕o1 = 𝜕(𝜎(o1))/𝜕o1 = a_o1 * (1 - a_o1)  
𝜕o1/𝜕w5 = a_h1  

𝜕E_Total/𝜕w5 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h1  

Similarly, gradients are calculated for w6, w7 and w8.   
𝜕E_Total/𝜕w6 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h2  
𝜕E_Total/𝜕w7 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h1  
𝜕E_Total/𝜕w8 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h2  

#### Back pass from hidden layer to input layer:
We have to calculate gradients of cost function E_Total w.r.t. all weights fom hidden layer to input layer we have four weights w1, w2, w3 and w4.  

𝜕E_Total/𝜕w1 = 𝜕E_Total/𝜕a_h1 * 𝜕a_h1/𝜕h1 * 𝜕h1/𝜕w1  
𝜕E1/𝜕a_h1 = (a_o1 - t1) * a_o1 * (1- a_o1) * w5  
𝜕E2/𝜕a_h1 = (a_o2 - t2) * a_o2 * (1- a_o2) * w7  
𝜕E_Total/𝜕a_h1 = (a_o1 - t1) * a_o1 * (1- a_o1) * w5 + (a_o2 - t2) * a_o2 * (1- a_o2) * w7  
𝜕E_Total/𝜕a_h2 = (a_o1 - t1) * a_o1 * (1- a_o1) * w6 + (a_o2 - t2) * a_o2 * (1- a_o2) * w8  

𝜕E_Total/𝜕w1 = ((a_o1 - t1) * a_o1 * (1- a_o1) * w5 + (a_o2 - t2) * a_o2 * (1- a_o2) * w7) * a_h1 * (1 - a_h1) * i1  

Similarly, gradients are calculated for w2, w3 and w4.   
𝜕E_Total/𝜕w2 = ((a_o1 - t1) * a_o1 * (1- a_o1) * w5 + (a_o2 - t2) * a_o2 * (1- a_o2) * w7) * a_h1 * (1 - a_h1) * i2  
𝜕E_Total/𝜕w3 = ((a_o1 - t1) * a_o1 * (1- a_o1) * w6 + (a_o2 - t2) * a_o2 * (1- a_o2) * w8) * a_h2 * (1 - a_h2) * i1  
𝜕E_Total/𝜕w4 = ((a_o1 - t1) * a_o1 * (1- a_o1) * w6 + (a_o2 - t2) * a_o2 * (1- a_o2) * w8) * a_h2 * (1 - a_h2) * i2  

### Weight updation in next iteration
We simply calculate gradient of cost function w.r.t. the weight and multiply with the learning rate. Then the previous weight is subtracted by this learning rate gradient factor.
![image](https://user-images.githubusercontent.com/65554220/119844436-0d6f6a00-bf26-11eb-9fad-986fccb1099e.png)

Below image shows the effect of learning rates on convergence.
**As the learning rate is increased the convergence becomes faster**. As the learning rate increases, there will be a point where loss stops decreasing and starts increasing.
We should select learning rate such that the losses are minimized and converge at global minima faster, without overshooting.
In this example, we do not have a very high learning rate. Learning rate of 0.1 is too low in this case whereas 2 is a very good learning rate, as observed from the graphs.  

![learning rate](https://user-images.githubusercontent.com/65554220/119846043-6a1f5480-bf27-11eb-8487-4cf95a00d3d8.JPG)

The below screenshot is from the excel file "Back Propagation.xlsx" in the repository.
This shows how feed forward network and back propagation works.

![neural_network_feedforward_backpropagation](https://user-images.githubusercontent.com/65554220/119373201-cb4fe980-bcd5-11eb-82a4-01ef1e6cc6d0.JPG)
