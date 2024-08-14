import numpy as np

np.random.seed(0)

X=[[1,2,3,2.5],
   [2.0,5.0,-1.0,2.0],
   [-1.5, 2.7,3.3,-0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights=0.10*np.random.randn(n_inputs, n_neurons) #check the params -- shapes
        self.biases=np.zeros((1,n_neurons)) #check why we're passing a tuple here and not above
    def forward(self, inputs):
        self.output=np.dot(inputs, self.weights)+self.biases

layer1=Layer_Dense(4,5)
layer2=Layer_Dense(5,2)

layer1.forward(X)

# print(layer1.output)

layer2.forward(layer1.output)

print(layer2.output)




#what exactly do weights and biases actually signify?

# layer_outputs=[]
# for neuron_weights, neuron_bias in zip(weights, biases): #what is this
#     neuron_output=np.dot(inputs, neuron_weights)+neuron_bias
#     layer_outputs.append(float(neuron_output))

# layer_outputs=np.dot(weights, inputs)+biases


#get a really good grasp of shape!! dot only works with weights, inputs -- not the other way around!!!


# output=[inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3]+bias1,
#         inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+bias2,
#         inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+bias3
#        ]

#update 140824: got sidetracked a lot because of other work, ready to get back to this :))
