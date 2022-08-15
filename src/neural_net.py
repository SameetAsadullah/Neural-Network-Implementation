#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Dense():
    def __init__(self, neurons, activation, input_shape = None):
        self.activation = activation
        self.bias = np.random.rand(neurons, 1)
        self.neurons = neurons
        self.initialiseWeights(input_shape)
    
    def initialiseWeights(self, input_shape):
        if input_shape != None:
            self.weights = np.random.rand(self.neurons, input_shape[1] * input_shape[2])
        else:
            self.weights = []
            
    # activation functions
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def relu(self, z):
        for i in range(z.shape[0]):
            if z[i] < 0:
                z[i] = 0
        return z
    
    def leaky_relu(self, z):
        if z > 0:
            return z
        else:
            return 0.01 * z
    
    def softmax(self, outputs):
        e = np.exp(outputs)
        return e / e.sum()
    
    # activation function gradients
    def relu_gradient(self, result):
        dev = np.array([])
        for x in result:
            if x > 0:
                dev = np.append(dev,1)
            else:
                dev = np.append(dev, 0)
        return dev
    
    def sigmoid_gradient(self, result):
        return sigmoid(result) * (1-sigmoid(result))
    
    # forward and backward pass
    def forwardPass(self, input_data):
        self.input = input_data
        if len(self.weights) == 0:
            input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1])
            self.initialiseWeights(input_data.shape)
        self.output = np.dot(self.weights, self.input) + self.bias
        
        if self.activation == "sigmoid":
            self.output = self.sigmoid(self.output)
        elif self.activation == "relu":
            self.output = self.relu(self.output)
        
        return self.output
    
    def backwardPass(self, mask, learning_rate):
        input_error = np.dot(self.weights.T, mask)
        weights_error = np.dot(self.input, mask.T)

        # update parameters
        self.weights -= learning_rate * weights_error.T
        self.bias -= learning_rate * mask
        
        return input_error
    
    # Optimizers
    def gradientDescent(self, alpha, gradient):
        self.weights = self.weights - alpha * gradient

    def adaGrad(self, alpha, gradient, Sum):
        Sum += gradient ** 2
        alpha = alpha / np.sqrt(Sum)
        self.weights = self.weights - alpha * gradient

    def RMSProp(self, alpha, gradient, average, beta=0.9):
        average = beta * average + (1 - beta) * gradient * gradient
        alpha = alpha / np.sqrt(average)
        self.weights = self.weights - alpha * gradient

    def adam(self, c, a, alpha, gradient, beta1=0.9, beta2=0.9):
        c = beta1 * c + (1 - beta1) * gradient
        a = beta2 * a + (1 - beta2) * np.power(gradient, 2)
        alpha = alpha / np.sqrt(a)
        self.weights = self.weights - alpha * gradient


# In[ ]:


class neural_net:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def fit(self, train_X, train_y, epochs, learning_rate, loss):
        samples = len(train_X)

        # training loop
        for i in range(epochs):
            err = 0
            
            for j in range(samples):
                # forward propagation
                output = train_X[j]
                output = output.reshape(output.shape[0] * output.shape[1], 1)
                for layer in self.layers:
                    output = layer.forwardPass(output)

                # compute loss
                if loss == "hinge-loss":
                    err_, mask  = self.hinge_loss(output, train_y[j], 1)
                elif loss == "cross-entropy-loss":
                    err_, mask = self.cross_entropy_loss(output, train_y[j])

                error = mask
                # backward propagation
                for layer in reversed(self.layers):
                    error = layer.backwardPass(error, learning_rate)
            
    def predict(self, test_X, test_y):
        predictions = []
        for test_point in test_X:
            output = test_point
            output = output.reshape(output.shape[0] * output.shape[1], 1)
            for layer in self.layers:
                output = layer.forwardPass(output)
            y = output.argmax(axis=-1)
            predictions.append(y)
            
#         print(predictions)
#         print(test_y)
        
    # loss functions
    def cross_entropy_loss(self, scores, true_class):
        scores_temp = np.copy(scores)
        scores_temp -= scores.max(keepdims = True)
        probs = np.exp(scores)/np.sum(np.exp(scores), keepdims = True)
        loss = -np.log(probs[true_class])
        loss = np.sum(loss)
        
        dscores[true_class] -= 1
        
        return loss, dscores
    

    def hinge_loss(self, scores, true_class, theta):
        correct_class_score = scores[true_class]
        margin = np.maximum(0, scores - correct_class_score + theta)
        margin[true_class] = 0
        
        valid_margin_mask = np.zeros(margin.shape)
        valid_margin_mask[margin > 0] = 1 
        valid_margin_mask[true_class] = -np.sum(valid_margin_mask)
        
        return np.sum(margin), valid_margin_mask
    
    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2));
    
    def mse_gradient(self, y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size;

