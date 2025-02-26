# Cell 2
import numpy as np

# Cell 3
def MSE(y,y_pred):
    m = y.shape[0]
    loss = np.sum((y - y_pred)**2)/m
    return loss
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(x):
    return x * (1 - x)

def Forward_propagation(x,weights,biases):
    A = x
    A_values = [A]
    for i in range(len(weights)):
        z = np.dot(A,weights[i]) + biases[i]
        #print(z)
        A = sigmoid(z)
        A_values.append(A)
    return A_values

#correct bak_propagation
def Back_propagation(x, y, A_values, weights, biases):
    m = x.shape[0]
    
    # Start from the output layer
    dA = A_values[-1] - y  # Gradient of the loss w.r.t. the output layer activations
    dZ_output = dA * sigmoid_derivative(A_values[-1])  # Derivative of the sigmoid applied to output
    dW_output = np.dot(A_values[-2].T, dZ_output) / m
    db_output = np.sum(dZ_output, axis=0, keepdims=True) / m
    
    gradients = []
    dA = dZ_output  # Update dA to propagate backward
    
    # Now propagate backward through the hidden layers
    for i in range(len(weights) - 2, -1, -1):
        # FIX: Corrected the access to the activation values by using A_values[i + 1]
        dZ = dA * sigmoid_derivative(A_values[i + 1])  # Corrected this line
        
        # Compute gradients w.r.t. weights and biases
        dW = np.dot(A_values[i].T, dZ) / m if i > 0 else np.dot(x.T, dZ) / m  # Input layer case
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        gradients.append((dW, db))
        
        # Compute dA for the previous layer
        dA = np.dot(dZ, weights[i].T)
    
    gradients.reverse()  # Reverse gradients to match the order for updating weights
    return dW_output, db_output, gradients


def fit(x,y,weights,biases,epochs=2000,learning_rate = 0.0001):
    for epoch in range(epochs):
        A_values = Forward_propagation(x,weights,biases)
        dW_output,db_output,gradients = Back_propagation(x,y,A_values,weights,biases)
        #update the weights and biases
        #for output layer
        weights[-1] -= learning_rate * dW_output
        biases[-1] -= learning_rate * db_output
        #for hidden layer
        for i in range(len(weights)-2,-1,-1):
            dW,db = gradients[i]
            weights[i] -= learning_rate * dW
            biases[i] -= learning_rate * db
        if epoch %100 == 0:
            y_pred = A_values[-1]
            loss = MSE(y,y_pred)
            print(f"Epoch{epoch}/{epochs}Loss:{loss}")
    
def initialize_network(input_size,hidden_sizes,output_size):
    np.random.seed(42)
    weights = []
    biases = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        weights.append(np.random.randn(previous_size, hidden_size))
        biases.append(np.random.randn(1, hidden_size))
        previous_size = hidden_size
    
    weights.append(np.random.randn(previous_size, output_size))
    biases.append(np.random.randn(1, output_size))
    return weights, biases

x1 = [1, 2, 3, 4, 5, 6]
x2 = [20, 40, 60, 80, 90, 95]
y = [0, 0, 0, 1, 1, 1]

x = np.array(list(zip(x1,x2)))

y = np.array(y).reshape(-1,1)

input_size = x.shape[1]

hidden_sizes = [2,2,2]
output_size = 1

weights,biases = initialize_network(input_size,hidden_sizes,output_size)

fit(x,y,weights,biases)

# Cell 4
def MSE(y,y_pred):
    m = y.shape[0]
    loss = np.sum((y - y_pred)**2)/m
    return loss
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(x):
    return x * (1 - x)

def Forward_propagation(x,weights,biases):
    A = x
    A_values = [A]
    for i in range(len(weights)):
        z = np.dot(A,weights[i]) + biases[i]
        #print(z)
        A = sigmoid(z)
        A_values.append(A)
    return A_values

#correct bak_propagation
def Back_propagation(x, y, A_values, weights, biases):
    m = x.shape[0]
    
    # Start from the output layer
    dA = A_values[-1] - y  # Gradient of the loss w.r.t. the output layer activations
    dZ_output = dA * sigmoid_derivative(A_values[-1])  # Derivative of the sigmoid applied to output
    dW_output = np.dot(A_values[-2].T, dZ_output) / m
    db_output = np.sum(dZ_output, axis=0, keepdims=True) / m
    
    gradients = []
    dA = dZ_output  # Update dA to propagate backward
    
    # Now propagate backward through the hidden layers
    for i in range(len(weights) - 2, -1, -1):
        # FIX: Corrected the access to the activation values by using A_values[i + 1]
        dZ = dA * sigmoid_derivative(A_values[i + 1])  # Corrected this line
        
        # Compute gradients w.r.t. weights and biases
        dW = np.dot(A_values[i].T, dZ) / m if i > 0 else np.dot(x.T, dZ) / m  # Input layer case
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        gradients.append((dW, db))
        
        # Compute dA for the previous layer
        dA = np.dot(dZ, weights[i].T)
    
    gradients.reverse()  # Reverse gradients to match the order for updating weights
    return dW_output, db_output, gradients


def fit(x,y,weights,biases,epochs=2000,learning_rate = 0.0001):
    for epoch in range(epochs):
        A_values = Forward_propagation(x,weights,biases)
        dW_output,db_output,gradients = Back_propagation(x,y,A_values,weights,biases)
        #update the weights and biases
        #for output layer
        weights[-1] -= learning_rate * dW_output
        biases[-1] -= learning_rate * db_output
        #for hidden layer
        for i in range(len(weights)-2,-1,-1):
            dW,db = gradients[i]
            weights[i] -= learning_rate * dW
            biases[i] -= learning_rate * db
        if epoch %100 == 0:
            y_pred = A_values[-1]
            loss = MSE(y,y_pred)
            print(f"Epoch{epoch}/{epochs}Loss:{loss}")
    
def initialize_network(input_size,hidden_sizes,output_size):
    np.random.seed(42)
    weights = []
    biases = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        weights.append(np.random.randn(previous_size, hidden_size))
        biases.append(np.random.randn(1, hidden_size))
        previous_size = hidden_size
    
    weights.append(np.random.randn(previous_size, output_size))
    biases.append(np.random.randn(1, output_size))
    return weights, biases

x1 = [1, 2, 3, 4, 5, 6]
x2 = [20, 40, 60, 80, 90, 95]
y = [0, 0, 0, 1, 1, 1]

x = np.array(list(zip(x1,x2)))

y = np.array(y).reshape(-1,1)

input_size = x.shape[1]
num_neurons = 4
hidden_layers = 4
hidden_sizes = [num_neurons]*hidden_layers
output_size = 1

weights,biases = initialize_network(input_size,hidden_sizes,output_size)

fit(x,y,weights,biases)

# Cell 5
def MSE(y,y_pred):
    m = y.shape[0]
    loss = np.sum((y - y_pred)**2)/m
    return loss
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(x):
    return x * (1 - x)

def Forward_propagation(x,weights,biases):
    A = x
    A_values = [A]
    for i in range(len(weights)):
        z = np.dot(A,weights[i]) + biases[i]
        #print(z)
        A = sigmoid(z)
        A_values.append(A)
    return A_values

#correct bak_propagation
def Back_propagation(x, y, A_values, weights, biases):
    m = x.shape[0]
    
    # Start from the output layer
    dA = A_values[-1] - y  # Gradient of the loss w.r.t. the output layer activations
    dZ_output = dA * sigmoid_derivative(A_values[-1])  # Derivative of the sigmoid applied to output
    dW_output = np.dot(A_values[-2].T, dZ_output) / m
    db_output = np.sum(dZ_output, axis=0, keepdims=True) / m
    
    gradients = []
    dA = dZ_output  # Update dA to propagate backward
    
    # Now propagate backward through the hidden layers
    for i in range(len(weights) - 2, -1, -1):
        # FIX: Corrected the access to the activation values by using A_values[i + 1]
        dZ = dA * sigmoid_derivative(A_values[i + 1])  # Corrected this line
        
        # Compute gradients w.r.t. weights and biases
        dW = np.dot(A_values[i].T, dZ) / m if i > 0 else np.dot(x.T, dZ) / m  # Input layer case
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        gradients.append((dW, db))
        
        # Compute dA for the previous layer
        dA = np.dot(dZ, weights[i].T)
    
    gradients.reverse()  # Reverse gradients to match the order for updating weights
    return dW_output, db_output, gradients


def fit(x,y,weights,biases,epochs=2000,learning_rate = 0.0001):
    for epoch in range(epochs):
        A_values = Forward_propagation(x,weights,biases)
        dW_output,db_output,gradients = Back_propagation(x,y,A_values,weights,biases)
        #update the weights and biases
        #for output layer
        weights[-1] -= learning_rate * dW_output
        biases[-1] -= learning_rate * db_output
        #for hidden layer
        for i in range(len(weights)-2,-1,-1):
            dW,db = gradients[i]
            weights[i] -= learning_rate * dW
            biases[i] -= learning_rate * db
        if epoch %100 == 0:
            y_pred = A_values[-1]
            loss = MSE(y,y_pred)
            print(f"Epoch{epoch}/{epochs}Loss:{loss}")
    
def initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size):
    hidden_sizes = [num_neurons]*hidden_layers
    np.random.seed(42)
    weights = []
    biases = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        weights.append(np.random.randn(previous_size, hidden_size))
        biases.append(np.random.randn(1, hidden_size))
        previous_size = hidden_size
    
    weights.append(np.random.randn(previous_size, output_size))
    biases.append(np.random.randn(1, output_size))
    return weights, biases

x1 = [1, 2, 3, 4, 5, 6]
x2 = [20, 40, 60, 80, 90, 95]
y = [0, 0, 0, 1, 1, 1]

x = np.array(list(zip(x1,x2)))

y = np.array(y).reshape(-1,1)

input_size = x.shape[1]
num_neurons = 4
hidden_layers = 4
#hidden_sizes = [num_neurons]*hidden_layers
output_size = 1

weights,biases = initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size)

fit(x,y,weights,biases)

# Cell 6
def MSE(y,y_pred):
    m = y.shape[0]
    loss = np.sum((y - y_pred)**2)/m
    return loss
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(x):
    return x * (1 - x)

def Forward_propagation(x,weights,biases):
    A = x
    A_values = [A]
    for i in range(len(weights)):
        z = np.dot(A,weights[i]) + biases[i]
        #print(z)
        A = sigmoid(z)
        A_values.append(A)
    return A_values

#correct bak_propagation
def Back_propagation(x, y, A_values, weights, biases):
    m = x.shape[0]
    
    # Start from the output layer
    dA = A_values[-1] - y  # Gradient of the loss w.r.t. the output layer activations
    dZ_output = dA * sigmoid_derivative(A_values[-1])  # Derivative of the sigmoid applied to output
    dW_output = np.dot(A_values[-2].T, dZ_output) / m
    db_output = np.sum(dZ_output, axis=0, keepdims=True) / m
    
    gradients = []
    dA = dZ_output  # Update dA to propagate backward
    
    # Now propagate backward through the hidden layers
    for i in range(len(weights) - 2, -1, -1):
        # FIX: Corrected the access to the activation values by using A_values[i + 1]
        dZ = dA * sigmoid_derivative(A_values[i + 1])  # Corrected this line
        
        # Compute gradients w.r.t. weights and biases
        dW = np.dot(A_values[i].T, dZ) / m if i > 0 else np.dot(x.T, dZ) / m  # Input layer case
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        gradients.append((dW, db))
        
        # Compute dA for the previous layer
        dA = np.dot(dZ, weights[i].T)
    
    gradients.reverse()  # Reverse gradients to match the order for updating weights
    return dW_output, db_output, gradients


def fit(x,y,weights,biases,epochs=2000,learning_rate = 0.0001):
    for epoch in range(epochs):
        A_values = Forward_propagation(x,weights,biases)
        dW_output,db_output,gradients = Back_propagation(x,y,A_values,weights,biases)
        #update the weights and biases
        #for output layer
        weights[-1] -= learning_rate * dW_output
        biases[-1] -= learning_rate * db_output
        #for hidden layer
        for i in range(len(weights)-2,-1,-1):
            dW,db = gradients[i]
            weights[i] -= learning_rate * dW
            biases[i] -= learning_rate * db
        if epoch %100 == 0:
            y_pred = A_values[-1]
            loss = MSE(y,y_pred)
            print(f"Epoch{epoch}/{epochs}Loss:{loss}")
    
def initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size):
    hidden_sizes = [num_neurons]*hidden_layers
    np.random.seed(42)
    weights = []
    biases = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        weights.append(np.random.randn(previous_size, hidden_size))
        biases.append(np.random.randn(1, hidden_size))
        previous_size = hidden_size
    
    weights.append(np.random.randn(previous_size, output_size))
    biases.append(np.random.randn(1, output_size))
    return weights, biases

x1 = [1, 2, 3, 4, 5, 6]
x2 = [20, 40, 60, 80, 90, 95]
y = [0, 0, 0, 1, 1, 1]

x = np.array(list(zip(x1,x2)))

y = np.array(y).reshape(-1,1)

input_size = x.shape[1]
num_neurons = 7
hidden_layers = 6
#hidden_sizes = [num_neurons]*hidden_layers
output_size = 1

weights,biases = initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size)

fit(x,y,weights,biases)

# Cell 7
def MSE(y,y_pred):
    m = y.shape[0]
    loss = np.sum((y - y_pred)**2)/m
    return loss
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(x):
    return x * (1 - x)

def Forward_propagation(x,weights,biases):
    A = x
    A_values = [A]
    for i in range(len(weights)):
        z = np.dot(A,weights[i]) + biases[i]
        #print(z)
        A = sigmoid(z)
        A_values.append(A)
    return A_values

#correct bak_propagation
def Back_propagation(x, y, A_values, weights, biases):
    m = x.shape[0]
    
    # Start from the output layer
    dA = A_values[-1] - y  # Gradient of the loss w.r.t. the output layer activations
    dZ_output = dA * sigmoid_derivative(A_values[-1])  # Derivative of the sigmoid applied to output
    dW_output = np.dot(A_values[-2].T, dZ_output) / m
    db_output = np.sum(dZ_output, axis=0, keepdims=True) / m
    
    gradients = []
    dA = dZ_output  # Update dA to propagate backward
    
    # Now propagate backward through the hidden layers
    for i in range(len(weights) - 2, -1, -1):
        # FIX: Corrected the access to the activation values by using A_values[i + 1]
        dZ = dA * sigmoid_derivative(A_values[i + 1])  # Corrected this line
        
        # Compute gradients w.r.t. weights and biases
        dW = np.dot(A_values[i].T, dZ) / m if i > 0 else np.dot(x.T, dZ) / m  # Input layer case
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        gradients.append((dW, db))
        
        # Compute dA for the previous layer
        dA = np.dot(dZ, weights[i].T)
    
    gradients.reverse()  # Reverse gradients to match the order for updating weights
    return dW_output, db_output, gradients


def fit(x,y,weights,biases,epochs=2000,learning_rate = 0.0001):
    for epoch in range(epochs):
        A_values = Forward_propagation(x,weights,biases)
        dW_output,db_output,gradients = Back_propagation(x,y,A_values,weights,biases)
        #update the weights and biases
        #for output layer
        weights[-1] -= learning_rate * dW_output
        biases[-1] -= learning_rate * db_output
        #for hidden layer
        for i in range(len(weights)-2,-1,-1):
            dW,db = gradients[i]
            weights[i] -= learning_rate * dW
            biases[i] -= learning_rate * db
        if epoch %100 == 0:
            y_pred = A_values[-1]
            loss = MSE(y,y_pred)
            print(f"Epoch{epoch}/{epochs}Loss:{loss}")
    
def initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size):
    hidden_sizes = [num_neurons]*hidden_layers
    np.random.seed(42)
    weights = []
    biases = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        weights.append(np.random.randn(previous_size, hidden_size))
        biases.append(np.random.randn(1, hidden_size))
        previous_size = hidden_size
    
    weights.append(np.random.randn(previous_size, output_size))
    biases.append(np.random.randn(1, output_size))
    return weights, biases

x1 = [1, 2, 3, 4, 5, 6]
x2 = [20, 40, 60, 80, 90, 95]
y = [0, 0, 0, 1, 1, 1]

x = np.array(list(zip(x1,x2)))

y = np.array(y).reshape(-1,1)

input_size = x.shape[1]
num_neurons = 7
hidden_layers = 6
#hidden_sizes = [num_neurons]*hidden_layers
output_size = 1

weights,biases = initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size)

fit(x,y,weights,biases)

x_new = np.array([[7, 85], [8, 100], [9, 110]])
y_true_new = np.array([[0], [1], [1]])


y_pred_binary = predict(x_new, weights, biases)
print(y_pred_binary)
#acc = accuracy(y_true_new, y_pred_binary)
#print(f"Accuracy: {acc * 100:.2f}%")

# Cell 8
def MSE(y,y_pred):
    m = y.shape[0]
    loss = np.sum((y - y_pred)**2)/m
    return loss
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(x):
    return x * (1 - x)

def Forward_propagation(x,weights,biases):
    A = x
    A_values = [A]
    for i in range(len(weights)):
        z = np.dot(A,weights[i]) + biases[i]
        #print(z)
        A = sigmoid(z)
        A_values.append(A)
    return A_values

#correct bak_propagation
def Back_propagation(x, y, A_values, weights, biases):
    m = x.shape[0]
    
    # Start from the output layer
    dA = A_values[-1] - y  # Gradient of the loss w.r.t. the output layer activations
    dZ_output = dA * sigmoid_derivative(A_values[-1])  # Derivative of the sigmoid applied to output
    dW_output = np.dot(A_values[-2].T, dZ_output) / m
    db_output = np.sum(dZ_output, axis=0, keepdims=True) / m
    
    gradients = []
    dA = dZ_output  # Update dA to propagate backward
    
    # Now propagate backward through the hidden layers
    for i in range(len(weights) - 2, -1, -1):
        # FIX: Corrected the access to the activation values by using A_values[i + 1]
        dZ = dA * sigmoid_derivative(A_values[i + 1])  # Corrected this line
        
        # Compute gradients w.r.t. weights and biases
        dW = np.dot(A_values[i].T, dZ) / m if i > 0 else np.dot(x.T, dZ) / m  # Input layer case
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        gradients.append((dW, db))
        
        # Compute dA for the previous layer
        dA = np.dot(dZ, weights[i].T)
    
    gradients.reverse()  # Reverse gradients to match the order for updating weights
    return dW_output, db_output, gradients


def fit(x,y,weights,biases,epochs=2000,learning_rate = 0.0001):
    for epoch in range(epochs):
        A_values = Forward_propagation(x,weights,biases)
        dW_output,db_output,gradients = Back_propagation(x,y,A_values,weights,biases)
        #update the weights and biases
        #for output layer
        weights[-1] -= learning_rate * dW_output
        biases[-1] -= learning_rate * db_output
        #for hidden layer
        for i in range(len(weights)-2,-1,-1):
            dW,db = gradients[i]
            weights[i] -= learning_rate * dW
            biases[i] -= learning_rate * db
        if epoch %100 == 0:
            y_pred = A_values[-1]
            loss = MSE(y,y_pred)
            print(f"Epoch{epoch}/{epochs}Loss:{loss}")
    
def initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size):
    hidden_sizes = [num_neurons]*hidden_layers
    np.random.seed(42)
    weights = []
    biases = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        weights.append(np.random.randn(previous_size, hidden_size))
        biases.append(np.random.randn(1, hidden_size))
        previous_size = hidden_size
    
    weights.append(np.random.randn(previous_size, output_size))
    biases.append(np.random.randn(1, output_size))
    return weights, biases

def predict(x, weights, biases, threshold=0.5):
    A_values = Forward_propagation(x, weights, biases)
    y_pred = A_values[-1]
    # Apply thresholding to convert predictions to binary values
    y_pred_binary = (y_pred >= threshold).astype(int)
    return y_pred_binary

def accuracy(y_true, y_pred):
    # Compare predicted values with true values
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = y_true.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy



x1 = [1, 2, 3, 4, 5, 6]
x2 = [20, 40, 60, 80, 90, 95]
y = [0, 0, 0, 1, 1, 1]

x = np.array(list(zip(x1,x2)))

y = np.array(y).reshape(-1,1)

input_size = x.shape[1]
num_neurons = 7
hidden_layers = 6
#hidden_sizes = [num_neurons]*hidden_layers
output_size = 1

weights,biases = initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size)

fit(x,y,weights,biases)

x_new = np.array([[7, 85], [8, 100], [9, 110]])
y_true_new = np.array([[0], [1], [1]])


y_pred_binary = predict(x_new, weights, biases)
print(y_pred_binary)
#acc = accuracy(y_true_new, y_pred_binary)
#print(f"Accuracy: {acc * 100:.2f}%")

# Cell 9
def MSE(y,y_pred):
    m = y.shape[0]
    loss = np.sum((y - y_pred)**2)/m
    return loss
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(x):
    return x * (1 - x)

def Forward_propagation(x,weights,biases):
    A = x
    A_values = [A]
    for i in range(len(weights)):
        z = np.dot(A,weights[i]) + biases[i]
        #print(z)
        A = sigmoid(z)
        A_values.append(A)
    return A_values

#correct bak_propagation
def Back_propagation(x, y, A_values, weights, biases):
    m = x.shape[0]
    
    # Start from the output layer
    dA = A_values[-1] - y  # Gradient of the loss w.r.t. the output layer activations
    dZ_output = dA * sigmoid_derivative(A_values[-1])  # Derivative of the sigmoid applied to output
    dW_output = np.dot(A_values[-2].T, dZ_output) / m
    db_output = np.sum(dZ_output, axis=0, keepdims=True) / m
    
    gradients = []
    dA = dZ_output  # Update dA to propagate backward
    
    # Now propagate backward through the hidden layers
    for i in range(len(weights) - 2, -1, -1):
        # FIX: Corrected the access to the activation values by using A_values[i + 1]
        dZ = dA * sigmoid_derivative(A_values[i + 1])  # Corrected this line
        
        # Compute gradients w.r.t. weights and biases
        dW = np.dot(A_values[i].T, dZ) / m if i > 0 else np.dot(x.T, dZ) / m  # Input layer case
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        gradients.append((dW, db))
        
        # Compute dA for the previous layer
        dA = np.dot(dZ, weights[i].T)
    
    gradients.reverse()  # Reverse gradients to match the order for updating weights
    return dW_output, db_output, gradients


def fit(x,y,weights,biases,epochs=2000,learning_rate = 0.0001):
    for epoch in range(epochs):
        A_values = Forward_propagation(x,weights,biases)
        dW_output,db_output,gradients = Back_propagation(x,y,A_values,weights,biases)
        #update the weights and biases
        #for output layer
        weights[-1] -= learning_rate * dW_output
        biases[-1] -= learning_rate * db_output
        #for hidden layer
        for i in range(len(weights)-2,-1,-1):
            dW,db = gradients[i]
            weights[i] -= learning_rate * dW
            biases[i] -= learning_rate * db
        if epoch %100 == 0:
            y_pred = A_values[-1]
            loss = MSE(y,y_pred)
            print(f"Epoch{epoch}/{epochs}Loss:{loss}")
    
def initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size):
    hidden_sizes = [num_neurons]*hidden_layers
    np.random.seed(42)
    weights = []
    biases = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        weights.append(np.random.randn(previous_size, hidden_size))
        biases.append(np.random.randn(1, hidden_size))
        previous_size = hidden_size
    
    weights.append(np.random.randn(previous_size, output_size))
    biases.append(np.random.randn(1, output_size))
    return weights, biases

def predict(x, weights, biases, threshold=0.5):
    A_values = Forward_propagation(x, weights, biases)
    y_pred = A_values[-1]
    # Apply thresholding to convert predictions to binary values
    y_pred_binary = (y_pred >= threshold).astype(int)
    return y_pred_binary

def accuracy(y_true, y_pred):
    # Compare predicted values with true values
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = y_true.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy



x1 = [1, 2, 3, 4, 5, 6]
x2 = [20, 40, 60, 80, 90, 95]
y = [0, 0, 0, 1, 1, 1]

x = np.array(list(zip(x1,x2)))

y = np.array(y).reshape(-1,1)

input_size = x.shape[1]
num_neurons = 7
hidden_layers = 6
#hidden_sizes = [num_neurons]*hidden_layers
output_size = 1

weights,biases = initialize_network(input_size,num_neurons,hidden_layers,hidden_sizes,output_size)

fit(x,y,weights,biases)

x_new = np.array([[7, 85], [8, 100], [9, 110]])
y_true_new = np.array([[0], [1], [1]])


y_pred_binary = predict(x_new, weights, biases)
print(y_pred_binary)
acc = accuracy(y_true_new, y_pred_binary)
print(f"Accuracy: {acc * 100:.2f}%")

