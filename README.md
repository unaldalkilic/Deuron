# Deuron (Alpha)
Deuron is an neural network train framework that uses Numpy. It has various abilities such as: normalization, standardization, 3 kind of regularization and 2 kind of optimization algorithms. Deuron's main purpose is to train your data with easiest and most flexible tool.

Note: Still in development

================================================================
HOW TO USE

1. Generate a Deuron object by using:
  - X, the training input data as numpy matrix
  - Y, the training output data as numpy matrix
  - layer list that includes number of nodes and activation types as list
    Exp. layer_list =  [(30, Deuron.ACTIVATION.SIGMOID), (30, Deuron.ACTIVATION.TANH), (10, Deuron.ACTIVATION.SOFTMAX)]
  - cost type parameter that neural network is going to use
    Exp. Deuron.COST.SQUARE_ERROR_COST, Deuron.COST.CROSS_ENTROPY_COST, Deuron.COST.EXTENDED_CROSS_ENTROPY_COST
  - epoch (iteration) count
  - initial learning rate alpha
  
  - Sample Code:
    - (In parameter order)
      ```
      deuron = Deuron(X, Y, layer_list, Deuron.COST.EXTENDED_CROSS_ENTROPY_COST, 1000, 0.15)
      ```
    - (Not in parameter order)
      ```
      deuron = Deuron(X = X, Y = Y, layer_list = layer_list, cost_type = Deuron.COST.EXTENDED_CROSS_ENTROPY_COST, epoch_number = 1000, alpha = 0.15)
      ```
    
2. (Optional) Predefine the algotihms that are going to use in learning process (For detail usage, check documentation)
  - Algorithm list:
    - Scale data --> Normalization, Standardization
    - Regularization --> L1, L2, Dropout
    - Batch size scale --> Batch, Mini batch or Stochastic gradient descent
    - Optimization --> Momentum, Adam
    - Learning Decay
    
3. Start the learning process (deuron's start function returns copies of the weights as tuple: (W, B))
  ```
  parameters = deuron.start()
  ```
    
4. (Optional) Test your test input data after learning by using simple test function
  ```
  Y_predict = deuron.test(X_test)
  ```
