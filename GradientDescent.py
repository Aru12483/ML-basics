import numpy as np
x=0.0
y=0.0
start = 0
def gradient_descent(
    gradient, x, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06,
    dtype="float64"
):

    # check if gradeint is callable or not 
    if not callable(gradient):
        raise TypeError ("Gradient must be callable")

    # initialise the data type of the NumPy arrays
    dtype_=np.dtype(dtype)

    # initalise x and y into numpy arrays and check if x and y are the same length
    x,y = np.array(x,dtype=dtype_),np.array(y,dtype=dtype_) 
    if x.shape[0] != y.shape[0]:
        raise ValueError ("X and Y lengths do not match")

    #initalise values of vector
    vector = np.array(start,dtype=dtype_)

    #checking learning rate
    learning_rate = np.array(learning_rate,dtype=dtype_)
    if learning_rate.any <= 0:
        raise ValueError ("Learning Rate must be above 0")

    #check max iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError ("Iterations must be above 0")

    #intialise and check tolerance 
    tolerance = np.array(tolerance,dtype=dtype_)
    if tolerance <= 0:
        raise ValueError ("Tolerance must be above 0")
    
    # Performing the gradient descent loop
    for _ in range(n_iter):
        # Recalculating the difference
        diff = -learn_rate * np.array(gradient(x, y, vector), dtype_)

        # Checking if the absolute difference is small enough
        if np.all(np.abs(diff) <= tolerance):
            break

        # Updating the values of the variables
        vector += diff

    return vector if vector.shape else vector.item()