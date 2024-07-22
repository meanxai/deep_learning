# [MXDL-1-05] gradient_descent.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/M7bghSjr6TE
#
# We compute the gradient of the loss function with respect to 
# each parameter via numerical differentiation. And then we use 
# gradient descent to update all parameters [wh, bh, wo, bo].
import numpy as np

h = 1e-4    # small value for numerical differentiation
def numerical_differentiation(x, y, f_loss, f_predict, parameters):
    p = parameters
    p_gradients = []
    for i in range(len(p)):
        # p[0]= wh, p[1]= bh,...
        rows, cols = p[i].shape
        grad = np.zeros_like(p[i])
        
        # Apply numerical differentiation to all elements in p[i].
        for row in range(rows):
            for col in range(cols):
                # Measures the change in loss when the p[i][row, col] element 
                # changes by h. The remaining elements are fixed.
                # This is an approximate gradient.
                # gradient = (f(w1 + h, w2, ...) - f(w1 - h, w2, ...)) / (2h)
                p_org = p[i][row, col]       # original value of p
                p[i][row, col] = p_org + h   # The element at position (row, col) increases by h.
                y_hat = f_predict(x)         # calculate y_hat
                f1 = f_loss(y, y_hat)        # The amount of change in loss.
                
                p[i][row, col] = p_org - h   # The element at position (row, col) decreases by h.
                y_hat = f_predict(x)         # calculate y_hat
                f2 = f_loss(y, y_hat)        # The amount of change in loss.
                p[i][row, col] = p_org       # Restore p back to its original value.
                
                grad[row, col] = (f1 - f2) / (2. * h) # the gradient at position (row, col)
        p_gradients.append(grad)             # gradients for all paraters
    return p_gradients

# Perform Gradient Descent
def gradient_descent(x, y, alpha, f_loss, f_predict, parameters):
    p = parameters
    grad = numerical_differentiation(x, y, f_loss, f_predict, parameters)
    for i in range(len(p)):
        p[i] = p[i] - alpha * grad[i]