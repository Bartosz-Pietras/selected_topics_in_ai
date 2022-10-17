from copy import deepcopy
import numpy as np

# LINEAR REGRESSION

w = np.array([1, 3, 3])
b = 12
num_of_samples = 1000
x = np.random.rand(3, num_of_samples)
bias = np.ones(shape=(num_of_samples, 1))

empty_matrix = np.empty(shape=(4, num_of_samples))

for row in range(x.shape[0]):
    for column in range(x.shape[1]):
        empty_matrix[row, column] = x[row, column]

empty_matrix[3, :] = bias[0, :]
biased_matrix = np.transpose(empty_matrix)

print(biased_matrix.shape)
y = np.dot(w, x) + b

inverse = np.linalg.inv(np.dot(np.transpose(biased_matrix), biased_matrix))
W_temp = np.dot(inverse, np.transpose(biased_matrix))
W_hat = np.dot(W_temp, y)

for i in range(4):
    print(f"| Calculated weight: {W_hat[i]}")



# GRADIENT DESCENT

num_of_iterations = 1000
learning_rate = 0.0001
w_rand = np.random.rand(3)
w_est = np.ones(shape=(4, ))


for i in range(w_rand.size):
    w_est[i] = w_rand[i]

transposed_matrix = np.transpose(biased_matrix)

for epoch in range(num_of_iterations):

    sum = 0
    for i in range(num_of_samples):
        X_W = np.dot(biased_matrix, w_est)
        difference = X_W - y
        sum += np.dot(transposed_matrix, difference)

    mse = (2 / num_of_samples) * sum
    w_est = w_est - learning_rate * mse

print(w_est)


# STOCHASTIC GRADIENT DESCENT


num_of_iterations = 1000
learning_rate = 0.0001
w_rand = np.random.rand(3)
w_est = np.ones(shape=(4, ))


for i in range(w_rand.size):
    w_est[i] = w_rand[i]

transposed_matrix = np.transpose(biased_matrix)

for epoch in range(num_of_iterations):
    shuffler = np.random.permutation(biased_matrix.shape[0])

    temp_x = biased_matrix[shuffler]
    temp_y = y[shuffler]
    for i in range(num_of_samples):
        mse = np.dot((2 * transposed_matrix), (np.dot(temp_x[i], w_est) - temp_y[i]))
        w_est = w_est - learning_rate * mse

print(w_est)

