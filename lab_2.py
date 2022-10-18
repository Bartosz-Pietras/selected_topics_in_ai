import numpy as np

# HELPER FUNCTIONS
def get_biased_matrix(matrix):
    rows, columns = matrix.shape
    new_matrix = np.empty(shape=(rows + 1, columns))
    bias_to_add = np.ones(shape=(columns, 1))

    for row in range(rows):
        for column in range(columns):
            new_matrix[row, column] = matrix[row, column]
    
    new_matrix[rows, :] = bias_to_add[0, :]
    
    return np.transpose(new_matrix)

def get_biased_starting_weights_matrix(initial_size=3):
    w_rand = np.random.rand(initial_size)
    w_est = np.ones(shape=(initial_size + 1, ))
    for i in range(w_rand.size):
        w_est[i] = w_rand[i]
    
    return w_est


# LINEAR REGRESSION
def linear_regression(input_data, expected_output):
    # Add a column of 1's to the input data
    X = get_biased_matrix(input_data)
    X_t = np.transpose(X)

    # Perform linear regression and print the results
    inverse = np.linalg.inv(np.dot(X_t, X))
    result = np.dot(np.dot(inverse, X_t), expected_output)
    print(f"Linear Regression result: {result}")



# BATCH GRADIENT DESCENT
def batch_gradient_descent(num_iters, learning_rate, input_data, expected_output):
    w_est = get_biased_starting_weights_matrix()
    X = get_biased_matrix(input_data)
    X_t = np.transpose(X)

    # Perform BGD
    for _ in range(num_iters):
        sum = 0
        for _ in range(X.shape[0]):
            X_W = np.dot(X, w_est)
            difference = X_W - expected_output
            sum += np.dot(X_t, difference)
        mse = (2 / X.shape[0]) * sum
        w_est = w_est - learning_rate * mse

    print(f"Batch Gradient Descent: {w_est}")


# STOCHASTIC GRADIENT DESCENT
def stochastic_gradient_descent(num_iters, learning_rate, input_data, expected_output):
    w_est = get_biased_starting_weights_matrix()
    X = get_biased_matrix(input_data)
    X_t = np.transpose(X)

    # Perform SGD
    for _ in range(num_iters):
        rand_index = np.random.randint(0, X.shape[0])
        X_W = np.dot(X[rand_index, :], w_est)
        mse = np.dot(2 * X_t[:, rand_index], X_W - expected_output[rand_index])
        w_est = w_est - learning_rate * mse        

    print(f"Stochastic Gradient Descent: {w_est}")


def mini_batch_gradient_descent(num_iters, batch_size, learning_rate, input_data, expected_output):
    w_est = get_biased_starting_weights_matrix()
    X = get_biased_matrix(input_data)
    X_t = np.transpose(X)

    # Perform MBGD
    for _ in range(num_iters):
        mse = 0
        for _ in range(batch_size):
            rand_index = np.random.randint(0, X.shape[0])
            X_W = np.dot(X[rand_index, :], w_est)
            mse += np.dot(2 * X_t[:, rand_index], X_W - expected_output[rand_index])
        w_est = w_est - learning_rate * mse

    print(f"Mini-Batch Gradient Descent: {w_est}")



if __name__ == "__main__":
    weights = np.array([1, 3, 3], dtype=np.float32)
    bias = 7
    num_of_samples = 1000
    x = np.random.rand(3, num_of_samples)
    y = np.dot(weights, x) + bias
    
    print(f"Original weights and bias: {weights}, {bias}")

    linear_regression(input_data=x, expected_output=y)
    batch_gradient_descent(num_iters=1000, learning_rate=0.0001, input_data=x, expected_output=y)
    stochastic_gradient_descent(num_iters=100000, learning_rate=0.001, input_data=x, expected_output=y)
    mini_batch_gradient_descent(num_iters=10000, batch_size=32, learning_rate=0.001, input_data=x, expected_output=y)
