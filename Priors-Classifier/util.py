import numpy as np

test_array = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 16]
]
test_array = np.array(test_array)

random_array = np.random.randint(low=0, high=100, size=3)
random_matrix = np.random.randint(0, 100, size=(3, 3))
print(random_matrix)

def mean(np_array):

    total_sum = 0
    count = 0

    for i in np_array:
        total_sum += i
        count += 1

    mean = total_sum / count

    return mean


def stdev(np_array, mu=None):

    if mu is not None:
        calculate_mean = mu 
    else:
        calculate_mean = mean(np_array)

    # Subtract the mean from each point and square the result
    squared_diff = [(x - calculate_mean) ** 2 for x in np_array]

    # Calculate the mean of the squared difference
    squared_diff_mean = mean(squared_diff)

    # Sqare root the mean
    stdev = squared_diff_mean ** 0.5

    return stdev
    

def sampleMean(np_array):

    sample_means = []
    
    # Get the sum of each feature
    for i in range(len(np_array)):
        feature_sum = 0
        for j in range(len(np_array[i])):
            feature_sum += np_array[i][j]
        
        # Divide each sum by the number of data points
        sample_mean = feature_sum / len(np_array[i])

        sample_means.append(sample_mean)
  
    return sample_means


def covariance(np_array):

    num_rows = np_array.shape[0]
    num_columns = np_array.shape[1]

    means = np.zeros(num_columns)

    # Calculate the means
    for col in range(num_columns):
        total = 0
        for row in np_array:
            total += row[col]

        means[col] = total / len(np_array)

    # Calculate the deviations
    deviations = np_array - means

    product_of_deviations = np.zeros((num_columns, num_columns))

    # Sum the products of the deviations
    for i in range(num_columns):
        for j in range(num_columns):
            product_of_deviations[i, j] = np.sum(deviations[:, i] * deviations[:, j])

    # Divide the product of deviations by (n-1)
    covariance_matrix = product_of_deviations / (num_rows - 1)
    
    return covariance_matrix

def mean_test(test_array):

    output1 = mean(test_array)
    output2 = np.mean(test_array)

    if np.array_equal(output1, output2):
        result = True
    else:
        result = False

    return result

def stdev_test(test_array):

    output1 = stdev(test_array)
    output2 = np.std(test_array)

    if np.array_equal(output1, output2):
        result = True
    else:
        result = False

    return result

def sample_mean_test(test_array):

    output1 = sampleMean(test_array)
    output2 = np.mean(test_array, axis=1)

    if np.array_equal(output1, output2):
        result = True
    else:
        result = False

    return result

def covariance_test(test_array):

    output1 = covariance(test_array)
    output2 = np.cov(test_array, rowvar=False)

    rounded_covariance_matrix = np.round(output1, 2)
    rounded_covariance_matrix2 = np.round(output2, 2)

    print(output1)
    print(output2)

    if np.array_equal(rounded_covariance_matrix, rounded_covariance_matrix2):
        result = True
    else:
        result = False

    return result


print(f"{mean.__name__}: {mean_test(random_array)}")

print(f"{stdev.__name__}: {stdev_test(random_array)}")

print(f"{sample_mean_test.__name__}: {sample_mean_test(random_matrix)}")

print(f"{covariance.__name__}: {covariance_test(random_matrix)}")