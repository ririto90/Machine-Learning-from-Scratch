import numpy as np
from discriminants_v2 import GaussianDiscriminant


class Classifier:
    def __init__(self):
        self.model_params = {}
        pass
    
    def predict(self, x):
        raise NotImplementedError
    
    def fit(self, x, y):
        raise NotImplementedError

class Prior(Classifier):
    
    def __init__(self):
        self.model_params = {}
        pass

    def predict(self, x):
        raise NotImplementedError
    
    def fit(self, x, y):
        raise NotImplementedError

class DiscriminantClassifier(Classifier):

    def __init__(self):
        self.model_params = {}
        
        # Dictionary stores tuples of discriminant classes with respective features
        self.classes = {}
        
    def set_classes(self, *discs):
        # Store existing discriminant objects
        for disc in discs:
            self.classes[disc.name] = disc

    def fit(self, dataframe, label_key=['Labels'], default_disc=GaussianDiscriminant):
        # Get the unique column names of the dataframe
        labels = dataframe[label_key].unique()
        for label in labels:
            
            # Drops the selcted column for discriminaant anaylsis
            class_data = dataframe[dataframe[label_key] == label].drop(label_key, axis=1).values

            # Creates a class for discriminant analysis and adds it to the classifier class
            disc = default_disc(data=class_data, name=label)
            self.classes[label] = disc
    
    def predict(self, x):
        # Create two variables to store the best class and highest discriminant value
        best_class, best_discriminant_value = None, -np.inf

        # Loop through classes
        for class_name, disc in self.classes.items():

            # Calculate the discriminant value for each sample
            discriminant_value = disc.calc_discriminant(x)

            # Update the best class if a higher discriminant value is found
            if discriminant_value > best_discriminant_value:
                best_class, best_discriminant_value = class_name, discriminant_value

        return best_class

    def pool_variances(self):
        # Create variables to store the total # of samples and pooled covariance
        total_samples = 0
        pooled_cov = 0

        # Iterate each discriminant class
        for _, disc in self.classes.items():

            # Get the number of smaples in the discriminant 
            n_samples = disc.params['sigma'].shape[0]

            # Calculate the pooled covariance
            pooled_cov += (n_samples - 1) * disc.params['sigma']

            # Update the number of samples
            total_samples += n_samples

        # Divide the pooled covariance by the total degrees of freedom 
        pooled_cov /= (total_samples - len(self.classes))

        # Update each classes covariance
        for disc in self.classes.values():
            disc.params['sigma'] = pooled_cov