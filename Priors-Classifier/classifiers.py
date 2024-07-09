import pandas as pd
import numpy as np

np.random.seed(67)

total_samples = 1000

stress_samples = 430
not_stress_samples = total_samples - stress_samples

stress_heart_rates = np.random.normal(110, 10, stress_samples)
not_stress_heart_rates = np.random.normal(65, 5, not_stress_samples)

class_labels = ["Stress"] * stress_samples + ["Not Stress"] * not_stress_samples

data = pd.DataFrame({
    "Heart-Rate": np.concatenate((stress_heart_rates, not_stress_heart_rates)),
    "Class": class_labels
})

print(data.head())

grouped_data = data.groupby("Class")
description = grouped_data["Heart-Rate"].describe()
print(description)

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
        if not self.model_params:
            raise NotImplementedError
        
        # Get the class with the highest probability
        max_prob_class = max(self.model_params, key=self.model_params.get)
        
        # Predict each label with that class
        predicted_labels = [[sample, max_prob_class] for sample in x]

        df = pd.DataFrame(predicted_labels, columns=['Sample', 'Max_Prob_Class'])

        return df
        
    
    def fit(self, x, y):        
        # Get the total number of samples and unique classes with their counts
        total_samples = len(x)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(unique_classes, class_counts)

        # Calculate the probability of each class appearing against all the data
        for class_label, class_count in zip(unique_classes, class_counts):
            prior_probability = class_count / total_samples
            print(prior_probability)
            self.model_params[class_label] = prior_probability

        return self.model_params

classifier = Prior()
classifier.fit(data['Heart-Rate'], data['Class'])
result = classifier.predict(data['Heart-Rate'])
result.info()