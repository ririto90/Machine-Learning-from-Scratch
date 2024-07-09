import numpy as np


class Discriminant:
    def __init__(self):
        self.params = {}
        self.name = ''
        
    def fit(self, data):
        raise NotImplementedError
    
    def calc_discriminant(self, x):
        raise NotImplementedError


class GaussianDiscriminant(Discriminant):
    def __init__(self, data = None, prior=0.5, name = 'Not Defined'):
        self.pi = np.pi
        self.params = {'mu':None, 'sigma':None, 'prior':prior}
        if data is not None:
            self.fit(data)
        self.name = name
    
    def fit(self, data):
        self.params['mu']    = np.mean(data)
        self.params['sigma'] = np.std(data)
        
    def calc_discriminant(self, x):
        mu = self.params['mu']
        sigma = self.params['sigma']
        prior = self.params['prior']
        
        # Calculate the natural log
        log_prior = np.log(prior)

        # Calculate normalization factor
        normalization = -0.5 * np.log(2 * self.pi * sigma ** 2)

        # Calculate the squared difference between the input sample and mean, scaled by sigma
        squared_diff = -((x - mu) ** 2) / (2 * sigma ** 2)

        discriminant = normalization + squared_diff + log_prior
        return discriminant


class MultivariateGaussian(Discriminant):
    
    def __init__(self, data=None, prior=0.5, name = 'Not Defined'):
        self.pi = np.pi
        self.params = {'mu':None, 'sigma':None, 'prior':prior}
        if data is not None:
            self.fit(data)
        self.name = name
        
    def fit(self, data):
        self.params['mu']    = np.average(data, axis=0)
        self.params['sigma'] = np.cov(data.T, bias=True)
        
    def calc_discriminant(self, x):

        mu, sigma, prior = self.params['mu'], self.params['sigma'], self.params['prior']

        # Get the number of features
        dimensions = len(mu)

        # Calculate the inverse of the covariance matrix
        sigma_inverse = np.linalg.inv(sigma)

        # Calculate the difference between the input sample and the mean vector
        difference = x - mu


        # Calculate the natural log
        log_prior = np.log(prior)

        # Calculate the quadratic which is the squared distance of the sample from the mean
        quadratic = -0.5 * np.dot(difference.T, np.dot(sigma_inverse, difference))

        # Calculate normalization factor
        normalization = -0.5 * dimensions * np.log(2 * self.pi)

        # Calculate the log of the determinant of the covariance matrix
        covar_determinant = -0.5 * np.log(np.linalg.det(sigma))

        discriminant = quadratic + normalization + covar_determinant + log_prior
        return discriminant