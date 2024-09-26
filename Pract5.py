import numpy as np

class NaiveBayes:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.mean = {}
        self.variance = {}
        self.priors = {}

        for c in self.classes:
            X_c = X_train[y_train == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X_train.shape[0]
    
    def gaussian_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.variance[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            posterior = np.sum(np.log(self.gaussian_pdf(c, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

# Example Usage:
X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [8, 9]])
y_train = np.array([0, 0, 0, 1, 1])
X_test = np.array([[2, 3], [6, 7]])

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
print(predictions)  # Output will be predicted labels for X_test
