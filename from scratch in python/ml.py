import numpy as np

class linearregression():
    def __init__(self,alpha = 0.01 , epocs = 1000):
        self.alpha = alpha
        self.epocs = epocs
        self.weight = None
        self.bais =  None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weight = np.zeros((n_features,1))
        self.bais = 0

        for i in range(self.epocs):

            y_pred = np.dot(X,self.weight) + self.bais

            dw = (1/n_samples) * np.dot(X.T,(y_pred-y)) 
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weight = self.weight - self.alpha * dw 
            self.bais = self.bais - self.alpha * db


    def predict(self,X):
        return np.dot(X,self.weight)+self.bais


class logisticregression(linearregression):

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def predict(self, X):
        ans = super().predict(X)
        return self.sigmoid(ans)


np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

obj = linearregression()
obj.fit(X,y)


x_pred = 2.2 * np.random.rand(10, 1)

obj1 = logisticregression()
obj1.fit(X,y)
ans = obj1.predict(x_pred)
print(ans)