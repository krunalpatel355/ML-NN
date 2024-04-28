import numpy as np 

class KNN():
    def __init__(self,k=5):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X_test):
        y_pred = []

        for x in X_test:
            distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
            nearest_neighbors = np.argsort(distances)[:self.k]
            y_nearest = [self.y_train[i] for i in nearest_neighbors]
            y_pred.append(max(set(y_nearest), key=y_nearest.count))

        return np.array(y_pred)






X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])
X_test = np.array([[2, 3.5]])



model = KNN()
model.fit(X,y)
ans = model.predict(X_test)
print(ans)