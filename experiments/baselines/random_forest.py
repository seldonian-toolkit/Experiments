from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierBaseline():
    def __init__(self,n_estimators=100):
        """Implements a random forest classifier baseline for 
        a binary classification task"""
        self.model_name = "random_forest"
        self.n_estimators = n_estimators

    def train(self,X,Y):
        """Instantiate a new model instance and train (fit) 
        it to the training data, X,y
        :param X: features
        :type X: 2D np.ndarray 
        :param y: labels
        :type y: 1D np.ndarray
        """
        self.trained_model = RandomForestClassifier(n_estimators=self.n_estimators)
        self.trained_model.fit(X,Y) # parent method
        return None 

    def predict(self,theta,X):
        """Use the trained model to predict positive class probabilities
        theta isn't used here because there are no fitted parameters 
        in random forests. 
        :param theta: Model weights, None in this case
        :param X: features
        :type X: 2D np.ndarray 
        """
        probs_bothclasses = self.trained_model.predict_proba(X)
        probs_posclass = probs_bothclasses[:,1]
        return probs_posclass
