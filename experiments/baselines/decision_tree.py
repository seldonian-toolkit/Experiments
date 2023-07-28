from sklearn.tree import DecisionTreeClassifier

class DecisionTreeClassifierBaseline():
    def __init__(self,max_depth=None):
        """Implements a random forest classifier baseline for 
        a binary classification task"""
        self.model_name = "decision_tree"
        self.max_depth = max_depth

    def train(self,X,Y):
        """Instantiate a new model instance and train (fit) 
        it to the training data, X,y
        :param X: features
        :type X: 2D np.ndarray 
        :param y: labels
        :type y: 1D np.ndarray
        """
        self.trained_model = DecisionTreeClassifier(max_depth=self.max_depth)
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

