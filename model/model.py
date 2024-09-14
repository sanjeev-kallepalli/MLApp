from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve
import joblib


class ModelSuite:
    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
    def logistic_regression(self):
        self.lr = LogisticRegression()
        self.lr.fit(self.X_train, self.y_train)
        return self.lr
    def decision_tree(self):
        # make models like this and store in model suite
        pass
    def evaluate(self, model):
        # we take model as input as we shall use the same function with different models
        # not just logistic regression.
        return classification_report(self.y_val, model.predict(self.X_val))
    

class PredictFromModel:
    def __init__(self, data, model_path):
        self.data = data
        self.model_path = model_path

    def predict(self):
        model = joblib.load(self.model_path)
        return model.predict(self.data)