# Module to calculate program results and metrics.

class FuzzyMatrix:
    def __init__(self):
        self.matrix = {}

    # Add pair (prediction, expectation) to the FuzzyMatrix count.
    def add(self, predicted, expected):
        key = (predicted, expected)
        self.matrix[key] = self.matrix.get(key, 0) + 1

    # Count all the correct predictions inside the FuzzyMatrix.
    def trace(self):
        matrixList = (self.matrix.items())
        return sum(map(lambda x: x[1], filter(lambda x: x[0][0] == x[0][1], matrixList)))

    # Count all the predictions (both correct and incorrect) inside the FuzzyMatrix.
    def sum(self):
        return sum(self.matrix.values())

    # Compute the accuracy of the predictive system.
    def accuracy(self):
        return self.trace()/float(self.sum())

    # Compute the precision of the predictive system for a given label.
    def precision(self, label):
        predictedLabel = sum(map(lambda x: x[1], filter(lambda x: x[0][0] == label, self.matrix.items())))
        predictedCorrectly = self.matrix.get((label, label), 0)
        try:
            return predictedCorrectly/float(predictedLabel)
        except ZeroDivisionError:
            print("[Warning] Unable to compute the precision for label %s. It was never predicted by the system!" % str(label))
            return 0

    # Compute the sensitivity of the predictive system for a given label.
    def sensitivity(self, label):
        expectedLabel = sum(map(lambda x: x[1], filter(lambda x: x[0][1] == label, self.matrix.items())))
        predictedCorrectly = self.matrix.get((label, label), 0)
        try:
            return predictedCorrectly/float(expectedLabel)
        except ZeroDivisionError:
            print("[Warning] Unable to compute the sensitivity for label %s. It was never expected by the system!" % str(label))
            return 0
