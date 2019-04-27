# Module to handle program results metrics.

class FuzzyMatrix:
    def __init__(self):
        self.matrix = {}

    def add(self, predicted, expected):
        key = (predicted, expected)
        self.matrix[key] = self.matrix.get(key, 0) + 1

    def trace(self):
        matrixList = (self.matrix.items())
        return sum(map(lambda x: x[1], filter(lambda x: x[0][0] == x[0][1], matrixList)))

    def sum(self):
        return sum(self.matrix.values())

    def accuracy(self):
        return self.trace()/float(self.sum())

    def precision(self, label):
        predictedLabel = sum(map(lambda x: x[1], filter(lambda x: x[0][0] == label, self.matrix.items())))
        predictedCorrectly = self.matrix.get((label, label), 0)
        try:
            return predictedCorrectly/float(predictedLabel)
        except ZeroDivisionError:
            print("[Warn] Não foi possível computar a precisão da classe %s, o perceptron nunca previu essa classe" % str(label))
            return 0

    def sensitivity(self, label):
        expectedLabel = sum(map(lambda x: x[1], filter(lambda x: x[0][1] == label, self.matrix.items())))
        predictedCorrectly = self.matrix.get((label, label), 0)
        return predictedCorrectly/float(expectedLabel)
