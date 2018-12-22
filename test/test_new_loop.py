import unittest
import numpy as np
import sys

sys.path.append("/home/lss/research")
from src.new_loop import LocalOutlierProbability as loop
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler

class TestNewLoOP(unittest.TestCase):

    def setUp(self):
        self.smtp = fetch_kddcup99(subset='smtp', percent10=True).data
        self.smtp = StandardScaler().fit_transform(self.smtp)

    def test_insert(self):
        model = loop(self.smtp, 1000000, extent=3, n_neighbors=3).fit()
        model_score = model.local_outlier_probabilities
        with open('model_score', 'w') as file:
            for val in model_score:
                file.write(str(val)+'\n')

        test_model = loop(self.smtp[:-1], 1000000, extent=3, n_neighbors=3).fit()
        test_data = self.smtp[-1]
        test_model.insert(test_data)
        test_model_score = test_model.local_outlier_probabilities
        with open('test_model_score', 'w') as file:
            for val in test_model_score:
                file.write(str(val)+'\n')
        size = len(model_score)
        for i in range(size):
            self.assertEqual(model_score[i], test_model_score[i])
    
    def test_nds(self):
        model = loop(self.smtp[:-1], 99, extent=3, n_neighbors=3).fit()
        test_data = self.smtp[-1]
        model.insert(test_data)
        print(model.local_outlier_probabilities[-1])



if __name__ == "__main__":
    unittest.main()







