import unittest
from NN_pr import NN
from NN_pr import activation_function as af
import numpy as np


class Test(unittest.TestCase):
    trainData = np.random.rand(100, 784)
    trainLabel = np.random.randint(10, size=(100,), dtype=np.uint8)
    train = [trainData, trainLabel]

    testData = np.random.rand(50, 784)
    testLabel = np.random.randint(10, size=(50,), dtype=np.uint8)
    test = [testData, testLabel]
    nn = NN.NN(training=train, testing=test, lr=0.003, mu=.99, minibatch=100, dropout=1, disableLog=True)

    def test_num_hidden_layers(self):
        self.nn.addLayers([10, 10], ['relu', 'relu'])
        self.assertEqual(self.nn.nHidden, 2)

    def test_relu(self):
        z = np.zeros((5, 5))
        r = af.ReLU(z)
        self.assertTrue(r.min() == r.max())
        
    def test_sigmoid_zero(self):
        self.assertEqual(af.sigmoid(0), 0.5)
        
    def test_sigmoid_max(self):
        self.assertAlmostEqual(af.sigmoid(1000), 1)
        
    def test_sigmoid_min(self):
        self.assertAlmostEqual(af.sigmoid(-1000), 0)
        
    
        
    
        



if __name__ == '__main__':
    unittest.main()



