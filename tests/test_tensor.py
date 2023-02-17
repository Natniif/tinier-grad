import numpy as np 
import torch 
import unittest 
from tinier_grad.tensor import Tensor

a_tg = Tensor(np.array([[1,2],[3,4]]))
b_tg = Tensor(np.randn([[5,6],[7,8]]))

a_pt = torch.Tensor([[1,2],[3,4]])
b_pt = torch.Tensor([[5,6],[7,8]])

class TestGrad(unittest.TestCase): 
    def test_ops(self):
        def test_tg(a, b):
            # test add
            c = a + b
            # test mul 
            d = c * a 
            # pow
            e = d ** 2
            # matmul
            f = e @ b
            # relu 
            g = f.relu()
            #sigmoid
            h = g.sigmoid()
            # do backward pass 
            h.backward()
            return a, b, h


        def test_pt(a, b):  
            # test add
            c = a + b
            # test mul 
            d = c * a 
            # pow
            e = d ** 2
            # matmul
            f = e @ b
            # relu 
            g = f.relu()
            #sigmoid
            h = g.sigmoid()
            # do backward pass
            h.backward()
            
            return a, b, h

        tolerance = 1e-5

        a_tg, b_tg, h_tg = test_tg(a_tg, b_tg) 
        a_pt, b_pt, h_pt = test_pt(a_pt, b_pt)

        # forward pass
        self.assertTrue(abs(h_tg.data - h_pt.data.item()) < tolerance)

        # back pass
        self.assertTrue(abs(a_tg.data - a_pt.data.item()) < tolerance)
        self.assertTrue(abs(b_tg.data - b_pt.data.item()) < tolerance) 
    
if __name__ == '__main__': 
    unittest.main()
