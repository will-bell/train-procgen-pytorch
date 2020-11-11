# Workout around for windows...
# Need to append parent directory to sys path to get ADR module
import sys
sys.path.append('..')

import unittest

from ADR import *

class TestEntrophyMethod(unittest.TestCase):
    
    def test_entrophy(self):
        
        param1 = ADREnvParameter(value=7, lower_bound=1, upper_bound=100, delta=0.5, thresh_low = 0, thresh_high=10)
        lam = param1.boundary_sample()
        # Repeat adding fake reward so it triggers a reach_max_buffer call
        for i in range(MAX_SIZE_BUFFER):
            lam.append_performance(-1.0)
        # print(param1.phi_h.return_val(), param1.phi_l.return_val())
        
        param2 = ADREnvParameter(value=8, lower_bound=1, upper_bound=100, delta=1, thresh_low = 0, thresh_high=10)
        lam = param2.boundary_sample()
        # Repeat adding fake reward so it triggers a reach_max_buffer call
        for i in range(MAX_SIZE_BUFFER):
            lam.append_performance(999.0)
        # print(param2.phi_h.return_val(), param2.phi_l.return_val())


        param3 = ADREnvParameter(value=9, lower_bound=1, upper_bound=100, delta=0.5, thresh_low = 0, thresh_high=10)
        lam = param3.boundary_sample()
        # Repeat adding fake reward so it triggers a reach_max_buffer call
        for i in range(MAX_SIZE_BUFFER):
            lam.append_performance(-1.0)
        # print(param3.phi_h.return_val(), param3.phi_l.return_val())

        entrophy = adr_entrophy([param1, param2, param3])
        # print(entrophy)
        
        # d = 3
        # 1/d \sum_{i=1}^{d} log(phi_ih - phi_il)
        self.assertEqual(entrophy,
                         (torch.log(torch.tensor(0.5)) + 
                            torch.log(torch.tensor(1.0)) + 
                            torch.log(torch.tensor(0.5))) / 3)

if __name__ == '__main__':
    torch.manual_seed(0)
    unittest.main()