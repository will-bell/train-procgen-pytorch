# Work around for windows...
# Need to append parent directory to sys path to get ADR module
import sys
sys.path.append('..')

import unittest

from ADR import *

class TestEntropyMethod(unittest.TestCase):
    
    def test_entropy(self):
        
        param1 = ADREnvParameter(name='param1', value=7, lower_bound=1, upper_bound=100, step_size=0.5, thresh_low = 0, thresh_high=10, is_continuous=True)
        lam = param1.boundary_sample()
        # Repeat adding fake reward so it triggers a reach_max_buffer call
        for i in range(MAX_SIZE_BUFFER):
            lam.append_performance(-1.0)
        # print(param1.phi_h.return_val(), param1.phi_l.return_val())
        
        param2 = ADREnvParameter(name='param2', value=8, lower_bound=1, upper_bound=100, step_size=1, thresh_low = 0, thresh_high=10, is_continuous=True)
        lam = param2.boundary_sample()
        # Repeat adding fake reward so it triggers a reach_max_buffer call
        for i in range(MAX_SIZE_BUFFER):
            lam.append_performance(999.0)
        # print(param2.phi_h.return_val(), param2.phi_l.return_val())


        param3 = ADREnvParameter(name='param3', value=9, lower_bound=1, upper_bound=100, step_size=0.5, thresh_low = 0, thresh_high=10, is_continuous=True)
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