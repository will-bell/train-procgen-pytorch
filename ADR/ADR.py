"""
Implementation of ADR (https://arxiv.org/pdf/1910.07113.pdf)
"""

import torch

MAX_SIZE_BUFFER = 1

class ADRParameter:
    """ Phi_L or Phi_H in the literature (section 5.2)
    
        Args:
            value (float): value of Phi
            delta (float): delta value to perturb phi when performance exceeds threshold
            boundaries (dict): keys = {'lower_bound', 'upper_bound'}
    """
    def __init__(self, value: float, delta: float, boundaries: dict,  thresh_low: float, thresh_high: float):
        self.value = value
        self.delta = delta #some delta to perturb Phi 
        self.boundaries = boundaries
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high
        # Each parameter has a buffer, which the performance will be averaged over
        # after its length reaches max_size_buffer
        self.performance_buffer = []
        self.max_size_buffer = MAX_SIZE_BUFFER # The papers default value is 240
        
        
    def return_val(self):
        return self.value

    def append_performance(self, performance: float):
        """Add performance of PPO algorithm to buffer

        Args:
            performance (float): performance of PPO algorithm
        """
        # Test data
        # self.performance_buffer = [0.0]* 240
        
        self.performance_buffer.append(performance)
        if len(self.performance_buffer) >= self.max_size_buffer:
            self.reach_max_buffer()
        
    def reach_max_buffer(self):
        """When buffer is at length = self.max_size_buffer,
            calculate average performance values and check against thresholds.
            
        Args:
            thresh_low (float): Low performance threshold
            thresh_high (float): High performance threshold

        """
        performance_average = torch.mean(torch.tensor(self.performance_buffer))
        self.performance_buffer = []

        if performance_average >= self.thresh_high:
            self.value = self.value + self.delta
        elif performance_average <= self.thresh_low:
            self.value = self.value - self.delta
        
        # print('--------Test Clipping--------')
        # print('preclip', self.value)
        self.value = torch.clip(torch.tensor(self.value), 
                                torch.tensor(self.boundaries['lower_bound']), 
                                torch.tensor(self.boundaries['upper_bound']))
        # print('after clip', self.value)

class ADREnvParameter:
    """
    Each Environment parameter contains 2 ADR Parameters (Phi_L, Phi_H)
    for a total of 2d parameters. (section 5.2)
    
    Args:
        value (float): starting value of phi's
        lower_bound (float): smallest lower bound possible
        upper_bound (float): largest upper bound possible
        delta (float): determines how much phi changes
    """
    
    def __init__(self, value:float, lower_bound: float, upper_bound: float, delta: float,  thresh_low: float, thresh_high: float):

        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        # adr_flag is to flag the SINGLE dimension/variable that will be Phi_L or Phi_R
        # if the flag is not set, then the value will be selected from U[Phi_L, Phi_R]
        self.adr_flag = False
        
        boundary_left  = dict(lower_bound = self.lower_bound, upper_bound = self.value)
        boundary_right = dict(lower_bound = self.value, upper_bound = self.upper_bound)

        self.phi_l = ADRParameter(self.value, delta, boundaries = boundary_left, thresh_low=thresh_low, thresh_high=thresh_high) # low
        self.phi_h = ADRParameter(self.value, delta, boundaries = boundary_right, thresh_low=thresh_low, thresh_high=thresh_high) # high
    
    def set_adr_flag(self, flag: bool):
        self.adr_flag = flag

    def sample(self):
        if self.adr_flag:
            return self.boundary_sample()
        else:
            return self.uniform_sample()

    def boundary_sample(self):
        """Select phi_l or phi_r with equal probability

        Returns:
            ADRParameter: phi_l or phi_r
        """
        x = torch.rand(1).item()
        print('probability: ', x)
        if x < 0.5:
            print('lower bound sample')
            lam = self.phi_l
        else:
            print('upper bound sample')
            lam = self.phi_h
        print('boundary sample: ', lam.return_val())
        return lam
            
    def uniform_sample(self):
        """ Sample with a uniform distribution between [phi_l, phi_h]

        Returns:
            float: value in Uniform Distribution
        """
        # sampled_event = self.lower_bound + (self.upper_bound - self.lower_bound)*torch.rand(1)[0]
        sampled_event = torch.FloatTensor(1).uniform_(self.phi_l.return_val(), self.phi_h.return_val())
        sampled_event = sampled_event.item()
        print(sampled_event)
        return sampled_event

    def evaluate_performance(self):
        # TODO maybe don't implement this here?
        pass
    

def adr_entrophy(env_parameters: list):
    d = len(env_parameters)
    phi_H = []
    phi_L = []
    
    for i in range(d):
        phi_H.append(env_parameters[i].phi_h.return_val())
        phi_L.append(env_parameters[i].phi_l.return_val())
    
    phi_H = torch.tensor(phi_H, dtype=torch.float)
    phi_L = torch.tensor(phi_L, dtype=torch.float)
    
    entrophy = torch.mean(torch.log(phi_H - phi_L))
    return entrophy

if __name__ == "__main__":
    torch.manual_seed(0)
    
    param1 = ADREnvParameter(value=7, lower_bound=1, upper_bound=100, delta=0.5, thresh_low = 0, thresh_high=10)
    lam = param1.boundary_sample()
    lam.append_performance(-1.0)
    print(param1.phi_h.return_val(), param1.phi_l.return_val())
    
    param2 = ADREnvParameter(value=8, lower_bound=1, upper_bound=100, delta=1, thresh_low = 0, thresh_high=10)
    lam = param2.boundary_sample()
    lam.append_performance(999.0)
    print(param2.phi_h.return_val(), param2.phi_l.return_val())

    param3 = ADREnvParameter(value=9, lower_bound=1, upper_bound=100, delta=0.5, thresh_low = 0, thresh_high=10)
    lam = param3.boundary_sample()
    lam.append_performance(-1.0)
    print(param3.phi_h.return_val(), param3.phi_l.return_val())

    entrophy = adr_entrophy([param1, param2, param3])
    print(entrophy)