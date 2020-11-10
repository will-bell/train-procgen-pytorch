"""
Implementation of ADR (https://arxiv.org/pdf/1910.07113.pdf)
"""

import torch


class ADRParameter:
    """ Phi_L or Phi_H in the literature (section 5.2)
    """
    def __init__(self, value: float, boundaries: dict):
        self.value = value
        self.boundaries = boundaries
    
        # Each parameter has a buffer, which the performance will be averaged over
        # after its length reaches max_size_buffer
        self.performance_buffer = []
        self.max_size_buffer = 240 # The papers default value, might be a hyper parameter to tune later
        
        self.delta = 0.5 #some delta to perturb Phi 
        
    def sample(self):
        """
        Sample with a uniform distribution between [lower_bound, upper_bound]
        """
        
        # sampled_event = self.lower_bound + (self.upper_bound - self.lower_bound)*torch.rand(1)[0]
        sampled_event = torch.FloatTensor(1).uniform_(
                                    self.boundaries['lower_bound'], self.boundaries['upper_bound'])
        sampled_event = sampled_event.item()
        print(sampled_event)
        return sampled_event
    
    def reach_max_buffer(self, p, thresh_low, thresh_high):
        """When buffer is at length = self.max_size_buffer,
            calculate average performance values and check against thresholds.
            
        Args:
            p (float): Performance of PPO
            thresh_low ([type]): [description]
            thresh_high ([type]): [description]

        """
        if len(self.performance_buffer) > self.max_size_buffer:
            performance_average = torch.mean(self.performance_buffer)
            self.performance_buffer = []

            if P >= thresh_high:
                self.value = self.value + self.delta
            elif p <= thresh_low:
                self.value = self.value - self.delta

class ADREnvParameter:
    """
    Each Environment parameter contains 2 ADR Parameters (Phi_L, Phi_H)
    for a total of 2d parameters. (section 5.2)
    """
    
    def __init__(self, lower_bound: float,  boundary: float, upper_bound: float):
        self.boundary = boundary
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        self.phi_l = ADRParameter(self.lower_bound, self.boundary, self.upper_bound) # low
        self.phi_h = ADRParameter(self.lower_bound, self.boundary, self.upper_bound) # high
        
    def boundary_sample(self):
        x = torch.rand(1).item()
        print(x)
        if x < 0.5:
            print('lower bound sample')
            self.phi_l.sample()
        else:
            print('upper bound sample')
            self.phi_r.sample()
    
    def evaluate_performance(self):
        pass
    


class Uniform:
    def __init__(self, param1: ADREnvParameter, param2: ADREnvParameter):
        pass


if __name__ == "__main__":
    torch.manual_seed(0)
    
    param = ADREnvParameter( lower_bound=5, boundary=7, upper_bound=10)
    param.boundary_sample()