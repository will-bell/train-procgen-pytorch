"""
Implementation of Automatic Domain Randomization (https://arxiv.org/pdf/1910.07113.pdf)
"""

import torch
from trainprocgen.common.storage import Storage

import os
import pandas as pd

MAX_SIZE_BUFFER = 1

class ADRParameter:
    """ Phi_L or Phi_H in the literature (section 5.2)
    
        Args:
            value (float): value of Phi
            step_size (float): step_size value to perturb phi when performance exceeds threshold
            boundaries (dict): keys = {'lower_bound', 'upper_bound'}
    """
    def __init__(self, value: float, step_size: float, boundaries: dict,  thresh_low: float, thresh_high: float):
        self.value = value
        self.step_size = step_size #some delta to perturb Phi 
        self.boundaries = boundaries
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high
        # Each parameter has a buffer, which the performance will be averaged over
        # after its length reaches max_size_buffer
        self.performance_buffer = []
        self.max_size_buffer = MAX_SIZE_BUFFER # The papers default value is 240
        
        # self.storage = Storage()
        
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
            return True
        return False
        
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
            self.value = self.value + self.step_size
        elif performance_average <= self.thresh_low:
            self.value = self.value - self.step_size
        
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
        step_size (float): determines how much phi changes
    """
    
    def __init__(self, name: str, value: float, lower_bound: float, upper_bound: float, 
                    step_size: float,  thresh_low: float, thresh_high: float, is_continuous: bool):
        self.name = name
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_continuous = is_continuous
        
        # adr_flag is to flag the SINGLE dimension/variable that will be Phi_L or Phi_R
        # if the flag is not set, then the value will be selected from U[Phi_L, Phi_R]
        self.adr_flag = False
        
        boundary_left  = dict(lower_bound = self.lower_bound, upper_bound = self.value)
        boundary_right = dict(lower_bound = self.value, upper_bound = self.upper_bound)

        self.phi_l = ADRParameter(self.value, step_size, boundaries = boundary_left, thresh_low=thresh_low, thresh_high=thresh_high) # low
        self.phi_h = ADRParameter(self.value, step_size, boundaries = boundary_right, thresh_low=thresh_low, thresh_high=thresh_high) # high
    
    def set_adr_flag(self, flag: bool):
        self.adr_flag = flag

    def sample(self, probability: float):
        if self.adr_flag:
            return self.boundary_sample(probability)
        else:
            return self.uniform_sample()

    def boundary_sample(self, probability: float):
        """Select phi_l or phi_r with equal probability

        Returns:
            ADRParameter: phi_l or phi_r
        """
        print('probability: ', probability)
        if probability < 0.5:
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
        if self.phi_l.return_val() == self.phi_h.return_val():
            return self.phi_l.return_val()
        
        if self.is_continuous:
            # sampled_event = self.lower_bound + (self.upper_bound - self.lower_bound)*torch.rand(1)[0]
            sampled_event = torch.FloatTensor(1).uniform_(self.phi_l.return_val(), self.phi_h.return_val())
            sampled_event = sampled_event.item()
        else:
            # sampled_event = torch.randint(self.phi_l.return_val(), self.phi_h.return_val(), size=(1,)).item()
            torch.LongTensor(1).random_(self.phi_l.return_val(), self.phi_h.return_val())

        print(sampled_event)
        return sampled_event

    def get_param(self, is_high: bool):
        if is_high:
            return self.phi_h
        return self.phi_l
    
    def evaluate_performance(self):
        # TODO maybe don't implement this here?
        pass
    
class ADRManager:
    def __init__(self, parameters_list: list):
        self.parameters_list = parameters_list
        self.parameters = {}
        for param in parameters_list:
            self.parameters[param.name] = param
        
        column_names = [param.name + '_low' for param in self.parameters_list] \
                    + [param.name + '_hi' for param in self.parameters_list]
        self.result_dataframe = pd.DataFrame(columns=column_names)
        self.running_dataframes = []
    
    def append_performance(self, feature_ind: int, is_high: bool, performance: float):
        """Append performance to performance buffer of boundary sampled feature

        Args:
            feature_ind (int): feature index
            is_high (bool): high or low flag for choosing between phi_L or phi_H
            performance (float): performance calculation
        """
        reached_max_buffer = self.parameters_list[feature_ind].get_param(is_high).append_performance(performance)
        
        if reached_max_buffer:
            d = dict()
            for param in self.parameters:
                d[param.name + '_low'] = param.get_param(is_high=False).value
                d[param.name + '_hi'] = param.get_param(is_high=True).value

            d['performance'] = performance
            self.running_dataframes.append(d)
            
            # Add new dataframes to running list of dataframes
            # Either concat now or write to csv
            if len(self.running_dataframes >= 10):
                self.running_dataframes = pd.DataFrame(self.running_dataframes)
                # frames = [self.result_dataframe, self.running_dataframes]
                # self.result_dataframe = pd.concat(frames)
                
                # If result file already exists, append to the file
                # Or else write a new file
                if os.path.exists('results.csv'):  
                    self.running_dataframes.to_csv('results.csv', mode='a', header=False)
                else:
                    self.running_dataframes.to_csv('results.csv')
                    
                # Reset dataframes
                self.running_dataframes = []

    def select_boundary_sample(self):
        """ Selects feature index to boundary sample. Also uniformly selects a probability
        between 0 < x < 1 for low and high of phi

        Returns:
            int: feature index
            float: probability to choose between phi_L and phi_H
        """
        
        # Reset adr flag
        for param in self.parameters:
            self.parameters[param].set_adr_flag(False)

        feature_to_boundary_sample = torch.randint(0, len(self.parameters_list), size=(1,)).item()
        self.parameters_list[feature_to_boundary_sample].set_adr_flag(True)
        
        probability = torch.rand(1).item() # 0 <= x < 1

        return feature_to_boundary_sample, probability
    
    def create_config(self, feature_to_boundary_sample: int, probability: float):
        config = {}
    
        # TODO how to handle append_performance in ADRParameter ??
        for i, env_parameter in enumerate(self.parameters_list):
            _lambda = env_parameter.sample(probability)
            if i == feature_to_boundary_sample:
                # boundary_sample returns ADRParameter, so call return_val to get its value
                _lambda = _lambda.return_val()
                
            config[env_parameter.name] = _lambda
        return config
        
    def adr_entropy(self):
        """ Calculate ADR Entrophy

        Returns:
            float: entropy =  1/d \sum_{i=1}^{d} log(phi_ih - phi_il)
        """
        d = len(self.parameters_list)
        phi_H = []
        phi_L = []
        
        for i in range(d):
            phi_H.append(self.parameters_list[i].phi_h.return_val())
            phi_L.append(self.parameters_list[i].phi_l.return_val())
        
        phi_H = torch.tensor(phi_H, dtype=torch.float)
        phi_L = torch.tensor(phi_L, dtype=torch.float)
        
        entropy = torch.mean(torch.log(phi_H - phi_L))
        return entropy

if __name__ == "__main__":
    torch.manual_seed(0)
    
    param1 = ADREnvParameter(value=7, lower_bound=1, upper_bound=100, step_size=0.5, thresh_low = 0, thresh_high=10)
    lam = param1.boundary_sample()
    lam.append_performance(-1.0)
    print(param1.phi_h.return_val(), param1.phi_l.return_val())
    
    param2 = ADREnvParameter(value=8, lower_bound=1, upper_bound=100, step_size=1, thresh_low = 0, thresh_high=10)
    lam = param2.boundary_sample()
    lam.append_performance(999.0)
    print(param2.phi_h.return_val(), param2.phi_l.return_val())

    param3 = ADREnvParameter(value=9, lower_bound=1, upper_bound=100, step_size=0.5, thresh_low = 0, thresh_high=10)
    lam = param3.boundary_sample()
    lam.append_performance(-1.0)
    print(param3.phi_h.return_val(), param3.phi_l.return_val())

    entropy = adr_entrophy([param1, param2, param3])
    print(entropy)