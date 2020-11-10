from abc import abstractmethod


class BaseAgent(object):
    """
    Class for the basic agent objects.
    To define your own agent, subclass this class and implement the functions below.
    """

    def __init__(self, 
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 num_checkpoints):
        """
        env: (gym.Env) environment following the openAI Gym API
        """
        self.env = env
        self.policy = policy
        self.logger = logger
        self.storage = storage
        self.device = device
        self.num_checkpoints = num_checkpoints
        
        self.t = 0

    @abstractmethod
    def predict(self, obs, hidden_state, done):
        """
        Predict the action with the given input 
        """
        pass

    @abstractmethod
    def update_policy(self):
        """
        Train the neural network model
        """
        pass

    @abstractmethod
    def train(self, num_timesteps: int):
        """
        Train the agent with collecting the trajectories
        """
        pass

    def evaluate(self):
        """
        Evaluate the agent
        """
        pass
