from procgen.env import ProcgenGym3Env
from procgen import ProcgenEnv
from trainprocgen.common.env.procgen_wrappers import *

from procgen.domains import BossfightDomainConfig
import os

from gym3 import Interactive, unwrap, VideoRecorderWrapper
import gym3 

class ProcgenInteractive(Interactive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_state = None

    def _update(self, dt, keys_clicked, keys_pressed):
        if "LEFT_SHIFT" in keys_pressed and "F1" in keys_clicked:
            print("save state")
            self._saved_state = unwrap(self._env).get_state()
        elif "F1" in keys_clicked:
            print("load state")
            if self._saved_state is not None:
                unwrap(self._env).set_state(self._saved_state)
        super()._update(dt, keys_clicked, keys_pressed)

def make_interactive_bossfight_env(config: BossfightDomainConfig = None, 
                                   config_name: str = 'config.json', 
                                   video_directory: str = './boss-fight-results',
                                   n_envs = 1,
                                   **kwargs) -> ProcgenInteractive:
    info_key = "rgb"
    kwargs["render_mode"] = "rgb_array"

    if config is None:
        config = BossfightDomainConfig()
    configs_dir = os.path.join(os.path.abspath('.'), '.configs')
    if not os.path.isdir(configs_dir):
        os.mkdir(configs_dir)
    config_path = os.path.join(configs_dir, config_name)
    config.to_json(config_path)

    env = ProcgenGym3Env(n_envs, 'dc_bossfight', domain_config_path=config_path, **kwargs)
    # env = VecExtractDictObs(env, "rgb")
    # env = TransposeFrame(env)
    # env = ScaledFloatFrame(env)
    env = VideoRecorderWrapper(
                env=env, directory=video_directory, ob_key=None, info_key=info_key
            )
    env = gym3.ToBaselinesVecEnv(env)
    return env


if __name__ == '__main__':
    test_config = BossfightDomainConfig(min_n_rounds=2, max_n_rounds=2, min_boss_round_health=2, max_boss_round_health=2, boss_scale=.5)
    ia = make_interactive_bossfight_env(test_config)
    # ia.run()