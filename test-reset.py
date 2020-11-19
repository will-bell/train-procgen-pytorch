from procgen import ProcgenEnv, ProcgenGym3Env
from trainprocgen.common.env.procgen_wrappers import *
import pathlib
from procgen.domains import datetime_name, BossfightDomainConfig
import gym3


def test_reset():
    experiment_dir = pathlib.Path().absolute() / 'adr_experiments' / ('experiment-' + datetime_name())
    experiment_dir.mkdir(parents=True, exist_ok=False)

    config_dir = experiment_dir / 'domain_configs'
    config_dir.mkdir(parents=True, exist_ok=False)

    train_config_path = config_dir / 'train_config.json'
    BossfightDomainConfig().to_json(train_config_path)

    torch.set_num_threads(1)
    env = ProcgenEnv(num_envs=2, env_name='dc_bossfight', domain_config_path=str(train_config_path))
    env = VecExtractDictObs(env, "rgb")
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    step = 0
    while step < 1000:
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, done = env.observe()
        if any(done):
            print("env: done")
        print(f"step {step} reward {rew} done {done}")
        step += 1



if __name__ == '__main__':
    test_reset()