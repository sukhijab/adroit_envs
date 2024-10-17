def main(env_name: str = 'hammer'):
    if env_name == 'hammer':
        from adroit_envs.adroit_hammer import AdroitHandHammerEnv
        env = AdroitHandHammerEnv(render_mode='human')
    elif env_name == 'pen':
        from adroit_envs.adroit_pen import AdroitHandPenEnv
        env = AdroitHandPenEnv(render_mode='human')
    elif env_name == 'hand':
        from adroit_envs.adroit_hand import AdroitHandDoorEnv
        env = AdroitHandDoorEnv(render_mode='human')
    else:
        raise NotImplementedError
    state, info = env.reset()
    for i in range(100):
        state, reward, terminate, truncate, info = env.step(env.action_space.sample())
        env.render()
    env.close()


if __name__ == '__main__':
    env_names = ['hammer', 'pen', 'hand']
    for env_name in env_names:
        main(env_name)