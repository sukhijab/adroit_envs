def main():
    from adroit_envs.adroit_hand import AdroitHandDoorEnv
    from gymnasium.wrappers import AddRenderObservation
    from gymnasium.wrappers import FrameStackObservation
    env = AdroitHandDoorEnv(render_mode='rgb_array')
    env = FrameStackObservation(AddRenderObservation(env=env, render_key='image'), stack_size=3)
    state, info = env.reset()
    for i in range(100):
        state, reward, terminate, truncate, info = env.step(env.action_space.sample())


if __name__ == '__main__':
    main()