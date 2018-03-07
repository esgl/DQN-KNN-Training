from env.Environment import Environment
class GymEnvironment(Environment):
    def __init__(self, config):
        super(GymEnvironment, self).__init__(config)

    def act(self, action, is_training=True):
        cumulated_reward = 0
        start_lives = self.lives

        for _ in range(self.action_repeat):
            self._step(action)
            cumulated_reward = cumulated_reward + self.reward

            if is_training and start_lives > self.lives:
                cumulated_reward -= 1
                self.terminal = True

            if self.terminal:
                break
        self.reward = cumulated_reward
        self.after_act(action)
        return self.state

