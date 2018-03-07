from env.Environment import Environment

class SimpleGymEnvironment(Environment):
    def __init__(self, config):
        super(SimpleGymEnvironment, self).__init__(config)

    def act(self, action, is_training=True):
        self._step(action)
        self.after_act(action)
        return self.state
    