import gym
import random

from utils.util import rgb2gray, imresize
class Environment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name).unwrapped

        screen_width, screen_height, self.action_repeat, self.randon_start = \
            config.screen_width, config.screen_height, config.action_repeat, config.random_start

        self.display = config.display
        self.dims = (screen_width, screen_height)

        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self, from_random_game=False):
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in range(random.randint(0, self.randon_start - 1)):
            self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def render(self):
        if self.display:
            self.env.render()

    def after_act(self, action):
        self.render()

    @property
    def screen(self):
        return imresize(rgb2gray(self._screen)/255., self.dims)

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    @property
    def lives(self):
        return self.env.ale.lives()

