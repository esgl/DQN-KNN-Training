import numpy as np

from knn.annoy_dict import Annoy_Dict
class Q_Annoy_Dict:
    def __init__(self, config, num_action):
        self.config = config
        self.num_action = num_action
        self.dicts = []

        for a in range(self.num_action):
            new_dict = Annoy_Dict(config=self.config )
            self.dicts.append(new_dict)

    def _query(self, embs, actions, knn):
        dnd_embs = []
        dnd_vals = []
        dnd_embs_next = []
        dnd_actions = []
        dnd_terminal = []
        for i, a in enumerate(actions):
            e, v, t, e_n = self.dicts[a].query([embs[i]], knn)
            dnd_embs.append(e)
            dnd_actions.append([a] * knn)
            dnd_vals.append(v)
            dnd_terminal.append(t)
            dnd_embs_next.append(e_n)
        dnd_embs = np.array(dnd_embs)
        dnd_actions = np.array(dnd_actions)
        dnd_vals = np.array(dnd_actions)
        dnd_terminal = np.array(dnd_terminal)
        dnd_embs_next = np.array(dnd_embs_next)
        return dnd_embs, dnd_actions, dnd_vals, dnd_terminal, dnd_embs_next

    def query_actions(self, embs, actions, knn):
        assert np.ndim(embs) == 2
        embs_shape = np.shape(embs)

        dnd_embs = np.empty(embs_shape)
        dnd_actions = np.zeros(embs_shape[0])
        dnd_vals = np.zeros(embs_shape[0])
        dnd_ters = np.zeros(embs_shape[0])
        dnd_embs_next = np.zeros(embs_shape)

        for i, a in enumerate(actions):
            e, v, t, t_n = self.dicts[a].query_([embs[i]], knn)
            dnd_embs[i] = e[0]
            dnd_actions[i] = a
            dnd_vals[i] = v[0]
            dnd_ters[i] = t[0]
            dnd_embs_next[i] = t_n[0]
        return dnd_embs, dnd_actions, dnd_vals, dnd_ters, dnd_embs_next


    def query_(self, embs, knn):
        assert np.ndim(embs) == 2
        embs_shape = np.shape(embs)
        dnd_embs = np.empty((embs_shape[0] * self.num_action, embs_shape[1]))
        dnd_actions = np.zeros(embs_shape[0] * self.num_action)
        dnd_vals = np.zeros(embs_shape[0] * self.num_action)
        dnd_ters = np.zeros(embs_shape[0] * self.num_action)
        dnd_embs_next = np.empty((embs_shape[0] * self.num_action, embs_shape[1]))
        for i, a in enumerate(range(self.num_action)):
            e, v, t, e_n = self.dicts[a].query_(embs, knn)
            start = i * embs_shape[0]
            end = (i+1) * embs_shape[0]
            dnd_embs[start: end] = e
            dnd_actions[start: end] = np.array([a] * embs_shape[0])
            dnd_vals[start: end] = v
            dnd_ters[start: end] = t
            dnd_embs_next[start: end] = e_n
        return dnd_embs, dnd_actions, dnd_vals, dnd_ters, dnd_embs_next

    def query(self, embs, knn):

        dnd_embs = []
        dnd_actions = []
        dnd_vals = []
        dnd_terminal = []
        dnd_embs_next = []
        for i, a in enumerate(range(self.num_action)):
            e, v, t, e_n = self.dicts[a].query(embs, knn)
            dnd_embs.append(e)
            dnd_actions_tmp = []
            for j in range(e.shape[0]):
                dnd_actions_tmp.append([a] * knn)
            dnd_actions.append(dnd_actions_tmp)
            dnd_vals.append(v)
            dnd_terminal.append(t)
            dnd_embs_next.append(e_n)
        dnd_embs = np.array(dnd_embs)
        dnd_actions = np.array(dnd_actions)
        dnd_vals = np.array(dnd_vals)
        dnd_terminal = np.array(dnd_terminal)
        dnd_embs_next = np.array(dnd_embs_next)

        return dnd_embs, dnd_actions, dnd_vals, dnd_terminal, dnd_embs_next

    def add(self, embs, actions, vals, terminals, embs_next):
        for a in range(self.num_action):
            e = []
            v = []
            t = []
            e_n = []
            for i, _ in enumerate(embs):
                if actions[i] == a:
                    e.append(embs[i])
                    v.append(vals[i])
                    t.append(terminals[i])
                    e_n.append(embs_next[i])
            if e:
                self.dicts[a].add(e, v, t, e_n)
        return True

    @property
    def key_dimension(self):
        return self.config.knn_key_dim

    @property
    def action_capacity(self):
        action_capacity_ = []
        for a in range(self.num_action):
            action_capacity_.append(self.dicts[a].curr_capacity)
        return action_capacity_

    @property
    def min_capacity(self):
        min_val = 0
        for a in range(self.num_action):
            min_val = min(min_val, self.dicts[a].curr_capacity)

    @property
    def tot_capacity(self):
        tot_val = 0
        for a in range(self.num_action):
            tot_val += self.dicts[a].curr_capacity
        return tot_val

    def queryable(self, k):
        for a in range(self.num_action):
            if not self.dicts[a].queryable(k):
                return False
        return True

    def save(self, name):
        for a in range(self.num_action):
            self.dicts[a].save(name + "_dict_" + str(a) + ".npy")

    def load(self, name):
        for a in range(self.num_action):
            self.dicts[a].load(name + "_dict_" + str(a) + ".npy")

