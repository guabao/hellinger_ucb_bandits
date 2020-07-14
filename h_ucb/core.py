__doc__ = "KL ucb algorithm"

import numpy
import pandas

# Importing the dataset
dataset = pandas.read_csv('Ads_CTR_Optimisation.csv')


class HUCB(object):
    def __init__(self, user_feature_hash, content_hash, convex_func=None, eps=1e-6):
        self.user_bin = user_feature_hash
        self.content_bin = content_hash
        self.total_reward = 1e-3 * numpy.ones([self.user_bin, self.content_bin])
        self.trials_arm = 1e-3 * numpy.ones([self.user_bin, self.content_bin])
        self.trials = numpy.ones(self.user_bin)
        self.parameter = {"p": numpy.zeros([self.user_bin, self.content_bin])}
        self.parameter["f(t)"] = numpy.log if convex_func is None else convex_func
        self.parameter["q"] = numpy.zeros([self.user_bin, self.content_bin])
        return None

    def update_reward_single_trial(self, user_hash, content_hash, reward):
        self.total_reward[user_hash, content_hash] += reward
        self.trials[user_hash] += 1
        self.trials_arm[user_hash, content_hash] += 1
        return None

    def get_total_reward(self):
        return numpy.sum(self.total_reward)

    def trial(self, user_hash):
        # if no pull yet, pull
        if numpy.any(self.trials_arm[user_hash] < 1):
            return numpy.argsort(self.trials_arm[user_hash])
        # update parameter
        #import pdb
        #pdb.set_trace()
        p = self.total_reward[user_hash] / self.trials_arm[user_hash]
        self.parameter["p"] = p[user_hash]
        f = self.parameter["f(t)"]
        #C = 1 - 1 / self.trials_arm[user_hash]
        # correct C
        C = 1 - numpy.exp(-0.25 * numpy.log(self.trials[user_hash]) / self.trials_arm[user_hash])
        #import pdb
        #pdb.set_trace()
        m1 = numpy.sqrt((1 - p) / p)
        m2 = (1 - C**2) / numpy.sqrt(p)
        m1_square = m1**2
        m2_square = m2**2
        a = (m1_square + 1)**2
        b = 2 * (m1_square * m2_square - m2_square - m1_square**2 - m1_square)
        c = (m2_square - m1_square) ** 2
        tmp1 = -b / (2*a)
        tmp2 = numpy.sqrt(b**2 - 4 * a * c) / (2 * a)
        q1 = tmp1 + tmp2
        q2 = tmp1 - tmp2
        q = numpy.where(q1 <= 1, q1, q2)
        self.parameter['q'][user_hash] = q
        return numpy.argsort(self.parameter['q'][user_hash])[::-1]
