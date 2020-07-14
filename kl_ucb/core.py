__doc__ = "KL ucb algorithm"

import numpy
import pandas

# Importing the dataset
dataset = pandas.read_csv('Ads_CTR_Optimisation.csv')


class KLUCB(object):
    def __init__(self, user_feature_hash, content_hash, convex_func=None, eps=1e-6):
        self.user_bin = user_feature_hash
        self.content_bin = content_hash
        self.total_reward = 1e-8 * numpy.ones([self.user_bin, self.content_bin])
        self.trials_arm = 2e-8 * numpy.ones([self.user_bin, self.content_bin])
        self.trials = numpy.ones(self.user_bin)
        self.parameter = {"p": numpy.zeros([self.user_bin, self.content_bin])}
        self.parameter["f(t)"] = numpy.log if convex_func is None else convex_func
        self.parameter["eps"] = eps
        self.parameter["q"] = numpy.ones([self.user_bin, self.content_bin]) - 1e-3
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
        eps = self.parameter["eps"]
        p = self.total_reward[user_hash] / self.trials_arm[user_hash]
        f = self.parameter["f(t)"]
        self.parameter["p"][user_hash] = p
        c = f(self.trials[user_hash]) / self.trials_arm[user_hash]
        # newton root finding based on KL divergence
        rhs = p * numpy.log(p) + (1 - p) * numpy.log(1 - p) - c
        err = numpy.ones([self.content_bin])
        flag = numpy.ones([self.content_bin], dtype=bool)
        x = self.parameter['q'][user_hash]
        while numpy.any(flag):
            p1 = p[flag]
            x1 = x[flag]
            flag1 = flag[flag]
            rhs1 = rhs[flag]
            fx = p1 * numpy.log(x1) + (1 - p1) * numpy.log(1 - x1) - rhs1
            fprime = p1 / x1 - (1 - p1) / (1 - x1)
            dx = fx / fprime
            flag1[numpy.abs(dx) < eps] = False
            x1 = numpy.maximum(numpy.minimum(x1 - dx, 1 - 1e-8), p1 + 1e-8)
            x[flag] = x1
            flag[flag]  = flag1

        self.parameter['q'][user_hash] = x
        return numpy.argsort(self.parameter['q'][user_hash])
