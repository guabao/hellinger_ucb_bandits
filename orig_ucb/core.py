# A demo of Upper Confidence Bound method
# Author: Jiazhou Wang

import numpy
import pandas

# Importing the dataset
dataset = pandas.read_csv('Ads_CTR_Optimisation.csv')

# UCB learner
class UCBLearner(object):
    def __init__(self, user_feature_hash, content_hash):
        self.user_bin = user_feature_hash
        self.content_bin = content_hash
        self.total_reward = numpy.zeros([self.user_bin, self.content_bin])
        self.trials_arm = 1e-8 * numpy.ones([self.user_bin, self.content_bin])
        self.trials = numpy.ones(self.user_bin)
        return None

    def update_reward_single_trial(self, user_hash, content_hash, reward):
        self.total_reward[user_hash, content_hash] += reward
        self.trials[user_hash] += 1
        self.trials_arm[user_hash, content_hash] += 1
        return None

    def get_total_reward(self):
        return numpy.sum(self.total_reward)

    def update_bound_batch(self, batch_result):
        raise NotImplementedError("Not implemented yet!")

    def trial(self, user_hash):
        base_reward = self.total_reward[user_hash] / self.trials_arm[user_hash]
        bonus = numpy.sqrt(2 * numpy.log(self.trials[user_hash]) / self.trials_arm[user_hash])
        return numpy.argsort(base_reward + bonus)


def test_ucb():
    g = pandas.read_csv('Ads_CTR_Optimisation.csv')
    user_hash = {'user': 0}
    contents = ["Ad " + str(i) for i in range(1, 11)]
    content_hash = {v: k for k, v in enumerate(contents)}
    learner = UCBLearner(user_feature_hash=1,
                         content_hash=10)
    topK = 3
    # run simulation
    for i in range(len(g)):
        print("%ith round"%i)
        # get priority list
        arms = learner.trial(user_hash['user'])
        # select top K
        action = numpy.zeros(len(contents))
        for armi in arms[:topK]:
            action[armi] = 1
        # get reward
        env = g.iloc[i].values.ravel()
        reward = env * action
        # update ucb
        for armi in arms[:topK]:
            learner.update_reward_single_trial(user_hash['user'], armi, reward[armi])
    print("Single trial reward:")
    print(g.sum())
    print("======================================")
    print("Learner reward:")
    print(learner.get_total_reward())
    return None


if __name__ == "__main__":
    test_ucb()
