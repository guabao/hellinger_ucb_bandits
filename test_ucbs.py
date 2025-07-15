__doc__ = "Test several ucb algos"

import time

import numpy
import pandas

import orig_ucb
import kl_ucb
import h_ucb

def test_ucb():
    g = pandas.read_csv('Ads_CTR_Optimisation.csv')
    user_hash = {'user': 0}
    contents = ["Ad " + str(i) for i in range(1, 11)]
    content_hash = {v: k for k, v in enumerate(contents)}
    learner1 = orig_ucb.core.UCBLearner(user_feature_hash=1,
                         content_hash=10)
    t1 = 0.
    learner2 = kl_ucb.core.KLUCB(user_feature_hash=1,
                         content_hash=10)
    t2 = 0.
    learner3 = h_ucb.core.HUCB(user_feature_hash=1,
                         content_hash=10)
    t3 = 0.
    topK = 3
    rewards1 = numpy.zeros(len(g))
    rewards2 = numpy.zeros(len(g))
    rewards3 = numpy.zeros(len(g))
    # run simulation
    for i in range(len(g)):
        if i%1000 == 0:
            print("%ith round"%i)
        env = g.iloc[i].values.ravel()

        # -------------------- Original UCB algo -------------------------
        # get priority list
        t = time.time()
        arms = learner1.trial(user_hash['user'])
        t1 += time.time() - t
        # select top K
        action = numpy.zeros(len(contents))
        for armi in arms[:topK]:
            action[armi] = 1
        # get reward
        reward = env * action
        rewards1[i] = numpy.sum(reward)
        # update ucb
        for armi in arms[:topK]:
            learner1.update_reward_single_trial(user_hash['user'], armi, reward[armi])


        # -------------------- KL UCB algo -------------------------
        # get priority list
        t = time.time()
        arms = learner2.trial(user_hash['user'])["res"]
        t2 += time.time() - t
        # select top K
        action = numpy.zeros(len(contents))
        for armi in arms[:topK]:
            action[armi] = 1
        reward = env * action
        rewards2[i] = numpy.sum(reward)
        # update ucb
        for armi in arms[:topK]:
            learner2.update_reward_single_trial(user_hash['user'], armi, reward[armi])


        # -------------------- H UCB algo -------------------------
        # get priority list
        t = time.time()
        arms = learner3.trial(user_hash['user'])
        t3 += time.time() - t
        # select top K
        action = numpy.zeros(len(contents))
        for armi in arms[:topK]:
            action[armi] = 1
        reward = env * action
        rewards3[i] = numpy.sum(reward)
        # update ucb
        for armi in arms[:topK]:
            learner3.update_reward_single_trial(user_hash['user'], armi, reward[armi])


    print("Single trial reward:")
    print(g.sum())
    print("======================================")
    print("Orig-UCB Learner reward: %.1f, total_running_time: %.6f"%(learner1.get_total_reward(), t1))
    print("Average reward per round: %.6f"%(numpy.mean(rewards1)))
    print("======================================")
    print("KL-UCB Learner reward: %.1f, total_running_time: %.6f"%(learner2.get_total_reward(), t2))
    print("Average reward per round: %.6f"%(numpy.mean(rewards2)))
    print("======================================")
    print("H-UCB Learner reward: %.1f, total_running_time: %.6f"%(learner3.get_total_reward(), t3))
    print("Average reward per round: %.6f"%(numpy.mean(rewards3)))
    return None


