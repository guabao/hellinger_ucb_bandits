__doc__ = "Latency test for several ucb algos"

import time

import numpy
import pandas

import orig_ucb
import kl_ucb
import h_ucb


KLUCB_MAX_ITER = 100
KLUCB_EPS = 1e-4
PRODUCTION_TTL = 0.02 # 20 milliseconds for timeout

def prepare_data(
        simulation_periods=10000,
        num_arms=-1,
    ) -> pandas.DataFrame:
    """
    Prepare data for testing
    """
    numpy.random.seed(42)
    # generate uniform random variable in [0, 1] then quantize to 0 or 1 based on different thresholds
    thresholds = numpy.random.gamma(1, scale=1, size=num_arms)
    # scale max threshold to 0.5
    thresholds = thresholds / numpy.max(thresholds) * 0.5
    data = numpy.random.rand(simulation_periods, num_arms)
    data = numpy.where(data < thresholds, 0, 1)
    columns = ["Ad " + str(i) for i in range(1, num_arms+1)]
    idx = numpy.arange(simulation_periods)
    return pandas.DataFrame(data, columns=columns, index=idx)

def test_latency(
        simulation_periods=10000,
        num_arms=-1,
    ):
    """
    Test the latency of several ucb algos
    """

    # load data
    g = prepare_data(simulation_periods, num_arms)
    user_hash = {"user": 0}
    contents = ["Ad " + str(i) for i in range(1, num_arms+1)]
    content_hash = {v: k for k, v in enumerate(contents)}
    time_table = numpy.zeros([len(g), 3])
    learner1 = orig_ucb.core.UCBLearner(user_feature_hash=1,
                         content_hash=num_arms)
    t1 = 0.
    learner2 = kl_ucb.core.KLUCB(user_feature_hash=1,
                         content_hash=num_arms,
                         max_iter=KLUCB_MAX_ITER,
                         eps=KLUCB_EPS)
    t2 = 0.
    learner3 = h_ucb.core.HUCB(user_feature_hash=1,
                         content_hash=num_arms)
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
        time_table[i, 0] = time.time() - t
        t1 += time_table[i, 0]
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
        arms = learner2.trial(user_hash['user'], print_log=False)["res"]
        time_table[i, 1] = time.time() - t
        t2 += time_table[i, 1]
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
        time_table[i, 2] = time.time() - t
        t3 += time_table[i, 2]
        # select top K
        action = numpy.zeros(len(contents))
        for armi in arms[:topK]:
            action[armi] = 1
        reward = env * action
        rewards3[i] = numpy.sum(reward)
        # update ucb
        for armi in arms[:topK]:
            learner3.update_reward_single_trial(user_hash['user'], armi, reward[armi])


    # timeout rate
    timeout_rate = numpy.sum(time_table > PRODUCTION_TTL, axis=0) / len(time_table)
    # average latency
    avg_latency = numpy.mean(time_table, axis=0)
    print("Single trial reward:")
    print(g.sum())
    print("======================================")
    print(f"Orig-UCB Learner reward: {rewards1.sum()}, total_running_time: {t1}")
    print(f"Average reward per round: {numpy.mean(rewards1)}")
    print(f"Timeout rate: {timeout_rate[0]}")
    print(f"Average latency: {avg_latency[0]}")
    print("======================================")
    print(f"KL-UCB Learner reward: {rewards2.sum()}, total_running_time: {t2}")
    print(f"Average reward per round: {numpy.mean(rewards2)}")
    print(f"Timeout rate: {timeout_rate[1]}")
    print(f"Average latency: {avg_latency[1]}")
    print("======================================")
    print(f"H-UCB Learner reward: {rewards3.sum()}, total_running_time: {t3}")
    print(f"Average reward per round: {numpy.mean(rewards3)}")
    print(f"Timeout rate: {timeout_rate[2]}")
    print(f"Average latency: {avg_latency[2]}")
    print("======================================")
    return {
        "simulation_periods": simulation_periods,
        "num_arms": num_arms,
        "reward_UCB": rewards1.sum(),
        "reward_KLUCB": rewards2.sum(),
        "reward_HUCB": rewards3.sum(),
        "total_running_time_UCB": t1,
        "total_running_time_KLUCB": t2,
        "total_running_time_HUCB": t3,
        "timeout_rate_UCB": timeout_rate[0],
        "timeout_rate_KLUCB": timeout_rate[1],
        "timeout_rate_HUCB": timeout_rate[2],
        "avg_latency_UCB": avg_latency[0],
        "avg_latency_KLUCB": avg_latency[1],
        "avg_latency_HUCB": avg_latency[2],
    }


def print_all_results(all_results):
    """
    Print all results
    """
    g = pandas.DataFrame(all_results)
    print(g)


if __name__ == "__main__":
    all_results = []
    res = test_latency(1000, 100)
    all_results.append(res)
    res = test_latency(2000, 100)
    all_results.append(res)
    res = test_latency(5000, 100)
    all_results.append(res)
    res = test_latency(1000, 1000)
    all_results.append(res)
    res = test_latency(2000, 1000)
    all_results.append(res)
    res = test_latency(5000, 1000)
    all_results.append(res)
    res = test_latency(1000, 10000)
    all_results.append(res)
    res = test_latency(2000, 10000)
    all_results.append(res)
    res = test_latency(5000, 10000)
    all_results.append(res)
    print_all_results(all_results)
    
