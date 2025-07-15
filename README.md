# hellinger_ucb_bandits

Three bandit algorithms: 
1. The Upper Confidence Bound (UCB) Bandit Algorithm
2. KL-UCB Algorithm
3. (joint work with Ruibo Yang) Hellinger-UCB Algorithm


# useage

## simple accuracy test
For simple accuracy test, run
```
python test_ucbs.py
```

Sample print log
```
Single trial reward:
Ad 1     1703
Ad 2     1295
Ad 3      728
Ad 4     1196
Ad 5     2695
Ad 6      126
Ad 7     1112
Ad 8     2091
Ad 9      952
Ad 10     489
dtype: int64
======================================
Orig-UCB Learner reward: 5925.0, total_running_time: 0.094947
Average reward per round: 0.592500
======================================
KL-UCB Learner reward: 6331.0, total_running_time: 0.777006
Average reward per round: 0.633100
======================================
H-UCB Learner reward: 6443.0, total_running_time: 0.414262
Average reward per round: 0.644300
```

## latency test
For latency test, run
```
python latency_test.py
```

```
   simulation_periods  num_arms  reward_UCB  reward_KLUCB  reward_HUCB  total_running_time_UCB  total_running_time_KLUCB  total_running_time_HUCB  timeout_rate_UCB  timeout_rate_KLUCB  timeout_rate_HUCB  avg_latency_UCB  avg_latency_KLUCB  avg_latency_HUCB
0                1000       100      2760.0        2862.0       2864.0                0.014518                  2.106075                 0.052174               0.0              0.0010                0.0         0.000015           0.002106          0.000052
1                2000       100      5553.0        5746.0       5769.0                0.028893                  4.211625                 0.102976               0.0              0.0000                0.0         0.000014           0.002106          0.000051
2                5000       100     13993.0       14414.0      14501.0                0.069654                 10.513188                 0.250987               0.0              0.0002                0.0         0.000014           0.002103          0.000050
3                1000      1000      2815.0        2853.0       2848.0                0.027757                  2.889836                 0.063442               0.0              0.0000                0.0         0.000028           0.002890          0.000063
4                2000      1000      5657.0        5686.0       5799.0                0.063099                  7.434541                 0.163255               0.0              0.0015                0.0         0.000032           0.003717          0.000082
5                5000      1000     14151.0       14273.0      14666.0                0.174882                 20.540135                 0.416225               0.0              0.0008                0.0         0.000035           0.004108          0.000083
6                1000     10000      2814.0        2795.0       2795.0                0.174475                  0.157389                 0.154574               0.0              0.0000                0.0         0.000174           0.000157          0.000155
7                2000     10000      5642.0        5615.0       5615.0                0.353018                  0.312580                 0.304327               0.0              0.0000                0.0         0.000177           0.000156          0.000152
8                5000     10000     14132.0       13671.0      14130.0                1.000729                 40.522028                 1.197865               0.0              0.3332                0.0         0.000200           0.008104          0.000240
```