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