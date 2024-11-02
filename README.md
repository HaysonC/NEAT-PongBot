# Deep RL and NeuroEvolution in Atari Games

## Reinforcement Learning -- DQN & PG

Please only use DDQN for training or inference. I hypothesize Q learning will outperform Policy Gradient, and I will train DDPG in the future to prove this point. For now, only DQN trains and works properly.

```
python3 pong_dqn.py
```

For inference, `import pong_dqn`, create `DQNAgent()`, `agent.load_weights()`, and `agent.inference()`. This has already been implemented in `PongAIvsAi.py`.

> NOTE: The two VPG agents are not working well, I will fix soon

The runner script `run_agent.py` is also kinda retarded, but it will be fixed. **I will fix the trigger script someday soon**.

```
python run_agent.py --help
python run_agent.py --mode train --agent 3 --episodes 100
```

## NEAT: Neuroevolution by Augmented Topology

The NEAT implementation is built on top of the neat-python package. A reference example can be found [here](https://github.com/NirajSawant136/Simple-AI-using-NEAT/tree/master).

The configuration is written in `config-fc.txt` and the training is done using a modified version of the main game engine (`PongAIvsAi.py`) in `sim.py`. The network is stored in `best_genome.pkl` and you can load it with the built in functions.

For customized training (which is recommended), modify the config file (e.g. fitness threshold).

> `neat_train.py` has already been modified to train with new environment. Tuning is ongoing...

Again, making inference is similar to discussed in [the previous section](#usage-of-rl)

## Dependencies

Gym, Torch, Numpy, ALE-py (this is used for pong atari game), Neat-Python

```
pip install -r requirements.txt
```
