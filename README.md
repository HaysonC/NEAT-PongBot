# A Study of Deep RL and NeuroEvolution in Atari Games

Note: DDQN and VPG are kinda bugging right now but hopefully they will get fixed sometime soon

## Usage of RL

The runner script `run_agent.py` is also kinda retarded, but it will be fixed.

```
python run_agent.py --help
python run_agent.py --mode train --agent 3 --episodes 100
```

## NEAT

The NEAT implementation is built on top of the neat-python package. A reference example can be found [here](https://github.com/NirajSawant136/Simple-AI-using-NEAT/tree/master).

The configuration is written in `config-fc.txt` and the training is done using a modified version of the main game engine (`PongAIvsAi.py`) in `sim.py`. The network is stored in `best_genome.pkl` and you can load it with the built in functions.

For customized training (which is recommended), modify the config file (e.g. fitness threshold).

> You should modify `neat_train.py` if you are using custom game env

## Dependencies

Gym, Torch, Numpy, ALE-py (this is used for pong atari game), Neat-Python

```
pip install -r requirements.txt
```
