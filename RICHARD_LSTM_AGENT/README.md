# RoboCup Farm Team (Team 26)

## Authors
  - Shourik Banerjee (sbanerjee332@gatech.edu)
  - Richard Heinrich Koulen (rkoulen3@gatech.edu)
  - Nathaniel Brian Wert (nwert3@gatech.edu)

## About This Agent
Each of us came together to understand the system and lay out how we wanted to approach the problem, then split up to make our own agents. This agent was primarily coded and trained by Richard.

Given the initial success of Shourik's agent, I wanted to build a model that hopefully just improved model expressivity. Because this is a POMDP, LSTM should be a good policy model, as the hidden state may be able to remember a context and perhaps not lose track of the ball. Thus, this was trained on the same shaped reward function with 4 dense and 1 sparse reward terms (see SHOURIK_SHAPED_AGENT/README.md for more details).

The internal cell state is 512 nodes large. The observations are passed through an MLP with two hidden layers of size 256, then fed into the forget, input, and output gates of the LSTM. The output from the cell state to the decision space is a single layer perceptron. Frankly, I thought `fc_hiddens` was defining that output gate to the decision space, but the proper kwarg would have been `post_fc_hiddens`, which is probably why this acts quite poorly. I didn't have time to retrain before the competition.

The model was trained (with PPO) for 10M timesteps simply doing self-play. After this point, I trained it against a league of CEIA_AGENT, NATE_PPO_AGENT, and RANDOM_AGENT for 5M timesteps. This was an attempt to make it not generate equilibrious self-play artifacts, since the reward signal was dwindling (getting better at tieing itself). Though it was available, I did not include Shourik's agent in the league, since that was our best bot, and learning against it would potentially degrade generalization, meaning we couldn't use it as a benchmark to decide who is better.

## Checkpoints
The agent defaults to the 15M checkpoint, but I also included the 10M checkpoint. It is technically unfair to play CEIA_AGENT or NATE_PPO_AGENT against my 15M checkpoint because I learned against them. If you so desire to revert to the checkpoint before league-play, open `agent.py` and replace:

`CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint_001500', 'checkpoint-1500')`

with

`CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint_001000', 'checkpoint-1000')`
