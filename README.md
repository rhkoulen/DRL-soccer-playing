# CS 8803 DRL Soccer Twos
## Team 26: RoboCup Farm Team

Some agent's to play in CEIA's Unity sandbox, forked from https://github.com/bryanoliveira/soccer-twos-starter.

Our final submission `TEAM26_ROBOCUP_FT_AGENT.zip` got tied 5th in our class's single elim bracket :)

## Available Agents

There are two agents that serve as baselines that we did not write:

- CEIA_AGENT: ostensibly developed by the developers of the soccer-twos package, Brazil's Center of Excellence in Artificial Intelligence
- RANDOM_AGENT: moves randomly

We wrote three agents:

- NATE_PPO_AGENT: trained on PPO (10M steps of self-play)
- SHOURIK_SHAPED_AGENT: trained on PPO + a shaped reward function (15M steps of self-play)
- RICHARD_LSTM_AGENT: an LSTM policy net trained on PPO + Shourik's shaped reward (10M steps of self-play then 5M steps of league-play)

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

### 1. Fork this repository
git clone https://github.com/rhkoulen/DRL-soccer-playing.git

cd DRL-soccer-playing/

### 2. Create and activate conda environment (venv also works)
`conda create --name soccertwos python=3.8 -y`

`conda activate soccertwos`

`pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4`

`pip install -r requirements.txt`

`pip install protobuf==3.20.3 pydantic==1.10.13`


## Testing

All of the above agents are compatible with the soccer-twos package.

To play a game (2m, 10 diff mercy rule):

`python -m soccer_twos.watch -m1 FIRST_AGENT -m2 SECOND_AGENT`

To play out a batch of episodes (where each episode is kickoff to goal, no match rules):

`python -m soccer_twos.evaluate -m1 FIRST_AGENT -m2 SECOND_AGENT -e 1000`
Of course, change 1000 to however many you would like to rollout.

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Examples are located under the `example_player_agent/` or `example_team_agent/` directories.
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent desription and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.).

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.
