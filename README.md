# Cogment Verse

[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-green?style=flat-square)](./LICENSE) [![Changelog](https://img.shields.io/badge/-Changelog%20-blueviolet?style=flat-square)](./CHANGELOG.md)

> 🚧 A new major version of Cogment Verse is under develelopment in the [`next`](https://github.com/cogment/cogment-verse/tree/next). Not all the algorithms and environments are available yet but it is fully operational. Do not hesitate to test it! 
>
> Follow and discuss the development in this [Pull Request](https://github.com/cogment/cogment-verse/pull/71). 

[Cogment](https://cogment.ai) is an innovative open source AI platform designed to leverage the advent of AI to benefit humankind through human-AI collaboration developed by [AI Redefined](https://ai-r.com). Cogment enables AI researchers and engineers to build, train and operate AI agents in simulated or real environments shared with humans. For the full user documentation visit <https://docs.cogment.ai>

This repository contains a library of environments and agent implementations
to get started with Human In the Loop Learning (HILL) and Reinforcement
Learning (RL) with Cogment in minutes. Cogment Verse is designed both
for practitioners discovering the field as well as for experienced
researchers or engineers as a framework to develop and benchmark new
approaches.

Cogment verse includes environments from:

- [OpenAI Gym](https://gym.openai.com),
- [Petting Zoo](https://www.pettingzoo.ml).
- [MinAtar](https://github.com/kenjyoung/MinAtar).
- [Procgen](https://github.com/openai/procgen).

## Documentation table of contents

- [Getting started](#getting-started)
- Tutorials 🚧
  - [Simple Behavior Cloning](/docs/tutorials/simple_bc.md)
- Experimental results 🚧
  - [A2C](/docs/results/a2c.md)
  - [REINFORCE](/docs/results/REINFORCE.md)
- Develop 🚧
  - [Development Setup](/docs/development_setup.md)
  - [Debug](#debug)
  - [Environment development](/docs/environment.md)
- [Changelog](/CHANGELOG.md)
- [Contributors guide](/CONTRIBUTING.md)
- [Community code of conduct](/CODE_OF_CONDUCT.md)
- [Acknowledgments](#acknowledgements)

## Getting started

1. Clone this repository
2. Install [Python 3.9](https://www.python.org/)
3. Create and activate a virtual environment by runnning
   ```console
   $ python -m venv .venv
   $ source .venv/bin/activate
   ```
4. Install the python dependencies by running
   ```console
   $ pip install -r requirements.txt
   ```
5. In another terminal, launch a mlflow server on port 3000 by running
   ```console
   $ source .venv/bin/activate
   $ python -m simple_mlflow
   ```
6. Start the default Cogment Verse run using `python -m main`
7. Open Chrome (other web browser might work but haven't tested) and navigate to http://localhost:8080/
8. Play the game!

That's the basic setup for Cogment Verse, you are now ready to train AI agents.

### Configuration

Cogment Verse relies on [hydra](https://hydra.cc) for configuration. This enables easy configuration and composition of configuration directly from yaml files and the command line.

The configuration files are located in the `config` directory, with defaults defined in `config/config.yaml`.

Here are a few examples:

- Launch a Simple Behavior Cloning run with the [Mountain Car Gym environment](https://www.gymlibrary.ml/environments/classic_control/mountain_car/) (which is the default environment)
  ```console
  $ python -m main services/actor=simple_bc run=simple_bc
  ```
- Launch a Simple Behavior Cloning run with the [Lunar Lander Gym environment](https://www.gymlibrary.ml/environments/box2d/lunar_lander/)
  ```console
  $ python -m main services/actor=simple_bc services/environment=lunar_lander run=simple_bc
  ```
- Launch and play a single trial of the Lunar Lander Gym environment with continuous controls
  ```console
  $ python -m main services/environment=lunar_lander_continuous
  ```
- Launch an A2C training run with the [Cartpole Gym environment](https://www.gymlibrary.ml/environments/classic_control/cartpole/)

  ```console
  $ python -m main +experiment=simple_a2c/cartpole
  ```

  This one is completely _headless_ (training doens't involve interaction with a human player). It will take a little while to run, you can monitor the progress using mlflow at <http://localhost:3000>

## List of publications and submissions using Cogment and/or Cogment Verse

- Analyzing and Overcoming Degradation in Warm-Start Off-Policy Reinforcement Learning [code](https://github.com/benwex93/cogment-verse)
- Multi-Teacher Curriculum Design for Sparse Reward Environments [code](https://github.com/kharyal/cogment-verse/)

(please open a pull request to add missing entries)

## Acknowledgements

The subdirectories `/tf_agents/cogment_verse_tf_agents/third_party` and `/torch_agents/cogment_verse_torch_agents/third_party` contains code from third party sources

- `hive`: Taken from the [Hive library](https://github.com/chandar-lab/RLHive)
- `td3`: Taken form the [authors' implementation](https://github.com/sfujim/TD3)
