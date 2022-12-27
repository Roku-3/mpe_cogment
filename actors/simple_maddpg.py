import logging
import copy
import time
import json
import math
import numpy as np

import cogment
import torch

from cogment_verse.specs import (
    AgentConfig,
    cog_settings,
    EnvironmentConfig,
    flatten,
    flattened_dimensions,
    PLAYER_ACTOR_CLASS,
    PlayerAction,
    SpaceValue,
    sample_space,
    WEB_ACTOR_NAME,
    HUMAN_ACTOR_IMPL,
)

from .Agent import Agent
from .Buffer import Buffer
from cogment_verse import Model # pylint: disable=abstract-class-instantiated
from pettingzoo.mpe import simple_tag_v2
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)

def get_env():
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    new_env = simple_tag_v2.parallel_env()

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info


class SimpleMADDPGModel(Model):
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, model_id, dim_info, capacity, batch_size, actor_lr, critic_lr, version_number=0):
        super().__init__(model_id=model_id, version_number=version_number)

        logging.warning(dim_info)
        # sum all the dims of each agent to get input dim for critic
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cpu')
        self.dim_info = dim_info
        self.batch_size = batch_size

        # 関数では使わないけど保存？
        self.capacity = capacity
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def add(self, obs, action, reward, next_obs, done):
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a]

            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            # buffer.pyのsample
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)

        return obs, act, reward, next_obs, done, next_act

    def select_action(self, obs):
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float()
            a = self.agents[agent].action(o)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax().item()
            self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma):
        for agent_id, agent in self.agents.items():
            # maddpgのsample, 全てのagentの値のリスト
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            # 終了していたらtarget_valueは0
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], model_out=True)
            act[agent_id] = action
            actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def get_model_user_data(self):
        return {
            "dim_info": json.dumps(self.dim_info),
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
        }

    # def save(self, reward):
    def save(self, model_data_f):
        """save actor parameters of all agents and training reward"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            model_data_f
        )
        return {"num_samples_seen": 1245}


    @classmethod
    def load(cls, model_id, version_number, model_user_data, version_user_data, model_data_f):
        """init maddpg using the model saved in `file`"""
        # 自身のクラスで初期化している
        model = SimpleMADDPGModel(
            model_id=model_id, 
            dim_info=json.loads(model_user_data["dim_info"]), 
            capacity=int(model_user_data["capacity"]), 
            batch_size=int(model_user_data["batch_size"]), 
            actor_lr=float(model_user_data["actor_lr"]), 
            critic_lr=float(model_user_data["critic_lr"]), 
            version_number=version_number,
        )
        data = torch.load(model_data_f)
        for agent_id, agent in model.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return model 


# 人間を含まないMADDPGモデルを保存するためのもの
class SimpleMADDPGActor:
    # tensorを何の型で扱うか初期化
    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config
        observation_space = config.environment_specs.observation_space
        action_space = config.environment_specs.action_space

        rng = np.random.default_rng(config.seed if config.seed is not None else 0)

        # 保存されているモデルの取り出し
        model, _, _ = await actor_session.model_registry.retrieve_version(
            SimpleMADDPGModel, config.model_id, config.model_version
        )

        action_value = None

        # 各プレイヤーのactionを反映
        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                # actor_session.do_action(PlayerAction(value=action_value))
                actor_session.do_action(None)

class SimpleMADDPGTraining:
    # default_cfg = {
    #     "seed": 10,
    #     "num_epochs": 50,
    #     "epoch_num_training_trials": 100,
    #     "num_parallel_trials": 7,
    #     "episode_num": 30000,   # total episode num during training procedure
    #     "episode_length": 25,   # steps per episode
    #     "learn_interval": 100,  # steps interval between learning time
    #     "random_steps": 5e4,    # random steps before the agent start to learn
    #     "tau": 0.02,            # soft update parameter
    #     "gamma": 0.95,          # discount factor
    #     "buffer_size": 1e6,     # capacity of replay buffer
    #     "batch_size": 1024,     # batch-size of replay buffer
    #     "actor_lr": 0.01,       # learning rate of actor
    #     "critic_lr": 0.01,      # learning rate of critic
    # }

    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg
        self._rng = np.random.default_rng(self._cfg.seed)

    async def sample_producer_impl(self, sample_producer_session):
        async for sample in sample_producer_session.all_trial_samples():
            pass
            # actor_sample = sample.actors_data[player_actor_name]
            # if actor_sample.observation is None:
                # This can happen when there is several "end-of-trial" samples
                # continue

            # if observation is not None:
            #     # It's not the first sample, let's check if it is the last
            #     done = sample.trial_state == cogment.TrialState.ENDED
            #     sample_producer_session.produce_sample(
            #         (
            #             observation,
            #             next_observation,
            #             action,
            #             reward,
            #             torch.ones(1, dtype=torch.int8) if done else torch.zeros(1, dtype=torch.int8),
            #             total_reward,
            #         )
            #     )
            done = sample.trial_state == cogment.TrialState.ENDED
            sample_producer_session.produce_sample(
                (
                    torch.ones(1, dtype=torch.int8) if done else torch.zeros(1, dtype=torch.int8),
                )
            )

    async def impl(self, run_session):
        # Initializing a model
        model_id = f"{run_session.run_id}_model"

        # assert len(self._environment_specs.action_space.properties) == 1
        # assert self._environment_specs.action_space.properties[0].WhichOneof("type") == "discrete"

        # model = SimpleMADDPGModel(
        #     model_id,
        #     environment_implementation=self._environment_specs.implementation,
        #     num_input=flattened_dimensions(self._environment_specs.observation_space),
        #     num_output=flattened_dimensions(self._environment_specs.action_space),
        #     num_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
        #     dtype=self._dtype,
        # )
        _env, dim_info = get_env()

        model = SimpleMADDPGModel(
            model_id,
            dim_info, 
            self._cfg.buffer_size, 
            self._cfg.batch_size, 
            self._cfg.actor_lr, 
            self._cfg.critic_lr, 
            0, # version_number
        )

        _model_info, version_info = await run_session.model_registry.publish_initial_version(model)

        def create_human_params():
            # 人間だったらhuman_actor_classを返す
            return cogment.ActorParameters(
                cog_settings,
                name=WEB_ACTOR_NAME,
                class_name=PLAYER_ACTOR_CLASS,
                implementation=HUMAN_ACTOR_IMPL,
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs,
                ),
            )
            
        def create_maddpg_params(name, version_number=-1):
            # AI側だったら全てのアクターを含むMADDPGを返す
            return cogment.ActorParameters(
                cog_settings,
                name=name,
                class_name=PLAYER_ACTOR_CLASS,
                implementation="actors.simple_maddpg.SimpleMADDPGActor",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    seed=self._rng.integers(9999),
                    model_id=model_id,
                    model_version=version_number,
                ),
            )

        def create_trials_params(p1_params, p2_params):
            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id,
                    render=HUMAN_ACTOR_IMPL in (p1_params.implementation, p2_params.implementation),
                    seed=self._rng.integers(9999),
                ),
                actors=[p1_params, p2_params],
            )

        previous_epoch_version_number = None
        for epoch_idx in range(self._cfg.num_epochs):

            # Self training trials
            for (step_idx, _trial_id, trial_idx, sample,) in run_session.start_and_await_trials(
                trials_id_and_params=[
                    (
                        f"{run_session.run_id}_{epoch_idx}_t_{trial_idx}",
                        create_trials_params(
                            # cogmentに人間/ MADDPG agentsをそれぞれアクターとして渡す
                            p1_params=create_human_params(),
                            p2_params=create_maddpg_params("player_maddpg"),
                        ),
                    )
                    for trial_idx in range(self._cfg.epoch_num_training_trials)
                ],
                sample_producer_impl=self.sample_producer_impl,
                num_parallel_trials=self._cfg.num_parallel_trials,
            ):
                # (_actor_name, observation, next_observation, action, reward, done, total_rewards) = sample
                done = sample