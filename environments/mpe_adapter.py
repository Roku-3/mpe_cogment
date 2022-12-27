from enum import Enum
import logging

import cogment
from pettingzoo.mpe import simple_tag_v2

from cogment_verse.specs import (
    encode_rendered_frame,
    EnvironmentSpecs,
    Observation,
    space_from_gym_space,
    gym_action_from_action,
    observation_from_gym_observation,
)
from cogment_verse.constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS
from cogment_verse.utils import import_class
import random

log = logging.getLogger(__name__)


class Environment:
    def __init__(self, cfg):
        self.env_class_name = cfg.env_class_name
        self.env_class = import_class(self.env_class_name)
        # pz_env = self.env_class.env(num_good=2, num_adversaries=0, num_obstacles=0, continuous_actions=False)
        pz_env = self.env_class.env()

        observation_space = None
        action_space = None
        num_players = 0

        for player in pz_env.possible_agents:
            num_players += 1
            if observation_space is None:
                observation_space = pz_env.observation_space(player)
                action_space = pz_env.action_space(player)

        assert num_players >= 1

        self.env_specs = EnvironmentSpecs(
            num_players=num_players,
            observation_space=space_from_gym_space(observation_space),
            action_space=space_from_gym_space(action_space),
            turn_based=False,
        )
        # log.warning(self.env_specs.observation_space)

    def get_implementation_name(self):
        return self.env_class_name

    def get_environment_specs(self):
        return self.env_specs

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]
        # assert len(player_actors) == self.env_specs.num_players  # pylint: disable=no-member

        session_cfg = environment_session.config

        # pz_env = self.env_class.env(num_good=2, num_adversaries=0, num_obstacles=0, continuous_actions=False)
        pz_env = self.env_class.env()

        pz_env.reset(seed=session_cfg.seed)

        pz_agent_iterator = iter(pz_env.agent_iter())

        def next_player():
            nonlocal pz_agent_iterator
            current_player_pz_agent = next(pz_agent_iterator)
            current_player_actor_idx, current_player_actor_name = next(
                (player_actor_idx, player_actor_name)
                for (player_pz_agent, (player_actor_idx, player_actor_name)) in zip(pz_env.agents, player_actors)
                # if player_pz_agent == current_player_pz_agent
                if True
            )
            return (current_player_pz_agent, current_player_actor_idx, current_player_actor_name)

        current_player_pz_agent, current_player_actor_idx, current_player_actor_name = next_player()

        pz_observation, _pz_reward, _pz_done, _pz_truncate, _pz_info = pz_env.last()
        # pz_observation, _pz_reward, _pz_done, _pz_info = pz_env.last()

        # log.warning(f"pz_observation: {pz_observation}")
        # log.warning(f"observation_space: {pz_env.observation_space(current_player_pz_agent)}")

        observation_value = observation_from_gym_observation(
            pz_env.observation_space(current_player_pz_agent), pz_observation
        )

        rendered_frame = None
        if session_cfg.render:
            rendered_frame = encode_rendered_frame(pz_env.render(mode='rgb_array'), session_cfg.render_width)

        environment_session.start(
            [
                (
                    "*",
                    Observation(
                        value=observation_value,  # TODO Should only be sent to the current player
                        rendered_frame=rendered_frame,  # TODO Should only be sent to observers
                        current_player=current_player_actor_name,
                    ),
                )
            ]
        )

        async for event in environment_session.all_events():
            if event.actions:
                player_action_value = event.actions[current_player_actor_idx].action.value
                action_value = player_action_value
                if not current_player_pz_agent in pz_env.agents:
                    action_value.properties.discrete = None


                # print(f"pz_env.agents: {pz_env.agents}")
                # print(f"current_agent: {current_player_pz_agent}")
                # print(f"{player_action_value.properties}-----{type(player_action_value.properties)}")

                # 人間の入力で行動決定
                gym_action = gym_action_from_action(
                    self.env_specs.action_space, action_value  # pylint: disable=no-member
                )

                # 人間じゃなかったら動かさない
                if not pz_env.agent_selection == "adversary_0":
                    # gym_action = random.randint(0,4)
                    gym_action = 0

                if pz_env.agents:
                    pz_env.step(gym_action)

                current_player_pz_agent, current_player_actor_idx, current_player_actor_name = next_player()
                pz_observation, _pz_reward, _pz_done, _pz_truncate, _pz_info = pz_env.last()
                # pz_observation, _pz_reward, _pz_done, _pz_info = pz_env.last()

                observation_value = observation_from_gym_observation(
                    pz_env.observation_space(current_player_pz_agent), pz_observation
                )

                rendered_frame = None
                if session_cfg.render:
                    rendered_frame = encode_rendered_frame(pz_env.render(mode='rgb_array'), session_cfg.render_width)

                observations = [
                    (
                        "*",
                        Observation(
                            value=observation_value,
                            rendered_frame=rendered_frame,
                            current_player=current_player_actor_name,
                        ),
                    )
                ]

                for (rewarded_player_pz_agent, pz_reward) in pz_env.rewards.items():
                    if pz_reward == 0:
                        continue
                    rewarded_player_actor_name = next(
                        player_actor_name
                        for (player_pz_agent, (player_actor_idx, player_actor_name)) in zip(
                            pz_env.agents, player_actors
                        )
                        # if player_pz_agent == rewarded_player_pz_agent
                        if True
                    )
                    environment_session.add_reward(
                        value=pz_reward,
                        confidence=1.0,
                        to=[rewarded_player_actor_name],
                    )

                # done = all(pz_env.dones[pz_agent] for pz_agent in pz_env.agents)

                # if done:
                # if not pz_env.agents:
                if _pz_done or _pz_truncate:
                    # The trial ended
                    environment_session.end(observations)
                elif event.type != cogment.EventType.ACTIVE:
                    # The trial termination has been requested
                    environment_session.end(observations)
                else:
                    # The trial is active
                    environment_session.produce_observations(observations)

        pz_env.close()
