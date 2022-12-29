from enum import Enum
import logging
import cogment
from pettingzoo.mpe import simple_tag_v2

from cogment_verse.specs import (
    encode_rendered_frame,
    serialize_ndarray,
    deserialize_ndarray,
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

        # assert len(player_actors) == self.env_specs.num_players  # pylint: disable=no-member

        session_cfg = environment_session.config
        pz_env = self.env_class.env()
        pz_env.reset(seed=session_cfg.seed)
        pz_observation, _pz_reward, _pz_done, _pz_truncate, _pz_info = pz_env.last()
        # pz_observation, _pz_reward, _pz_done, _pz_info = pz_env.last()

        # log.warning(f"pz_observation: {pz_observation}")
        # log.warning(f"observation_space: {pz_env.observation_space(current_player_pz_agent)}")

        rendered_frame = None
        if session_cfg.render:
            rendered_frame = encode_rendered_frame(pz_env.render(mode='rgb_array'), session_cfg.render_width)


        environment_session.start(
            [
                (
                    "*",
                    Observation(
                        current_player = pz_env.agent_selection,
                        # observation=pz_observation,
                        observation=serialize_ndarray(pz_observation),
                        reward=_pz_reward,
                        done=_pz_done,
                        rendered_frame=rendered_frame,
                    ),
                )
            ]
        )

        async for event in environment_session.all_events():
            if event.actions:
                # actions[0]が人間

                # print(f"pz_env.agents: {pz_env.agents}")
                # print(f"current_agent: {current_player_pz_agent}")
                # print(f"{player_action_value.properties}-----{type(player_action_value.properties)}")
                # log.warning(event.actions)
                # log.warning(f"actions[0]:[0]: {event.actions[0].action.value.properties[0]}")
                # log.warning(f"actions[1]:[0]: {event.actions[1].action.value.properties[0]}")


                if pz_env.agent_selection == "adversary_0":
                    # gym_action = random.randint(0,4)
                    action_value = event.actions[0].action.value
                else:
                    action_value = event.actions[1].action.value

                gym_action = gym_action_from_action(
                    self.env_specs.action_space, action_value  # pylint: disable=no-member
                )

                if pz_env.agents:
                    pz_env.step(gym_action)

                pz_observation, _pz_reward, _pz_done, _pz_truncate, _pz_info = pz_env.last()

                # log.warning(f"pz_observation: {pz_observation}")
                # log.warning(f"_pz_reward: {_pz_reward}")
                # log.warning(f"_pz_done: {_pz_done}")
                # log.warning(f"_pz_truncate: {_pz_truncate}")
                # log.warning(f"_pz_info: {_pz_info}")

                rendered_frame = None
                if session_cfg.render:
                    rendered_frame = encode_rendered_frame(pz_env.render(mode='rgb_array'), session_cfg.render_width)

                observations = [
                    (
                        "*",
                        Observation(
                            current_player = pz_env.agent_selection,
                            # observation=pz_observation,
                            observation=serialize_ndarray(pz_observation),
                            reward=_pz_reward,
                            done=_pz_done,
                            rendered_frame=rendered_frame,
                        ),
                    )
                ]

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
