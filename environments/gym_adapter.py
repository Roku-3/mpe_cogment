# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import cogment
import gym
import numpy as np

from cogment_verse.specs import (
    encode_rendered_frame,
    deserialize_ndarray,
    serialize_ndarray,
    EnvironmentSpecs,
    Observation,
    Space,
    SpaceValue,
)
from cogment_verse.constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS

# configure pygame to use a dummy video server to be able to render headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

SPACES_BOUND_MAX = float.fromhex("0x1.fffffep+127")
SPACES_BOUND_MIN = -SPACES_BOUND_MAX


def space_from_gym_space(gym_space):
    if isinstance(gym_space, gym.spaces.Box):
        return Space(
            properties=[
                Space.Property(
                    box=Space.Box(
                        shape=list(gym_space.shape),
                        low=[
                            Space.Bound(bound=v) if SPACES_BOUND_MIN < v < SPACES_BOUND_MAX else Space.Bound()
                            for v in gym_space.low.flat
                        ],
                        high=[
                            Space.Bound(bound=v) if SPACES_BOUND_MIN < v < SPACES_BOUND_MAX else Space.Bound()
                            for v in gym_space.high.flat
                        ],
                    )
                )
            ]
        )
    if isinstance(gym_space, gym.spaces.Discrete):
        return Space(properties=[Space.Property(discrete=Space.Discrete(num=gym_space.n))])
    raise RuntimeError(f"[{type(gym_space)}] is not a supported gym space type")


def gym_action_from_action(space, action):
    if len(action.properties) == 1:
        prop_value = action.properties[0]
        value_type = prop_value.WhichOneof("value")
        if value_type == "discrete":
            return prop_value.discrete
        if value_type == "box":
            return deserialize_ndarray(prop_value.box)
        # value_type == "simple_box"
        return np.array(prop_value.simple_box.values).reshape(space.properties[0].box.shape)
    raise RuntimeError(f"Not supporting spaces not having one property, got {len(action.properties)}")


def observation_from_gym_observation(gym_space, gym_observation):
    if isinstance(gym_space, gym.spaces.Box):
        return SpaceValue(properties=[SpaceValue.PropertyValue(box=serialize_ndarray(gym_observation))])
    if isinstance(gym_space, gym.spaces.Discrete):
        return SpaceValue(properties=[SpaceValue.PropertyValue(discrete=gym_observation.item())])
    raise RuntimeError(f"[{type(gym_space)}] is not a supported gym space type")


class Environment:
    def __init__(self, cfg):
        self.gym_env_name = cfg.env_name

        gym_env = gym.make(self.gym_env_name)
        self.env_specs = EnvironmentSpecs(
            num_players=1,
            observation_space=space_from_gym_space(gym_env.observation_space),
            action_space=space_from_gym_space(gym_env.action_space),
        )

    def get_implementation_name(self):
        return self.gym_env_name

    def get_environment_specs(self):
        return self.env_specs

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]
        assert len(player_actors) == 1
        [(player_actor_idx, player_actor_name)] = player_actors

        teacher_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == TEACHER_ACTOR_CLASS
        ]
        assert len(teacher_actors) <= 1
        has_teacher = len(teacher_actors) == 1
        if has_teacher:
            [(teacher_actor_idx, _teacher_actor_name)] = teacher_actors

        session_cfg = environment_session.config

        gym_env = gym.make(self.gym_env_name)

        gym_observation, _info = gym_env.reset(seed=session_cfg.seed, return_info=True)
        observation_value = observation_from_gym_observation(gym_env.observation_space, gym_observation)

        rendered_frame = None
        if session_cfg.render:
            rendered_frame = encode_rendered_frame(gym_env.render(mode="rgb_array"), session_cfg.render_width)

        environment_session.start([("*", Observation(value=observation_value, rendered_frame=rendered_frame))])

        async for event in environment_session.all_events():
            if event.actions:
                player_action_value = event.actions[player_actor_idx].action.value
                action_value = player_action_value
                overridden_players = []
                if has_teacher and event.actions[teacher_actor_idx].action.HasField("value"):
                    teacher_action_value = event.actions[teacher_actor_idx].action.value
                    action_value = teacher_action_value
                    overridden_players = [player_actor_name]

                gym_action = gym_action_from_action(
                    self.env_specs.action_space, action_value
                )  # pylint: disable=no-member

                gym_observation, reward, done, _info = gym_env.step(gym_action)
                observation_value = observation_from_gym_observation(gym_env.observation_space, gym_observation)

                rendered_frame = None
                if session_cfg.render:
                    if session_cfg.render:
                        rendered_frame = encode_rendered_frame(
                            gym_env.render(mode="rgb_array"), session_cfg.render_width
                        )

                observations = [
                    (
                        "*",
                        Observation(
                            value=observation_value,
                            rendered_frame=rendered_frame,
                            overridden_players=overridden_players,
                        ),
                    )
                ]

                if reward is not None:
                    environment_session.add_reward(
                        value=reward,
                        confidence=1.0,
                        to=[player_actor_name],
                    )

                if done:
                    # The trial ended
                    environment_session.end(observations)
                elif event.type != cogment.EventType.ACTIVE:
                    # The trial termination has been requested
                    environment_session.end(observations)
                else:
                    # The trial is active
                    environment_session.produce_observations(observations)

        gym_env.close()