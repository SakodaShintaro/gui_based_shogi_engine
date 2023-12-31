import torch
import torch.nn as nn
import numpy as np

from model import RSSM, RewardModel, ContinueModel
from encoder import Encoder
from decoder import Decoder
from actor import Actor
from critic import Critic

from utils import (
    compute_lambda_values,
    create_normal_dist,
    DynamicInfos,
)
from buffer import ReplayBuffer
from tqdm import tqdm


class Dreamer:
    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        writer,
        device,
        config,
    ):
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool

        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(self.device)
        self.actor = Actor(discrete_action_bool,
                           action_size, config).to(self.device)
        self.critic = Critic(config).to(self.device)

        self.buffer = ReplayBuffer(
            observation_shape, action_size, self.device, config)

        self.config = config.parameters.dreamer

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.writer = writer
        self.num_total_episode = 0

    def train(self, env):
        for iteration in range(self.config.train_iterations):
            print(f"iteration = {iteration}")
            self.environment_interaction(
                env, num_interaction_step=100, train=True)
            for _ in tqdm(range(self.config.collect_interval)):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                posteriors, deterministics = self.dynamic_learning(data)
                self.behavior_learning(posteriors, deterministics)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(
            len(data.action))

        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, data, posterior_info):
        step = self.model_optimizer.state[self.model_optimizer.param_groups[0]
                                          ["params"][-1]].get("step", 0)
        reconstructed_observation_dist = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )
        self.writer.add_scalar("train/reconstruction_observation_loss",
                               reconstruction_observation_loss.mean(), step)
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])
        self.writer.add_scalar("train/reward_loss", reward_loss.mean(), step)

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(
                self.device), kl_divergence_loss
        )
        self.writer.add_scalar("train/kl_divergence_loss",
                               kl_divergence_loss, step)
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        self.writer.add_scalar("train/model_loss", model_loss, step)
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()

    def behavior_learning(self, states, deterministics):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1,
                                               self.config.deterministic_size)

        # continue_predictor reinit
        for t in range(self.config.horizon_length):
            action = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(
                state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic
            )

        self._agent_update(self.behavior_learning_infos.get_stacked())

    def _agent_update(self, behavior_learning_infos):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        step_actor = self.actor_optimizer.state[self.actor_optimizer.param_groups[0]
                                                ["params"][-1]].get("step", 0)
        actor_loss = -torch.mean(lambda_values)
        self.writer.add_scalar("train/actor_loss", actor_loss, step_actor)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))
        step_critic = self.critic_optimizer.state[self.critic_optimizer.param_groups[0]
                                                  ["params"][-1]].get("step", 0)
        self.writer.add_scalar("train/value_loss", value_loss, step_critic)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.critic_optimizer.step()

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_step, train):
        posterior, deterministic = self.rssm.recurrent_model_input_init(1)
        action = torch.zeros(1, self.action_size).to(self.device)

        observation, _ = env.reset()
        embedded_observation = self.encoder(
            torch.from_numpy(observation).float().to(self.device)
        )

        ideal_action_num = 0
        total_action_num = 0

        for _ in range(num_interaction_step):
            deterministic = self.rssm.recurrent_model(
                posterior, action, deterministic
            )
            embedded_observation = embedded_observation.reshape(1, -1)
            _, posterior = self.rssm.representation_model(
                embedded_observation, deterministic
            )
            action = self.actor(posterior, deterministic).detach()

            if self.discrete_action_bool:
                buffer_action = action.cpu().numpy()
                env_action = buffer_action.argmax()

            else:
                buffer_action = action.cpu().numpy()[0]
                env_action = buffer_action

            next_observation, reward, done, truncated, info = env.step(
                env_action)
            if train:
                self.buffer.add(
                    observation, buffer_action, reward, next_observation, done
                )
            is_ideal_action = info["is_ideal_action"]
            ideal_action_num += int(is_ideal_action)
            total_action_num += 1
            embedded_observation = self.encoder(
                torch.from_numpy(next_observation).float().to(self.device)
            )
            observation = next_observation

        ideal_action_rate = 100 * ideal_action_num / total_action_num
        if train:
            self.num_total_episode += 1
            self.writer.add_scalar(
                "train/ideal_action_rate", ideal_action_rate, self.num_total_episode
            )
        else:
            print(
                f"global_step: {self.num_total_episode:07d}, ideal_action_rate: {ideal_action_rate:05.1f}%")
            self.writer.add_scalar(
                "eval/ideal_action_rate", ideal_action_rate, self.num_total_episode
            )
