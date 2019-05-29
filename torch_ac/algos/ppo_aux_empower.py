import numpy
import torch

from torch_ac.algos.ppo import PPOAlgo
from torch_ac.utils import DictList


class PPOAuxEmpowerAlgo(PPOAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347)).
    See [Mohamed et al., 2015](https://arxiv.org/abs/1509.08731) for details of empowerment.
    """

    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, aux_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, use_aux_reward=False, aux_reward_coef=0.1,
                 shaping_aux_reward=False,
                 empower_beta_coef=1.0, empower_value_loss_coef=0.5):
        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda,
                         entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                         adam_eps, clip_eps, epochs, batch_size, preprocess_obss,
                         reshape_reward)

        self.aux_loss_coef = aux_loss_coef
        self.use_aux_reward = use_aux_reward
        self.shaping_aux_reward = shaping_aux_reward
        self.aux_reward_coef = aux_reward_coef
        self.empower_beta_coef = empower_beta_coef
        self.empower_value_loss_coef = empower_value_loss_coef

        shape = (self.num_frames_per_proc, self.num_procs)
        self.empower_values = torch.zeros(*shape, device=self.device)
        self.sample_entropies = torch.zeros(*shape, device=self.device)
        self.prev_aux_logprobs = torch.zeros(*shape, device=self.device)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                masked_prev_action = self.prev_action * self.mask.int()
                if self.acmodel.recurrent:
                    dist, value, memory, auxdist, empower_value = self.acmodel(
                        preprocessed_obs, masked_prev_action,
                        self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value, _, auxdist, empower_value = self.acmodel(
                        preprocessed_obs, masked_prev_action)
            action = dist.sample()
            # Auxiliary reward is expected mutual information

            prev_aux_logprob = auxdist.log_prob(masked_prev_action)
            sample_entropy = -dist.log_prob(action)

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            self.prev_actions[i] = self.prev_action
            self.prev_action[:] = action
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.empower_values[i] = empower_value
            self.sample_entropies[i] = sample_entropy
            if i != 0:
                self.prev_aux_logprobs[i-1] = prev_aux_logprob
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            masked_prev_action = self.prev_action * self.mask.int()
            if self.acmodel.recurrent:
                _, next_value, _, next_auxdist, _ = self.acmodel(
                    preprocessed_obs, masked_prev_action,
                    self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value, _, next_auxdist, _ = self.acmodel(
                    preprocessed_obs, masked_prev_action)

        self.prev_aux_logprobs[self.num_frames_per_proc-1] = \
            next_auxdist.log_prob(masked_prev_action)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            if self.use_aux_reward:
                if i < self.num_frames_per_proc - 1:
                    next_value = self.values[i+1] + \
                        self.aux_reward_coef * self.empower_values[i+1]
                else:
                    next_value = next_value
            else:
                next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            if self.use_aux_reward and self.shaping_aux_reward:
                delta -= self.aux_reward_coef * self.empower_values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.prev_action = self.prev_actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.empower_value = self.empower_values.transpose(0, 1).reshape(-1)
        exps.sample_entropy = self.sample_entropies.transpose(0, 1).reshape(-1)
        exps.prev_aux_logprob = self.prev_aux_logprobs.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_sample_entropies = []
            log_prev_aux_logprobs = []
            log_empower_values = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_empower_value_losses = []
            log_aux_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_sample_entropy = 0
                batch_prev_aux_logprob = 0
                batch_empower_value = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_empower_value_loss = 0
                batch_aux_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]
                    masked_prev_action = sb.prev_action * sb.mask.squeeze(1).int()

                    # Compute loss
                    if self.acmodel.recurrent:
                        dist, value, memory, auxdist, empower_value = self.acmodel(
                            sb.obs, masked_prev_action, memory * sb.mask)
                    else:
                        dist, value, _, auxdist, empower_value = self.acmodel(
                            sb.obs, masked_prev_action)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    aux_loss = -auxdist.log_prob(masked_prev_action).mean()
                    empower_value_loss = (
                        self.empower_beta_coef * sb.prev_aux_logprob.detach() - \
                        (dist.log_prob(sb.action) + empower_value)
                    ).pow(2).mean()

                    loss = policy_loss - self.entropy_coef * entropy \
                        + self.value_loss_coef * value_loss \
                        + self.aux_loss_coef * aux_loss \
                        + self.empower_value_loss_coef * empower_value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_sample_entropy += sb.sample_entropy.mean().item()
                    batch_prev_aux_logprob += sb.prev_aux_logprob.mean().item()
                    batch_value += value.mean().item()
                    batch_empower_value += empower_value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_aux_loss += aux_loss.item()
                    batch_empower_value_loss += empower_value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_sample_entropy /= self.recurrence
                batch_prev_aux_logprob /= self.recurrence
                batch_value /= self.recurrence
                batch_empower_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_empower_value_loss /= self.recurrence
                batch_aux_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_sample_entropies.append(batch_sample_entropy)
                log_prev_aux_logprobs.append(batch_prev_aux_logprob)
                log_values.append(batch_value)
                log_empower_values.append(batch_empower_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_empower_value_losses.append(batch_empower_value_loss)
                log_aux_losses.append(batch_aux_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "sample_entropy": numpy.mean(log_sample_entropies),
            "prev_aux_logprob": numpy.mean(log_prev_aux_logprobs),
            "value": numpy.mean(log_values),
            "empower_value": numpy.mean(log_empower_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "empower_value_loss": numpy.mean(log_empower_value_losses),
            "aux_loss": numpy.mean(log_aux_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs
