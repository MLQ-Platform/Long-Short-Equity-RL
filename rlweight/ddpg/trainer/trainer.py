import torch
import mlflow
import numpy as np
from datetime import datetime
from dataclasses import dataclass

# torch
from torch.optim import Adam
from torch.nn import MSELoss

# Model
from rlweight.ddpg.model.actor import Actor
from rlweight.ddpg.model.critic import Critic
from rlweight.ddpg.model.config import ModelConfig
from rlweight.ddpg.trainer.buffer import ReplayBuffer

# Env
from rlweight.env.data import Data
from rlweight.env.eventenv import EventEnvConfig


@dataclass
class TrainerConfig:
    lr_actor: float
    lr_critic: float
    gamma: float
    num_tickers: int
    total_steps: int
    batch_size: int
    tau: float
    fee: float
    buffer_size: int
    std_start: float
    std_end: float


class DDPGTrainer:
    def __init__(self, config: TrainerConfig, env_obj, data: Data):
        self.config = config
        # Env
        self.env = env_obj(
            config=EventEnvConfig(
                gamma=config.gamma,
                fee=config.fee,
            ),
            data=data,
        )
        # Actor
        self.actor = Actor(
            config=ModelConfig(
                num_tickers=config.num_tickers,
            )
        )
        # Critic
        self.critic = Critic(
            config=ModelConfig(
                num_tickers=config.num_tickers,
            )
        )
        # Target Networks
        self.actor_target = Actor(config)
        self.critic_target = Critic(config)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.lr_critic)

        # Noise for exploration
        self.std = self.config.std_start
        # Noise Exponential Decay
        self.std_decay = np.exp(
            np.log(self.config.std_end / self.config.std_start)
            / self.config.total_steps
        )

        # buffer
        self.buffer = ReplayBuffer(max_size=self.config.buffer_size)

    @staticmethod
    def to_tensor(data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data).float()

    def train(self, verbose: bool = True, mlflow_run: str = None):
        """
        Train Loop
        """

        steps = 0
        score = 0

        # MLflow 설정
        if mlflow_run:
            run_name = f"{self.config.__str__()}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            mlflow.set_experiment(mlflow_run)
            mlflow.start_run(run_name=run_name)
            mlflow.log_params(self.config.__dict__)

        # (num_tickers, 2)
        state = self.env.reset()

        while steps < self.config.total_steps:
            # (1, 1)
            action = self.actor(self.to_tensor(state).unsqueeze(0))
            # (1,)
            action = action.detach().squeeze(0).numpy()
            # Exploration Noise 추가
            action += np.random.normal(0, self.std, size=action.shape)
            # 0 ~ 1로 바운드
            action = np.clip(action, 0, 1)
            # (num_tickers,)
            target_weight = action * state[:, 0] + (1 - action) * state[:, 1]
            # (num_tickers,)
            gap = target_weight - state[:, 1]

            # 환경 실행
            next_state, reward, done, info = self.env.execute(state, gap)

            # 배치 단위 트렌지션
            transition = (
                state[np.newaxis],
                action[np.newaxis],
                reward[np.newaxis],
                next_state[np.newaxis],
                done[np.newaxis],
            )

            self.buffer.add(transition)
            score += 1e-4 * (reward.item() - score)
            state = next_state

            if len(self.buffer) >= self.config.batch_size:
                sampled_data = self.buffer.sample(self.config.batch_size)
                update_result = self._update(*sampled_data)

                # MLflow 로깅
                if mlflow_run:
                    mlflow.log_metrics(
                        {
                            **update_result,
                            "exploration_noise": self.std,
                            "action": action.item(),
                            "score": score,
                        },
                        step=steps,
                    )

            # Exploration Noise Decay
            self.std *= self.std_decay
            # Step Up
            steps += 1

            if done:
                state = self.env.reset()

            if verbose and steps % 100 == 0:
                print(f"\n[Step {steps}/{self.config.total_steps}]")
                print(f"  Critic Loss: {update_result['critic_loss']:.6f}")
                print(f"  Actor Loss: {update_result['actor_loss']:.6f}")
                print(f"  Avg Value: {update_result['avg_value']:.6f}")
                print(f"  Exploration Noise: {self.std:.6f}")
                print(f"  Action: {action.item():.6f}")
                print(f"  Score: {score:.6f}")

        if mlflow_run:
            mlflow.end_run()

    def test(self) -> list[dict]:
        """
        Test Loop
        """

        results = []
        self.actor.eval()

        state = self.env.reset()

        while True:
            action = self.actor(self.to_tensor(state).unsqueeze(0))
            action = action.detach().squeeze(0).numpy()

            target_weight = action * state[:, 0] + (1 - action) * state[:, 1]
            gap = target_weight - state[:, 1]

            next_state, reward, done, info = self.env.execute(state, gap)
            state = next_state

            results.append(info)

            if done:
                break

        self.actor.train()
        return results

    def _update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ):
        """
        DDPG Update
        """
        states = self.to_tensor(states)
        actions = self.to_tensor(actions)
        rewards = self.to_tensor(rewards)
        next_states = self.to_tensor(next_states)
        dones = self.to_tensor(dones)

        criterion = MSELoss()

        # **Target Q-value 계산 (TD 타겟)**
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.config.gamma * target_Q

        # **Critic 업데이트 (MSE Loss)**
        current_Q = self.critic(states, actions)
        critic_loss = criterion(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # **Actor 업데이트 (Deterministic Policy Gradient)**
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # **Target Network Soft Update**
        self._soft_update(self.actor, self.actor_target, self.config.tau)
        self._soft_update(self.critic, self.critic_target, self.config.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "avg_value": current_Q.mean().item(),
        }

    def _soft_update(self, net, target_net, tau):
        """
        Target Network Soft Update
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
