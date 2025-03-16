import torch
import mlflow
import numpy as np
from datetime import datetime
from dataclasses import dataclass

# torch
from torch.optim import Adam
from torch.nn import MSELoss

# Model
from rlweight.bcq.model import VAE
from rlweight.bcq.model import Qnet
from rlweight.bcq.model import Perturbation
from rlweight.bcq.model import ModelConfig
from rlweight.ddpg.trainer import ReplayBuffer


@dataclass
class TrainerConfig:
    lr_actor: float
    lr_critic: float
    lr_purturb: float
    action_scale: float
    total_steps: int
    batch_size: int
    gamma: float
    num_tickers: int
    tau: float
    lam: float


class BCQTrainer:
    def __init__(self, config: TrainerConfig, buffer: ReplayBuffer):
        self.config = config
        self.buffer = buffer

        # Actor
        self.actor = VAE(
            config=ModelConfig(
                num_tickers=config.num_tickers,
                action_scale=config.action_scale,
            )
        )
        # Critic
        self.critic = Qnet(
            config=ModelConfig(
                num_tickers=config.num_tickers,
                action_scale=config.action_scale,
            )
        )
        # Perturbation
        self.perturbation = Perturbation(
            config=ModelConfig(
                num_tickers=config.num_tickers,
                action_scale=config.action_scale,
            )
        )
        # Target Networks
        self.purturb_target = Perturbation(
            config=ModelConfig(
                num_tickers=config.num_tickers,
                action_scale=config.action_scale,
            )
        )
        self.critic_target = Qnet(config)

        self.purturb_target.load_state_dict(self.perturbation.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.lr_critic)
        self.perturb_optimizer = Adam(
            self.perturbation.parameters(), lr=self.config.lr_purturb
        )

    @staticmethod
    def to_tensor(data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data).float()

    def train(self, verbose: bool = True, mlflow_run: str = None):
        """
        Train the Actor and Critic
        """
        criterion = MSELoss()

        # MLflow 설정
        if mlflow_run:
            run_name = f"{self.config.__str__()}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            mlflow.set_experiment(mlflow_run)
            mlflow.start_run(run_name=run_name)
            mlflow.log_params(self.config.__dict__)

        for step in range(1, self.config.total_steps + 1):
            transitions = self.buffer.sample(self.config.batch_size)

            s, a, r, ns, done = transitions

            s = BCQTrainer.to_tensor(s)
            a = BCQTrainer.to_tensor(a)
            r = BCQTrainer.to_tensor(r)
            ns = BCQTrainer.to_tensor(ns)
            done = BCQTrainer.to_tensor(done)

            # VAE 업데이트
            recon, mu, std = self.actor(s, a)
            recon_loss = criterion(recon, a)
            KL = -0.5 * (1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL

            self.actor_optimizer.zero_grad()
            vae_loss.backward()
            self.actor_optimizer.step()

            # Soft CLipped Double Q-learning
            with torch.no_grad():
                # 배치마다 100개 반복 생성
                ns = torch.repeat_interleave(ns, 100, dim=0)
                next_a = self.actor.decoder(ns)
                next_pa = self.perturbation(ns, next_a)
                next_q1 = self.critic.q1(ns, next_pa)
                next_q2 = self.critic.q2(ns, next_pa)

                target = self.config.lam * torch.min(next_q1, next_q2) + (
                    1 - self.config.lam
                ) * torch.max(next_q1, next_q2)

                # 배치마다 반복 생성 100개 중 최대 값 선택
                target = target.reshape(self.config.batch_size, -1).max(1)[0]
                target = target.reshape(self.config.batch_size, 1)
                target = r + self.config.gamma * target * (1 - done)

            q1 = self.critic.q1(s, a)
            q2 = self.critic.q2(s, a)

            critic_loss = criterion(q1, target) + criterion(q2, target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Perturbation 업데이트
            actions = self.actor.decoder(s).detach()
            pactions = self.perturbation(s, actions)

            perturb_loss = -self.critic.q1(s, pactions).mean()

            self.perturb_optimizer.zero_grad()
            perturb_loss.backward()
            self.perturb_optimizer.step()

            self._soft_update(self.perturbation, self.purturb_target, self.config.tau)
            self._soft_update(self.critic, self.critic_target, self.config.tau)

            # MLflow 로깅
            if mlflow_run:
                mlflow.log_metrics(
                    {
                        "vae_loss": vae_loss.item(),
                        "critic_loss": critic_loss.item(),
                        "perturb_loss": perturb_loss.item(),
                    },
                    step=step,
                )

            if verbose and step % 100 == 0:
                print(f"\n[Step {step}/{self.config.total_steps}]")
                print(f"  VAE Loss: {vae_loss.item():.6f}")
                print(f"  Critic Loss: {critic_loss.item():.6f}")
                print(f"  Perturb Loss: {perturb_loss.item():.6f}")

    def _soft_update(self, net, target_net, tau):
        """
        Target Network Soft Update
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
