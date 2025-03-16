# rlweight

Reinforcement Learning based Portfolio Weight Control Module.


## Install

```bash
poetry install
```


## Train

Example code to train DDPG agent.

```python
from rlweight.ddpg.trainer import DDPGTrainer
from rlweight.ddpg.config import TrainerConfig
from rlweight.env.eventenv import EventEnv

# Configuration
trainer_config = TrainerConfig(
    tau=0.0005,
    fee=0.001,
    gamma=0.9,
    lr_actor=1e-4,
    lr_critic=1e-4,
    num_tickers=29,
    total_steps=1000000,
    batch_size=64,
    buffer_size=500000,
    std_start=0.2,
    std_end=0.0005,
)

# DDPG Agent Trainer
trainer = DDPGTrainer(config=trainer_config, env_obj=EventEnv, data=train_data)

# Start DDPG Agent Train
trainer.train(mlflow_run="Experiment: DDPG")

# Evaluate DDPG Agent
train_pv = trainer.test()
```

## License

MIT License
