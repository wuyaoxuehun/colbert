import wandb

wandb.init(project="my-test-project1", entity="alovet")
wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
}
loss = 100
wandb.log({"loss": loss})

# Optional
# wandb.watch(model)
