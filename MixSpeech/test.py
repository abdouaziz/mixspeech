import wandb




import wandb

wandb.init(project="my-test-project", entity="abdouaziz")

 
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}

for i in range (1000):
    wandb.log({"loss": i})
