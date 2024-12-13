import wandb
def init(args):
  if args.use_wandb:
    kwargs = dict(project="INAR", entity="camb-mphil", config=args)
    if args.wandb_name is not None:
      kwargs["name"] = f"{args.wandb_name}-{args.seed}"
    wandb.init(**kwargs)
    return wandb.config
  return args

def log(*args, **kwargs):
  if wandb.run is not None:
    wandb.log(*args, **kwargs)