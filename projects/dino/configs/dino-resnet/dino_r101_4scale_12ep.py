from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify training config
train.init_checkpoint = "/path/to/r101.pkl"
train.output_dir = "./outputs/dino_r101_4scale_12ep"

# modify model config
model.backbone.stages.depth = 101
