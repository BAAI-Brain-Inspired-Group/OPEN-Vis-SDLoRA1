from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDatasetCOCO
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
torch.set_float32_matmul_precision('high')

torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

# Configs
resume_path = './pretrained/control_sd15_ini.ckpt'
batch_size = 8
logger_freq = 500
learning_rate = 5e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15_lora.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


train_dataset = MyDatasetCOCO(root='data/ms-coco')
val_dataset = MyDatasetCOCO(root='data/ms-coco', train=False)
train_dataloader = DataLoader(train_dataset, num_workers=16, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, num_workers=16, batch_size=batch_size, shuffle=False, pin_memory=True)
logger = ImageLogger(batch_frequency=logger_freq,split='train_lora')
checkpoint_callback = ModelCheckpoint(
    dirpath=f'image_log/checkpoint_lora01/', 
    save_top_k=-1,
    save_last=True,
    save_weights_only=False, 
    every_n_epochs=1,
)

# trainer = pl.Trainer(devices=1, precision='bf16-mixed', callbacks=[logger])
trainer = pl.Trainer(devices=1, precision=32, callbacks=[logger, checkpoint_callback], max_epochs=15)
# trainer = pl.Trainer(strategy='ddp', accelerator='gpu', devices=1, precision=32, callbacks=[logger,checkpoint_callback], accumulate_grad_batches=4)

# # Train!
trainer.fit(model, train_dataloader, val_dataloader)

# model_path = 'image_log/checkpoint_lora01/epoch=5-step=88716.ckpt'
# trainer.validate(model, val_dataloader, model_path)

