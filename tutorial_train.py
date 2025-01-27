from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset, MyDatasetHC,MyDatasetCOCO,MyDatasetCOCO_val
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
torch.set_float32_matmul_precision('high')

# Configs
resume_path = './pretrained/control_sd15_ini.ckpt'
# resume_path = './image_log/checkpoint/last.ckpt'
batch_size = 2
logger_freq = 3000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDatasetCOCO(root='data/ms-coco')
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq,split='train_deconv00')
checkpoint_callback = ModelCheckpoint(
    dirpath=f'image_log/checkpoint_deconv00/', 
    save_top_k=-1,
    save_last=True,
    save_weights_only=False, 
    every_n_epochs=1,
)



trainer = pl.Trainer(devices=1, precision=32, callbacks=[logger], accumulate_grad_batches=2)


# # Train!
trainer.fit(model, dataloader)
