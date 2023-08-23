from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from utils import *


if __name__ == '__main__':
    # 设置参数
    args=func_config()

    # 设置种子
    pl.seed_everything(0)

    # 建立模型
    light_model = BuildLightning(batch_size=args.batch_size)

    # 建立训练器
    trainer = Trainer(max_epochs=args.max_epochs,
                      logger=TensorBoardLogger("logs"),
                      accelerator="gpu",
		      gpus=args.gpus,
                      default_root_dir='checkpoint',
                      callbacks=[pl.callbacks.ModelCheckpoint(every_n_train_steps=100, save_top_k=-1)]
                      )

    # 模型训练
    trainer.fit(model=light_model)

    # 模型推理
    trainer.test()
