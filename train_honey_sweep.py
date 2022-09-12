import numpy as np
import torch 
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import WeightedRandomSampler

# from src.dm_luis_smp_ESRI import EsriDataModule as DataModule
# from src.models_ESRGAN_dual import SMP_NewDual as SMP
from src.dm_honey import HoneyDataModule as DataModule
from src.models_sweep import PLModel
import pandas as pd
import timm
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
# from pytorch_lightning.callbacks import ModelCheckpoint 
import torch.nn.functional as F
import sys
import os
import warnings
import yaml
warnings.simplefilter('ignore')
device='cuda' if torch.cuda.is_available() else 'cpu'
import wandb


config_def = dict(
    experiment_prefix= 'Sweep_LR_BZ_DenseNet121',
    model_name= 'densenet121',
    lr= 0.0005,
    batch_size= 4,
    gpus=1,
    precision= 16,
    max_epochs= 30,
    log= True,
    num_workers= 4, 
    resume=None,
    num_classes=12,
    use_callback=True,
    save_path= "/mnt/gpid08/datasets/remote_sensing/tmp_from_gpid07/honey/results/",
)


wandb.init(config=config_def)
# Config parameters are automatically set by W&B sweep agent
config = wandb.config


def get_class_weights(datamodule, num_classes):
    # dm = DualDataModule(batch_size_val=2, batch_size_train=1)
    # dm=datamodule(batch_size_train=1)
    total_count_classes = np.zeros(num_classes)
    for batch_i, data in enumerate(datamodule.train_dataloader()):
        # print(batch_i, data.keys(), data['gt'].shape)
        classes, count_classes = np.unique(data['target'], return_counts=True)
        # print(classes, count_classes)
        for class_value, class_count in zip(classes, count_classes):
            total_count_classes[class_value] += class_count
        # print(f'totalclass-{batch_i} : {total_count_classes}')
        # print(f'Suma: {total_count_classes.sum()}')
        # print(f'weights: {1/np.log(1.02+ (total_count_classes/total_count_classes.sum()))}')
        # if batch_i ==2:
        #     break
    class_weights = torch.from_numpy(1/np.log(1.02+ (total_count_classes/total_count_classes.sum())))
    # print(batch_i)
    # print(f'CE weights: {class_weights}')
    return class_weights


def train(config):
    pl.seed_everything(42)  
    ### Read Dataset splits ###
    df_train = pd.read_csv('/home/usuaris/imatge/lsalgueiro/git/honey/List_train.csv')
    df_val = pd.read_csv('/home/usuaris/imatge/lsalgueiro/git/honey/List_val.csv')

    print(f'Datasets: Train {len(df_train)} \t Val: {len(df_val)}')

    ## create sampler ##
    # count_train = df_train.groupby('labels').agg({'count'})
    # print(count_train)
    # count_train['weights'] = 1.02/count_train
    # count_train['weights_norm'] = count_train['weights']/count_train['weights'].sum()
    count_train = df_train.groupby('labels').agg({'count'})
    count_train['weights'] = 1.02/count_train['name']
    count_train['weights_norm'] = count_train['weights']/count_train['weights'].sum()
    # count_train['weights_nor'].sum()
    # count_train
    # 1/np.log(1.02+ (total_count_classes/total_count_classes.sum()

    ## Getting class Weights ###
    class_weights = torch.Tensor([count_train.loc['Pinus','weights_norm'].item(),\
                      count_train.loc['Erica.m','weights_norm'].item(),\
                      count_train.loc['Cistus sp','weights_norm'].item(),\
                      count_train.loc['Lavandula','weights_norm'].item(),\
                      count_train.loc['Citrus sp','weights_norm'].item(),\
                      count_train.loc['Helianthus annuus','weights_norm'].item(),\
                      count_train.loc['Eucalyptus sp.','weights_norm'].item(),\
                      count_train.loc['Rosmarinus officinalis','weights_norm'].item(),\
                      count_train.loc['Brassica','weights_norm'].item(),\
                      count_train.loc['Cardus','weights_norm'].item(),\
                      count_train.loc['Tilia','weights_norm'].item(),\
                      count_train.loc['Taraxacum','weights_norm'].item(),\
                      ])
    # class_weights

    ## Getting the Sampler ###
    samples_weight = np.array([count_train.loc[t,'weights_norm'].item() for t in df_train['labels']])
    samples_weight=torch.from_numpy(samples_weight)
    # print(samples_weight.shape)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    ## DEFINE DATASET AND DATAMODULE
    # dm = DataModule()
    dm = DataModule(
        batch_size=int(config['batch_size']),
        sampler  = sampler,\
        df_train = df_train,\
        df_val   = df_val,\
        workers= int(config['num_workers'])
        )

    ## DEFINE WEIGHT CLASSES (for CEw) ##
    # class_weights=get_class_weights(dm, config['num_classes'])

    # experiment_name: "Baseline_DLV3P_resnext50_32x4d_BCE-notFL_lr1e-3_MultiStep2"
    experiment_name = str(config['experiment_prefix'])
    checkpoint_name_prefix = str(config['experiment_prefix'])
    save_path = config['save_path'] + experiment_name  #'/mnt/gpid07/users/luis.salgueiro/segmentation/canarias/maspalomas/NewDualSR/full_v15_dataset3/'
    save_path_pngs  = save_path+"/pngs/"
    save_path_ious  = save_path+"/ious/"
    save_path_preds = save_path+"/preds/"
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(save_path_pngs,exist_ok=True)
    os.makedirs(save_path_ious,exist_ok=True)
    os.makedirs(save_path_preds,exist_ok=True)
    # print(os.path.isdir(save_path))

    ###################################
    ## DEFINE MODEL ###
    ######################################
    base_model = timm.create_model(config['model_name'],\
        pretrained='True',\
        num_classes=12)
    print(f"Defined Base model: {config['model_name']} ")
    
    model = PLModel( model = base_model,\
        config=config).to(device)

    # dm = DataModule(**config)


    ## DEFINE LOGGER ###

    wandb_logger = WandbLogger(project="Honey",
                            config=config,
                            save_dir=f'{config["save_path"]}/{experiment_name}',
                            name=experiment_name)
    parallel    = "ddp" if config["gpus"] > 1 else None 
    resume_dir  = config['resume'] if config['resume']  else None
    model_name  = config['model_name']

    if config['use_callback']:
        checkpoint = ModelCheckpoint(
            save_weights_only=True,
            dirpath=f'{save_path}',
            # filename=f'{checkpoint_name_prefix}-{datetime.datetime.now().strftime("%d")}{datetime.datetime.now().strftime("%m")}{datetime.datetime.now().strftime("%y")}-'+'{epoch}-{val_iou:.3f}',
            # filename=f"Best_-{{AVG_VAL_IOU:.4f}}",
            filename=f"Best-{{epoch}}-{{val_loss:.2f}}-{{avg_val_f1w:.2f}}",
            save_top_k = 1, 
            monitor = 'EarlyStop_Log', 
            mode = 'max',
            verbose = True,
            # period = 1,
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stop_callback = EarlyStopping(
            monitor='EarlyStop_Log',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='min'
            )
        cbs = [early_stop_callback, checkpoint, lr_monitor]
    trainer = pl.Trainer(
        gpus                    = config['gpus'],
        accelerator             = parallel, #None,
        precision               = config['precision'] if config['precision'] else 16,
        logger                  = wandb_logger if config['log'] else None,
        max_epochs              = config['max_epochs'],
        callbacks               = cbs if config['use_callback'] else None,
        deterministic           = False,
        num_nodes               = 1,
        amp_backend             = "native",
        # amp_level               = "O2",
        accumulate_grad_batches = 1,
        resume_from_checkpoint  = resume_dir,
        # overfit_batches    = 1 
        # profiler           = True,
        # limit_train_batches = 1,
        # limit_val_batches   = 1,
        # # AdvancedProfiler(output_filename='Full_profile_DebugDataset_BZ4_2GPUsv3.txt' ),\
    )
    trainer.fit(model, dm)

    # #########################3
    # ## FAST_DEV_RUN ###
    # #########################3
    # trainer = pl.Trainer(fast_dev_run=True)
    # trainer.fit(model,dm)
    # #########################3
    wandb_logger.experiment.finish()


if __name__ == '__main__':
    print(f'*** Starting run with {config}')
    # config_file = sys.argv[1]
    # if config_file:
    #     with open(config_file, 'r') as stream:
    #         loaded_config = yaml.safe_load(stream)
    #     config.update(loaded_config)
    train(config)