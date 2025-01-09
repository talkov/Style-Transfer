import torch
import torch.nn as nn
import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--train_dataset_path', default='/data/talkoz/Image_Style_Transfer/Classification_Task_Aviv/Datasets/WikiArt/Tal/traindataset.pth')
    parser.add_argument('--val_dataset_path', default='/data/talkoz/Image_Style_Transfer/Classification_Task_Aviv/Datasets/WikiArt/Tal/valdataset.pth')
    parser.add_argument('--test_dataset_path', default='/data/talkoz/Image_Style_Transfer/Classification_Task_Aviv/Datasets/WikiArt/Tal/testdataset.pth')



    # Training configuration.

    parser.add_argument('--batch_size', default = 32)
    parser.add_argument('--num_epochs', default = 1000)
    parser.add_argument('--lr', default = 1e-4)
    parser.add_argument('--image_size', default = 512) #according to the paper
    parser.add_argument('--l2_reg', default = 1e-3, help='l2 regularization parameter')
    parser.add_argument('--criterion', default = nn.CrossEntropyLoss())
    parser.add_argument('--device', default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--checkpoint_path', default=f"/data/talkoz/Image_Style_Transfer/Classification_Task_Aviv/trained_model", type=str, help='snapshot interval path')
    parser.add_argument('--snapshot', default = 10)
    parser.add_argument('--WANDB_TRACKING', default=False, type=bool, help='log to wandb or not')
    parser.add_argument('--load_model', default=False, type=bool, help='weather to load model fir re-training')
    parser.add_argument('--load_model_path', default='/data/talkoz/Image_Style_Transfer/Classification_Task_Aviv/trained_model/best_model.pth',
                    type=str, help='path for model to re-train')

    parser.add_argument('--use_scheduler', type=bool, default=False, help="weather to use scheduler")
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help="in how much to divide the lr")
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help="amount of episodes with no improvement to wait for lr reduction")

    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=500)

    # Avoid parsing issues in Jupyter/Colab
    config = parser.parse_args(args=[])

    return config
