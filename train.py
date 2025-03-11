import torch

import src.train_model as train_model
import src.dataset as dataset
import src.utils as plot
import src.ResNet_18 as model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader = dataset.train_val_dataloader()
    model = model.ResNet18(model.ResidualBlock).to(device)
    train_process = train_model.model_train(model, train_dataloader, val_dataloader, 10)
    plot_train_process = plot.plot_acc_loss(train_process)


