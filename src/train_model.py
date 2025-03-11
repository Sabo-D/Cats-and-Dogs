import copy
import time
import sys
import torch
from tqdm import tqdm
import pandas as pd
from datetime import datetime


def model_train(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    for epoch in range(num_epochs):
        since = time.time()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 10)

        train_loss, val_loss = 0.0, 0.0
        train_correct, val_correct = 0.0, 0.0
        train_num, val_num = 0, 0

        model.train()
        for b_x, b_y in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", file=sys.stdout):
            b_x, b_y = b_x.to(device), b_y.to(device)
            b_num = b_x.size(0)

            optimizer.zero_grad()
            outputs = model(b_x)
            y_pred = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_num
            train_correct += torch.sum(y_pred == b_y).item()
            train_num += b_num

        with torch.no_grad():
            model.eval()
            for b_x, b_y in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}", file=sys.stdout):
                b_x, b_y = b_x.to(device), b_y.to(device)
                b_num = b_x.size(0)

                outputs = model(b_x)
                loss = criterion(outputs, b_y)
                y_pred = torch.argmax(outputs, dim=1)

                val_loss += loss.item() * b_num
                val_correct += torch.sum(y_pred == b_y).item()
                val_num += b_num

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_correct / train_num)
        val_acc_all.append(val_correct / val_num)

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{} Train loss:{:.4f}  Train acc:{:.4f}'.format(epoch + 1, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val loss:{:.4f}    val acc:{:.4f}'.format(epoch + 1, val_loss_all[-1], val_acc_all[-1]))
        print('Training time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    model_path = f'D:\AA_Py_learn\Cats_and_Dogs\outputs\checkpoints\\best_model_{current_time}.pth'
    torch.save(best_model_wts, model_path)

    train_process = pd.DataFrame(data={
        "epoch": range(1, num_epochs + 1),
        "train_loss": train_loss_all,
        "train_acc": train_acc_all,
        "val_loss": val_loss_all,
        "val_acc": val_acc_all
    })
    log_path = f'D:\AA_Py_learn\Cats_and_Dogs\outputs\logs\\train_logs\\logs_{current_time}.csv'
    train_process.to_csv(log_path)
    print('训练过程已成功保存')

    return train_process