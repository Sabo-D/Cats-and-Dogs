from tqdm import tqdm
import sys
import torch
from torch import nn
import time
from datetime import datetime
import pandas as pd


def model_test(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loss, test_acc = 0.0, 0.0
    test_correct, test_num = 0, 0
    criterion = nn.CrossEntropyLoss()

    since = time.time()
    with torch.no_grad():
        model.eval()
        for b_x, b_y in tqdm(test_dataloader, desc=f"Testing", file=sys.stdout):
            b_x, b_y = b_x.to(device), b_y.to(device)

            outputs = model(b_x)
            y_pred = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, b_y)

            test_loss += loss.item()
            test_correct += torch.sum(y_pred == b_y).item()
            test_num += 1

    time_elapsed = time.time() - since
    test_acc = test_correct / test_num
    test_loss = test_loss / test_num
    print("Test_loss:{:.4f}, Test_correct:{}, Test_acc:{:.4f}".format(test_loss, test_correct, test_acc))
    print('Testing time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    test_process = pd.DataFrame(data={
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_correct": test_correct,
    })
    log_path = f'D:\AA_Py_learn\Cats_and_Dogs\outputs\logs\\test_logs\\logs_{current_time}.csv'
    test_process.to_csv(log_path)
    print('测试过程已成功保存')

    return test_process


