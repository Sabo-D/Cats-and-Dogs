from matplotlib import pyplot as plt
from datetime import datetime

def plot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_process.epoch, train_process.train_loss,'ro-', label="train_loss")
    plt.plot(train_process.epoch, train_process.val_loss,'bx-', label="val_loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(train_process.epoch, train_process.train_acc, 'ro-', label="train_acc")
    plt.plot(train_process.epoch, train_process.val_acc, 'bx-', label="val_acc")
    plt.legend()
    plt.show()

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_path = f'D:\AA_Py_learn\Cats_and_Dogs\outputs\logs\\train_logs\\acc_loss_{current_time}'
    plt.savefig(out_path + '.png')
