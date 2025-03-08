import torch

import src.test_model as test_model
import src.dataset as dataset
import src.model as model
from src.model import Inception

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataloader = dataset.test_dataloader()
    model = model.GoogLeNet(Inception).to(device)
    model.load_state_dict(torch.load('D:\AA_Py_learn\Cats_and_Dogs\outputs\checkpoints\\best_model_2025-03-07-43.pth'))
    test_process = test_model.model_test(model, test_dataloader)


