import torch
import src.GoogLeNet as model
import src.dataset as dataset
from src.GoogLeNet import Inception
import src.inference_model as inference_model


if __name__ == '__main__':
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     dataloader = dataset.inference_dataloader()
     model = model.GoogLeNet(Inception).to(device)
     model.load_state_dict(torch.load("D:\AA_Py_learn\Cats_and_Dogs\outputs\checkpoints\\best_model_2025-03-07_19-43.pth"))
     inference_process = inference_model.model_inference(model, dataloader)