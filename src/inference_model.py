from tqdm import tqdm
import sys
import torch
import time
import pandas as pd
from datetime import datetime


def model_inference(model, inference_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    names = []
    labels = []

    since = time.time()
    with torch.no_grad():
        model.eval()
        for images, image_names in tqdm(inference_dataloader, desc=f"Inferencing", file=sys.stdout):
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            labels.extend(predictions.cpu().numpy())
            names.extend(image_names)

    time_elapsed = time.time() - since
    print('Inferencing time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    results = ['cat' if label == 0 else 'dog' for label in labels]
    inference_process = pd.DataFrame(data={
        "names": names,
        "labels": labels,
        "results": results,
    })
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_path = f'D:\AA_Py_learn\Cats_and_Dogs\outputs\inference\\inference_{current_time}.csv'
    inference_process.to_csv(out_path, index=False)
    print('推理过程已成功保存')

    return inference_process