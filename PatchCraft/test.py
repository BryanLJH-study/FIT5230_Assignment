import torch
import cv2
from patchcraft import PatchCraftDetector

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    classifier = PatchCraftDetector('D:/OneDrive - Monash University/Master in Artificial Intelligence/2024 Sem 2/FIT5230 Malicious AI/Assisgnment/Code/FIT5230_Assignment/PatchCraft/pretrained_example/pretrained_patchcraft_model_700.pth', device)

    img = cv2.imread('D:/OneDrive - Monash University/Master in Artificial Intelligence/2024 Sem 2/FIT5230 Malicious AI/Assisgnment/Code/FIT5230_Assignment/PatchCraft/pretrained_example/obama.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = classifier.classify(img)
    print(res)
    