# # model.py
# import torch.nn as nn
# from torchvision.models import inception_v3

# class PatchClassifier(nn.Module):
#     def __init__(self):
#         super(PatchClassifier, self).__init__()
#         self.model = inception_v3(pretrained=True)
#         self.model.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(self.model.fc.in_features, 2)  # Assuming 2 classes: normal and tumor
#         )

#     def forward(self, x):
#         return self.model(x)
