import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import os

LABELS = ['bak kut teh', 'chai tow kuay', 'cheng tng', 'Laksa', 'Mee siam', 'mixed vegetables', 'nasi lemak', 'Prawn Noodle', 'sliced fish soup', 'yong tau foo']

MODEL_PATH = os.path.join(os.path.dirname(__file__), "11JanModelV0.pth")

class FoodModelV0(nn.Module):
  def __init__(self,
               input_shape,
               output_shape,
               sample_X,
               hidden_units=10):
    super().__init__()
    self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=1,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=1,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    r = torch.randn(sample_X.shape)
    r = self.conv_block2(self.conv_block1(r))
    # Derive the input_features needed for the Linear layer
    in_feat = r.shape[1] * r.shape[2] * r.shape[3]

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=in_feat,
                  out_features=output_shape)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.conv_block2(self.conv_block1(x)))
  
def preprocess_image(image_path):
    data_transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        # Resize to 64 x 64 for faster processing, without losing too much detail
        transforms.ToTensor()
        # Convert it to a tensor
    ])
    img = Image.open(image_path).convert("RGB")
    # Unsqueeze because model expects [Batch_size, C, H, W]
    return data_transform(img).unsqueeze(dim=0)

def load_model(saved_model=MODEL_PATH):
   model = FoodModelV0(input_shape=3, # Number of Colour Channels
                      output_shape=10, # Number of classes
                      sample_X=torch.rand(32, 3, 32, 32),
                      hidden_units=20)
   model.load_state_dict(torch.load(saved_model))
   model.eval()
   return model

def predict(image_path):
   model = load_model()
   input = preprocess_image(image_path)
   label = model(input).softmax(dim=1).argmax(dim=1)
   return LABELS[label]
