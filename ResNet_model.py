from torchvision import models
# Define the model and modify it for the WikiArt dataset
class StyleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(StyleClassifier, self).__init__()  # StyleClassifier
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
