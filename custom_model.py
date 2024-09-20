from torch import nn


#model
class Custom_Resnet(nn.Module):
    def __init__(self,model,num_classes=3):
        super(Custom_Resnet,self).__init__()
        self.num_classes=num_classes
        self.model=model
        for params in self.model.parameters():
            params.requires_grad=True
        self.classifier=nn.Sequential(
                    nn.Linear(self.model.fc.in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, self.num_classes),  # Normal vs Infiltration
                    nn.Softmax(dim=1)
        )
        self.model.fc=self.classifier
    def forward(self,x):
        bs=x.shape[0]
        x=self.model(x)
        return x


#load the best model
