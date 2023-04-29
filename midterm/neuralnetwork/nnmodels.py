import torch.nn as nn
def get_model(name):
    if name == 'base':
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=1), # 20
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=5, padding=1), # 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),       

            nn.Conv2d(64, 64, kernel_size=5, padding=1), # 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),     

            nn.Flatten(),

            nn.Linear(64*2*2, 500),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(500, 3)
        )
        return model
    else:
        raise NotImplementedError