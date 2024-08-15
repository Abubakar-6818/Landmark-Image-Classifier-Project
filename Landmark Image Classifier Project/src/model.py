
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
        
#         self.conv_block = nn.Sequential(
#             OrderedDict([
#                 ("Conv1",
#                     nn.Sequential( 
#                         OrderedDict([
#                             ("Conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)),
#                             ("Batch_Norm", nn.BatchNorm2d(out_channels)),
#                             ("Relu", nn.ReLU(inplace=True)),
#                         ])
#                     )
#                 ),
#                 ("Conv2",
#                     nn.Sequential(
#                         OrderedDict([
#                             ("Conv", nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)),
#                             ("Batch_Norm", nn.BatchNorm2d(out_channels)),
#                             ("Relu", nn.ReLU(inplace=True)),
#                         ])
#                     )
#                 )
#             ])
#         )
#         self.relu = nn.ReLU()
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         F = self.conv_block(x)
#         residual = x
#         H = F + residual
#         result = self.relu(H)
#         return result

# def conv_layers(in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False):
#     conv_block = nn.Sequential(
#         OrderedDict([
#             ("Conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
#             ("Batch_Norm", nn.BatchNorm2d(out_channels)),
#             ("Relu", nn.ReLU(inplace=True)),
#         ])
#     )
    
#     layers = [("conv_block", conv_block)]
    
#     if pool:
#         layers.append(("max_pool", nn.MaxPool2d(kernel_size=2, stride=2)))
    
#     return nn.Sequential(OrderedDict(layers))

# def fc_layers(in_channels, out_channels, dropout):
#     layers = OrderedDict([
#         ("dropout", nn.Dropout(dropout)),
#         ("linear", nn.Linear(in_channels, out_channels)),
#         ("relu", nn.ReLU())
#     ])
    
#     return nn.Sequential(layers)

# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
#         super(MyModel, self).__init__()
        
#         self.model = nn.Sequential(OrderedDict([
#             ("conv_1", conv_layers(3, 64, kernel_size=7, stride=2, padding=3)),
#             ("resnet_block_1", ResidualBlock(64, 64)),
#             ("resnet_block_2", ResidualBlock(64, 64)),
            
#             ("conv_2", conv_layers(64, 128, kernel_size=3, stride=2, padding=1, pool=True)),
#             ("resnet_block_3", ResidualBlock(128, 128)),
#             ("resnet_block_4", ResidualBlock(128, 128)),
            
#             ("conv_3", conv_layers(128, 256, kernel_size=3, stride=1, padding=1, pool=True)),
#             ("resnet_block_5", ResidualBlock(256, 256)),
#             ("resnet_block_6", ResidualBlock(256, 256)),
            
#             ("conv_4", conv_layers(256, 512, kernel_size=3, stride=1, padding=1, pool=True)),
#             ("resnet_block_7", ResidualBlock(512, 512)),
#             ("resnet_block_8", ResidualBlock(512, 512)),
            
#             ("avgpool", nn.AdaptiveAvgPool2d(output_size=(3, 3))),
            
#             ("flatten", nn.Flatten()),
#             ("fc1", fc_layers(3 * 3 * 512, 1024, dropout=dropout)),
#             ("fc2", fc_layers(1024, num_classes, dropout=dropout))
#         ]))
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.model(x)
#         return x

        
        
            






# import torch 
# import torch.nn as nn
# from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            OrderedDict([
                ("Conv1", nn.Sequential( 
                    OrderedDict([
                        ("Conv", nn.Conv2d(inputs, outputs, kernel_size=3, padding="same")),
                        ("Batch_Norm", nn.BatchNorm2d(outputs)),
                        ("Relu", nn.ReLU()),
                    ])
                )),
                ("Conv2", nn.Sequential(
                    OrderedDict([
                        ("Conv", nn.Conv2d(outputs, inputs, kernel_size=3, padding="same")),
                        ("Batch_Norm", nn.BatchNorm2d(inputs))
                        #("Relu", nn.ReLU(inplace=True)),
                    ])
                ))
            ])
        )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        F = self.conv_block(x)
        
        H = F + x
        result = self.relu(H)
        return result

    
    
def conv_layers(in_channels,  out_channels, kernel_size = 3, stride = 1, padding = 1, pool = False):
    
    conv_block = nn.Sequential(
        
        OrderedDict([
            
             ("Conv", nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding)),
            ("Batch_Norm", nn.BatchNorm2d(out_channels)),
            ("Relu", nn.ReLU(inplace = True)),
           # ( "MaxPool", nn.MaxPool2d(2,2)), 
            
        ])
    )
    
    conv_layer = OrderedDict([
        ("conv_block",  conv_block)
    ])
    
    if pool:
        conv_layer["max_pool"] = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    return nn.Sequential(conv_layer)



def fc_layers(in_channels, out_channels, dropout):
    
    layers = OrderedDict(
    [
        ("dropout", nn.Dropout(dropout)),
          ("linear", nn.Linear(in_channels, out_channels)),
            ("relu", nn.ReLU())
            
    ])
    
    return nn.Sequential(layers)




class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super(MyModel, self).__init__()

        self.model = nn.Sequential(OrderedDict([
            ("conv_1", conv_layers(3, 64, kernel_size=7, stride=2, padding=3)),
            ("resnet_block_1", ResidualBlock(64, 64)),
            ("resnet_block_2", ResidualBlock(64, 64)),
            
            ("conv_2", conv_layers(64, 128, kernel_size=3, stride=2, padding=1, pool=True)),
            ("resnet_block_3", ResidualBlock(128, 128)),
            ("resnet_block_4", ResidualBlock(128, 128)),
            
            ("conv_3", conv_layers(128, 256, kernel_size=3, stride=2, padding=1, pool=True)),
            ("resnet_block_5", ResidualBlock(256, 256)),
            ("resnet_block_6", ResidualBlock(256, 256)),
            
            ("conv_4", conv_layers(256, 512, kernel_size=3, stride=2, padding=1, pool=True)),
            ("resnet_block_7", ResidualBlock(512, 512)),
            ("resnet_block_8", ResidualBlock(512, 512)),
            
            ("avgpool", nn.AdaptiveAvgPool2d(output_size=(3, 3))),
            
            ("flatten", nn.Flatten()),
            
            ("fc1", fc_layers(3 * 3 * 512, 1024, dropout=dropout)),
            ("fc2", nn.Linear(1024, num_classes))
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

            
        
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
            














# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

#         super().__init__()

#         # YOUR CODE HERE
 
#         self.model = nn.Sequential(
        
#                  nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1, bias = False ),
#                  nn.BatchNorm2d(16),
#                  nn.ReLU(inplace=True) ,
#                  nn.MaxPool2d(2,2),
                    
                    
#                 nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
#                 nn.BatchNorm2d(32),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2,2), 
                            
                            
                            
#                 nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2,2), 
                
                
#                 nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2,2), 
                                 
#                 nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2,2), 
#                 nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                 
#                 nn.Flatten(),
            
#            nn.Linear(256, 500), 
#                 nn.Dropout(p = dropout),
#                 nn.BatchNorm1d(500),
#                 nn.ReLU(),
            
#                 nn.Linear(500, 250), 
#                 nn.Dropout(p = dropout),
#                 nn.BatchNorm1d(250),
                
#                 nn.Linear(250, num_classes)
        
        
#         )
       
        
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))
     

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)
            
#         # F(X)
        
#         x = self.model(x)
                    
                    
#         return x

# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

#         super().__init__()

#         # YOUR CODE HERE
#         self.model = nn.Sequential(
        
#                  nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1, bias = False ),
#                  nn.BatchNorm2d(16),
#                  nn.ReLU(inplace=True) ,
#                  nn.MaxPool2d(2,2),
                    
                    
#                 nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
#                 nn.BatchNorm2d(32),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2,2), 
                            
                            
                            
#                 nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2,2), 
                
                
#                 nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2,2), 
                                 
#                 nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2,2), 
#                 nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                 
#                 nn.Flatten(),
            
#                 nn.Linear(256 , 500), 
#                 nn.Dropout(p = dropout),
#                 nn.BatchNorm1d(500),
#                 nn.ReLU(),
            
#                 nn.Linear(500, 250), 
#                 nn.Dropout(p = dropout),
#                 nn.BatchNorm1d(250),
                
#                 nn.Linear(250, num_classes)
        
#         )
#         self.relu = nn.ReLU(inplace=True)
        
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)
#         F = self.model(x)
        
#         # residual = x
#         # H = F + residual
#         # result = self.relu(H)
                    
                    
#         return F


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
