import torch.nn as nn
import torch

DEBUG = False

# Gradient Reversal Layer (GRL)
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)

class DomainDiscriminator(nn.Module):
    def __init__(self,embed_dim=512, num_domains=2, alpha=1.0):
        super(DomainDiscriminator, self).__init__()

        self.alpha = alpha
        self.num_domains = num_domains

        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),  # Another hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),  # Another hidden layer
            nn.ReLU(),
            nn.Linear(512, embed_dim),  # Another hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(embed_dim, num_domains)  # Predict subject
        # self.cls_sigmoid = nn.Sigmoid()
    
    def forward(self, x, alpha=None):
        if alpha is None: 
            alpha = self.alpha
            
        reversed_x = grad_reverse(x, alpha)  # Apply GRL
        domain_features = self.feature_extractor(reversed_x)  # Extract subject features
        domain_pred = self.classifier(domain_features)  # Predict subject
        # domain_pred = self.cls_sigmoid(domain_pred)
        return domain_pred, domain_features  # Return both


class MyNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MyNet, self).__init__()

        n_classes = 10
        if "n_classes" in kwargs:
            n_classes = kwargs["n_classes"]
        else:
            print("n_classes arg not found, using 10 by default")
        image_size = 224
        if "image_size" in kwargs:
            image_size = kwargs["image_size"]
        else:
            print("image_size arg not found, using 224 by default")
        num_domains = 2
        if "num_domains" in kwargs:
            num_domains = kwargs["num_domains"]
        else:
            print("num_domains arg not found, using 2 by default")

        self.adv_training = False
        if "adv_training" in kwargs:
            self.adv_training = kwargs["adv_training"]
        else:
            print("adv_training arg not found, using False by default")



        self.conv_model = nn.Sequential(
            nn.Conv2d(3,16,(3,3), stride=(1,1), padding='valid'),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(16,32,(3,3), stride=(1,1), padding='valid'),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(32,8,(3,3), stride=(1,1), padding='valid'),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU(),
        )

        dummy_input = torch.rand(size=(1,3,image_size,image_size), dtype=torch.float32)
        conv_out = self.getConv(dummy_input)
        flat_out = nn.Flatten()(conv_out)
        # print(flat_out.size(0))

        self.features_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_out.size(-1), 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(512, n_classes)
        )
        self.domain_desc = DomainDiscriminator(embed_dim=512,num_domains=num_domains)

        del dummy_input, conv_out, flat_out

    
    def residual(self, x):
        pass

    def getheadFeatures(self, x):
        head_f = self.features_head(x)
        if DEBUG: print(f"head_f shape: {head_f.shape}")
        return head_f

    
    def getClassification(self, x):
        cls_h = self.cls_head(x)
        if DEBUG: print(f"cls_h shape: {cls_h.shape}")
        return cls_h
    
    def getConv(self, x):
        conv_out = self.conv_model(x)
        if DEBUG: print(f"conv_out shape: {conv_out.shape}")
        return self.conv_model(x)

    
    def forward(self, x,alpha=1):
        if DEBUG: print(f"input shape: {x.shape}")
        domain_cls_pred, domain_features = [],[]

        x = self.getConv(x)
        base_feat = self.getheadFeatures(x)
        cls_pred_x = self.getClassification(base_feat)

        if self.adv_training:
            domain_cls_pred, domain_features = self.domain_desc(base_feat, alpha)

        return cls_pred_x,base_feat,domain_cls_pred,domain_features