import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
# the seNet154 model
class SEResnext50(nn.Module):
    def __init__(self, num_classes):
        super(SEResnext50, self).__init__()
        self.output_dim = num_classes # used for knowing output size in evaluation -- Must have on all models
        self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        
        # change the classification layer
        self.l0= nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout2d(0.4)
        
    def forward(self, x):
        # get the batch size only, ignore(c, h, w)
        batch = x.shape[0]
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0