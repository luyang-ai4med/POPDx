import torch
import torch.nn as nn

class POPDxModel(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDxModel, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size, y_emb.shape[1], bias=True)])

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
        x = torch.relu(x)  
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))        
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                
                
class LGclassifier(nn.Module):
    def __init__(self, feature_size, nlabel):
        super(LGclassifier, self).__init__()
        self.main = nn.Sequential(           
            nn.Linear(feature_size, nlabel)
        )

    def forward(self, input):
        return self.main(input)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

