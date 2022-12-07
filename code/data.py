import torch

class Dataset(Dataset):
    def __init__(self, x, y):
        self.feature_size = x.shape[1]
        self.label_size = y.shape[1]
        self.len = x.shape[0]
        self.x_data = torch.as_tensor(x, dtype=torch.float)
        self.y_data = torch.as_tensor(y, dtype=torch.float)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len