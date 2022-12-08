import numpy as np

class ModelSaving:
    def __init__(self, waiting=5, printing=True):
        self.patience = waiting
        self.printing = printing
        self.count = 0
        self.best = None
        self.save = False

    def __call__(self, validation_loss, model):
        if not self.best:
            self.best = -validation_loss
        elif self.best <= -validation_loss:
            self.best = -validation_loss
            self.count = 0
        elif self.best > -validation_loss:
            self.count += 1
            print(f'Validation loss has increased: {self.count} / {self.patience}.')
            if self.count >= self.patience:
                self.save = True
          
