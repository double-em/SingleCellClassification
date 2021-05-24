from fastai.callback.core import Callback

class LearningRateScheduler(Callback):
    
    def __init__(self, lr_manipulator = None, threshold = 0.4):
        self.last_loss = 0
        self.threshold = threshold
        self.lr_manipulator = lr_manipulator
        
        self.lr_history = []
    
    def after_validate(self):
        if abs(self.last_loss - self.loss) < self.threshold:
            if self.lr_manipulator is not None:
                self.learn.lr = self.lr_manipulator(self.learn.lr)
                self.lr_history.append(self.learn.lr)

class ConsoleBar(Callback):
    
    def after_loss(self):
        if self.training:
            print(f"iter: {self.iter} n_iter: {self.n_iter}")
