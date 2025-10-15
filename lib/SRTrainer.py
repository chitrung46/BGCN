from lib import BasicTrainer

class SRTrainner(BasicTrainer):
    def __init__(self, opt, model, train_loader, valid_loader=None, test_loader=None):
        super(SRTrainner, self).__init__(opt, model, train_loader, valid_loader, test_loader)

    def loss(self, batch):
        lr, hr = batch
        sr = self.model(lr)
        loss = self.criterion(sr, hr)
        return loss
