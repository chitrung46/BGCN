from torch.utils.data import DataLoader

class SRDataloader:
    def __init__(self, args, dataset):
        super(SRDataloader, self).__init__()
        self.args = args
        self.dataset = dataset
        
    def get_data_loaders(self):
        train_loader = DataLoader(self.dataset['train'], batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(self.dataset['val'], batch_size=self.args.batch_size, shuffle=False) if dataset.get('val') is not None else None
        test_loader = DataLoader(self.dataset['test'], batch_size=self.args.batch_size, shuffle=False) if dataset.get('test') is not None else None
        return train_loader, val_loader, test_loader