from torch.utils.data import DataLoader
from lib.dataset import load_dataset

def get_data_loaders(args):
    data = load_dataset(args)
    train_data = dict()
    val_data = dict()
    test_data = dict()

    for uidx in data.keys():
        if len (data[uidx]) < 3:
            train_data[uidx] = data[uidx]
            val_data[uidx] = []
            test_data[uidx] = []
        else:
            train_data[uidx] = data[uidx][:-2]
            val_data[uidx] = [data[uidx][-2]]
            test_data[uidx] = [data[uidx][-1]]

    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False) if val_data is not None else test_data
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False) if test_data is not None else None
    
    return train_loader, val_loader, test_loader





