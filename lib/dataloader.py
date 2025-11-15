from torch.utils.data import DataLoader
from lib.dataset import SRDataset, load_dataset

def leave_one_out_strategy(data, masks):
    train_data = dict()
    val_data = dict()
    test_data = dict()
    train_mask = dict()
    val_mask = dict()
    test_mask = dict()

    for uidx in data.keys():
        if len(data[uidx]) < 3:
            train_data[uidx] = data[uidx]
            val_data[uidx] = []
            test_data[uidx] = []
            train_mask[uidx] = masks[uidx]
            val_mask[uidx] = []
            test_mask[uidx] = []

        else:
            train_data[uidx] = data[uidx][:-2]
            val_data[uidx] = data[uidx][:-1]
            test_data[uidx] = data[uidx]
            train_mask[uidx] = masks[uidx][:-2]
            val_mask[uidx] = masks[uidx][:-1]
            test_mask[uidx] = masks[uidx]

    return train_data, val_data, test_data, train_mask, val_mask, test_mask

def get_data_loaders(args):
    data, masks, global_graph, item_num = load_dataset(args)
    train_data, val_data, test_data, train_mask, val_mask, test_mask = leave_one_out_strategy(data, masks)
    train_dataset = SRDataset(train_data, train_mask)
    val_dataset = SRDataset(val_data, val_mask)
    test_dataset = SRDataset(test_data, test_mask)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else test_dataset
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) if test_dataset is not None else None
    
    return train_loader, val_loader, test_loader, global_graph, item_num





