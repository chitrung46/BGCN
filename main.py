from lib import BasicTrainer
from lib.dataloader import get_data_loaders
from lib.func import parse_args

args = parse_args()

train_loader, val_loader, test_loader = get_data_loaders(args)
model = None
print(f"Num of train_loader, val_loader, test_loader: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")

# trainer = BasicTrainer(args, 
#                        optimizer, 
#                        loss, 
#                        scaler, 
#                        model, 
#                        train_loader, 
#                        val_loader, 
#                        test_loader, 
#                        device)

# if args.phase == 'train':
#     trainer.train()
# elif args.phase == 'test':
#     trainer.test()
# else:
#     raise ValueError("Invalid phase. Choose 'train' or 'test'.")

