from lib import BasicTrainer
from lib.dataloader import get_data_loaders
from lib.func import parse_args

args = parse_args()

train_loader, val_loader, test_loader = get_data_loaders(args)
model = None

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

