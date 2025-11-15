from lib import BasicTrainer
from lib.dataloader import get_data_loaders
from lib.func import parse_args
from model.BGCN import BGCN

args = parse_args()

train_loader, val_loader, test_loader = get_data_loaders(args)
model = BGCN(args)

trainer = BasicTrainer(args, 
                       optimizer=None, 
                       loss=None, 
                       scaler=None, 
                       model=model, 
                       train_loader=train_loader, 
                       val_loader=val_loader, 
                       test_loader=test_loader, 
                       device=args.device)

if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    trainer.test()
else:
    raise ValueError("Invalid mode. Choose 'train' or 'test'.")

