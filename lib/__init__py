from .SRTrainer import SRTrainer

TRAINERS = {
    SRTrainer.code(): SRTrainer
}

def trainer_factory(args, model, train_loader, val_loader, test_loader):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader)