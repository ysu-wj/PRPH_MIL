import torch.optim.lr_scheduler as lr_scheduler

def define_scheduler(args, optimizer):
    if args.scheduler == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True)
    elif args.scheduler == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=0)
    elif args.scheduler == 'None':
        return None
    else:
        raise NotImplementedError(f'Scheduler [{args.scheduler}] is not implemented')