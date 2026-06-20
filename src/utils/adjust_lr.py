import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    # 根据训练进度递减学习速率
    # 这个也是深度学习固定的模板
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 范例 之后会改 收集下来一些好用的code
def freeze(model):
   for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:	
            param.requires_grad = False