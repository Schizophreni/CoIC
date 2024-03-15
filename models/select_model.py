"""
This file defines the model selecting strategy
"""

def define_model(opt):
    model_name = opt.model_name
    if model_name == 'DGUNet':
        from models.DGUNet_CoIM import DGUNet as M
    elif model_name == "RCDNet":
        from models.RCDNet_CoIM import RCDNet as M
    elif model_name == "BRN":
        from models.BRN_CoIM import BRN as M
    elif model_name == "IDT":
        from models.IDT_CoIM import IDT as M
    elif model_name == "DRSformer":
        from models.DRSformer_PA import DRSformer as M
    else:
        raise NotImplementedError
    
    m = M(opt)
    print('Training model [{}] is created.'.format(model_name))
    return m