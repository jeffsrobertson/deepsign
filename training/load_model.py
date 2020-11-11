import os
from torch import nn
import torch
from models.mobilenetv2 import MobileNetV2


def generate_model(name, num_classes, sample_size, width_mult=1.0, use_cuda=True):
    print('>> Creating model architecture for {}'.format(name))
    if name.lower() == 'mobilenetv2':
        model = MobileNetV2(num_classes=num_classes, sample_size=sample_size, width_mult=width_mult)
    else:
        raise ValueError("Haven't implemented other models yet in generate_model()")
    print('>> Successfully created model architecture for {}.'.format(name))
    return model
    
def mount_to_gpu(model, default_device="cuda:0"):
    if not torch.cuda.is_available():
        raise ValueError("use_cuda was specified in generate_model(), but no cuda device was found.")
    
    device = torch.device(default_device)
    model = model.to(device)
    print(">> Loaded blank model onto the following device: {}".format(str(device)))
    model = nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if
                       p.requires_grad)
    print(">> Total number of trainable parameters: ", pytorch_total_params)
    
    return model

def load_pretrained_weights(model, weights_path):
    
    if not os.path.exists(weights_path):
        cwd = os.getcwd()
        weights_path = os.path.join(cwd, weights_path)
        if not os.path.exists(weights_path):
            raise ValueError("The following path could not be found: {}".format(weights_path))
    
    print('>> Loading pretrained weights from: {}'.format(weights_path))
    if torch.cuda.is_available():
        pretrained_weights = torch.load(weights_path)
    else:
        pretrained_weights = torch.load(weights_path, map_device="cpu")
    model.load_state_dict(pretrained_weights['state_dict'])
    
    return model

class AttrDict(dict):
    """Allows elements of dictionary to be accessed like attributes.
    
        Example:
            dict = {'a':1, 'b':2}
            dict = AttrDict(dict)
            
            print(dict.b)
            >> 2
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
#root_path = "/home/ec2-user/ASL_transfer_learner"
#'pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth'
def load_model(name='mobilenetv2', num_classes=27, sample_size=112, width_mult=1.0, pretrain_path=None, use_cuda=False, transfer_learning=None):
    
    # Load blank model architecture
    if pretrain_path is not None:
        model = generate_model(name, 27, sample_size, width_mult, use_cuda=True)
    else:
        model = generate_model(name, num_classes, sample_size, width_mult, use_cuda=True)
    
    if use_cuda:
        model = mount_to_gpu(model)
    
    if pretrain_path is not None:
        model = load_pretrained_weights(model, pretrain_path)
        
    if transfer_learning is not None:
        assert transfer_learning in ['last_layer', 'fine_tune']
        
        if transfer_learning == 'last_layer':
            # Freeze all layers except last
            for param in model.parameters():
                param.requires_grad = False
            
            new_classifier = torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
            model.module.classifier = new_classifier.cuda()
            print('>> Froze all weights except last layer, which now has {} neurons.'.format(num_classes))
        elif transfer_learning == 'fine_tune':
            # Unfreeze all layers, but also replace last
            new_classifier = torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
            model.module.classifier = new_classifier.cuda()
            print('>> Replaced last layer of neural network, which now has {} neurons.'.format(num_classes))
        else:
            print('>> Did not recognize method of transfer learning requested')
    
    return model

