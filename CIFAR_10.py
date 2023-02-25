from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy
from robustbench.utils import load_model
from torchsummary import summary
from torch import nn
from autoattack import AutoAttack
import foolbox as fb
import torch

corruptions = ['fog']
x_test, y_test = load_cifar10c(n_examples=1000, corruptions=corruptions, severity=5)

for model_name in ['Standard', 'Engstrom2019Robustness', 'Rice2020Overfitting',
                   'Carmon2019Unlabeled', 'Andriushchenko2020Understanding']:
 model = load_model(model_name, dataset='cifar10', threat_model='Linf')
 num_params = sum(p.numel() for p in model.parameters())
 num_layers = len([l for l in model.modules() if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear)])
 summary(model, (3, 32, 32), device='cpu')
 acc = clean_accuracy(model, x_test, y_test)
 print(f'Model: {model_name}, CIFAR-10-C accuracy: {acc:.1%}, Num params: {num_params}, Num layers: {num_layers}')

 torch.cuda.empty_cache()
 
 batch_size = 10
 n_batches = len(x_test) // batch_size

 model = load_model(model_name, dataset='cifar10', threat_model='Linf')
 fmodel = fb.PyTorchModel(model, bounds=(0, 1))

 success_list = []
 for i in range(n_batches):
     x_batch = x_test[i*batch_size:(i+1)*batch_size].to('cuda:0')
     y_batch = y_test[i*batch_size:(i+1)*batch_size].to('cuda:0')
    
     _, advs, success = fb.attacks.LinfPGD()(fmodel, x_batch, y_batch, epsilons=[8/255])
     success_list.append(success.float().mean())
     torch.cuda.empty_cache()

    
 print('Robust accuracy: {:.1%}'.format(1 - torch.stack(success_list).mean()))
