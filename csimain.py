import torch
import torch.optim as optim
from utils import *
from algorithm import *
import os
import torchvision.transforms as transforms
import mytransforms
from model import *

def act_param_init(args):
    args.select_position = {'emg': [0]}
    args.select_channel = {'emg': np.arange(8)}
    args.hz_list = {'emg': 1000}
    args.act_people = {'emg': [[i*9+j for j in range(9)]for i in range(4)]}
    args.num_classes = 6

    return args

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--alpha', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--alpha1', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--batch_size', type=int,
                        default=32, help="batch_size")
    parser.add_argument('--beta1', type=float, default=0.5, help="Adam")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--dis_hidden', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--latent_domain_num', type=int, default=3)
    parser.add_argument('--local_epoch', type=int,
                        default=1, help='local iterations')
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--max_epoch', type=int,
                        default=50, help="max iterations")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args = act_param_init(args)
    return args

args = get_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = GeneFi(args).to(device)
params = net.parameters()
optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weight_decay, betas=(args.beta1, 0.9))
milestones = [10,20]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
global_train_acc = []
global_test_acc = []
img_transform = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.RandomCrop(224, padding=(32,0),padding_mode='reflect')],p=0.2),
        mytransforms.RandomSpi(p=0.2),
        mytransforms.RandomComPre(p=0.2),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
img_transformte = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

if __name__ == '__main__':

    trainer(net, img_transform, img_transformte, device, optimizer, scheduler, args.max_epoch)