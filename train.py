# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="TIFFSetField: tempfile.tif:\
 Unknown pseudo-tag 65538.")
# ours
#
# from load_data import SintelDataset
# from load_data_mb import MotBlurDataset
# from load_data_mb_patch import MotBlurDataset_ptch

LR0 = 1e-7

# def train(net, optimizer, train_data_loader, test_data_loader,device,num_epochs=10,first_epoch=0):
def train(net, train_data_loader, test_data_loader,device, args):
    net.to(device)
    batch_size = train_data_loader.batch_size

    optimizer = optim.Adam(net.parameters(), lr=1e-6)
    # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.000)

    #criterion = nn.L1Loss()
    criterion = nn.SmoothL1Loss()

    train_epochs_loss = []
    val_epochs_loss = []

    if args.load_model:
        load_model(net, optimizer, train_epochs_loss, val_epochs_loss, device, epoch=args.epoch2continue,
                   experiment_name=get_args().experiment_name)

    num_epochs = args.epochs
    first_epoch = args.epoch2continue+1

    optimizer.zero_grad()

    net.zero_grad()

    #train_steps_per_e = len(train_data_loader.dataset) // batch_size
    #val_steps_per_e   = len(test_data_loader.dataset) // batch_size

    train_steps_per_e = np.ceil(len(train_data_loader.dataset) / batch_size)
    val_steps_per_e = np.ceil(len(test_data_loader.dataset) / batch_size)

    prev_loss = 1e5
    for e in range(first_epoch,num_epochs):
        print ("Epoch: ", e)
        val_loss_sum = 0.
        train_loss_sum = 0
        net.train()
        for i, data in enumerate(train_data_loader):
            x,y = data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                print('Step: {:3} / {:3} Train loss: {:3.3}'.format(i, train_steps_per_e,loss.item()))
            train_loss_sum += loss.item()
        train_epochs_loss += [train_loss_sum / train_steps_per_e]

        net.eval()
        for i, val_data in enumerate(test_data_loader):

            x,y = val_data

            x,y = x.to(device), y.to(device)

            with torch.no_grad():
                out = net(x)

            loss = criterion(out, y)
            if i%10 == 0:
                print('Step: {:3} / {:3} Val. loss: {:3.3}'.format(i, val_steps_per_e,loss.item()))
            val_loss_sum += loss.item()
        val_epochs_loss += [val_loss_sum / val_steps_per_e]

        if True: #val_epochs_loss[-1] < prev_loss:
            print ("Saving Model")
            save_model(net, optimizer, train_epochs_loss, val_epochs_loss, epoch=e, experiment_name=get_args().experiment_name)
        prev_loss = val_epochs_loss[-1]
        print("\nepoch {:3} Train Loss: {:1.5} Val loss: {:1.5}".format(e, train_epochs_loss[-1],
                                                                        val_epochs_loss[-1]))
        _plot_fig(train_epochs_loss,val_epochs_loss,'Current loss')
    return train_epochs_loss, val_epochs_loss


def _plot_fig(train_loss, val_loss, title):
    plt.title(title)
    plt.cla()
    plt.xlabel('epoch num'), plt.ylabel('loss')
    plt.plot(train_loss, 'bo-', label="Train Loss")
    plt.plot(val_loss, 'ro-', label="Val Loss")
    plt.legend()
    file_name = title + '.png'
    plt.savefig(file_name)
    #plt.show()
    #plt.draw()
    #plt.pause(0.01)



def get_model_name(model):
    return model.__class__.__name__


def model_file_name(epoch):
    return 'checkpoint_'+str(epoch)+'.pth.tar'


def save_model(model, optimizer, train_epochs_loss, val_epochs_loss, epoch, experiment_name):
    model_name = get_model_name(model) + '_' +experiment_name
    save_dir = os.path.join('./trained_models', model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save model
    #model_dict = {'state_dict': model.state_dict()}
    filename = model_file_name(epoch)

    filename = os.path.join(save_dir, filename)
    #torch.save(model_dict, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
       # 'loss': loss,
    }, filename)

    # Save losses
    lossFileName = os.path.join(save_dir, "loss_"+str(epoch)+".out")
    np.savetxt(lossFileName, (train_epochs_loss, val_epochs_loss), delimiter=',')



def load_model(model, optimizer, train_loss, val_loss, device, epoch, experiment_name):
    print("loading checkpoint")
    model_name = get_model_name(model) + '_' +experiment_name
    filename = model_file_name(epoch)
    model_path = os.path.join('./trained_models', model_name, filename)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    lossFileName = os.path.join('./trained_models', model_name, "loss_" + str(epoch) + ".out")
    temp = np.loadtxt(lossFileName, delimiter=',')  # X is an array
    train_loss += list(temp[0,:])
    val_loss += list(temp[1, :])


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
    parser.add_argument('--load_model', action='stroe_true', default=False,
                        help='load trained model')
    parser.add_argument('--train', action='stroe_true', default=True,
                        help='train model')
    parser.add_argument('--epochs', type=int, default=300,
                        help='num of epochs')
    parser.add_argument('--epoch2continue', type=int, default=-1,
                        help='epoch2continue')
    parser.add_argument('--experiment_name', type=str, default='default',
                        help='exp name')
    parser.add_argument('--machine', type=str, default='local',
                        help='machine name')
    parser.add_argument('--inpMode', type=str, default='full',
                        help='machine name')
    parser.add_argument('--data_root', type=str, default='',
                        help='root directory of the data')
    parser.add_argument('--data_suf', type=str, default='',
                        help='data suffix')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--model', default='EDOF', const='EDOF', nargs='?', choices=['EDOF', 'Unet'],
                        help='model architecture (default: %(default)s)')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.machine == 'local':
        root_dir = r'D:\Datasets\GoPro\motBlurMaskDataset_2'
        suf = '_tiny'
        # suf = '_small'

    elif args.machine == 'AWS':
        root_dir = r'/home/ubuntu/EDOF_Data/'
        suf = ''
    else:
        root_dir = args.root_dir
        suf = args.data_suf

    batch_size = 50

    if args.inpMode == 'full':
        train_dir = os.path.join(root_dir, 'train'+suf)
        test_dir = os.path.join(root_dir, 'test'+suf)
        label_dir = os.path.join(root_dir, 'labels')
        train_dataset = MotBlurDataset(filtered_image_path=train_dir, clean_image_path=label_dir, train=True)
        test_dataset = MotBlurDataset(filtered_image_path=test_dir,  clean_image_path=label_dir, train=False)

    elif args.inpMode == 'patch':
        train_dir = os.path.join(root_dir,'trainP')
        test_dir = os.path.join(root_dir, 'testP')
        label_dir = os.path.join(root_dir, 'labelsP')
        train_dataset = MotBlurDataset_ptch(filtered_image_path=train_dir, clean_image_path=label_dir, train=True)
        test_dataset = MotBlurDataset_ptch(filtered_image_path=test_dir, clean_image_path=label_dir, train=False)

    else:
        raise Exception('Unknown input mode')

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=0)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if args.model == 'EDOF':
        from models import EdofNet
        net = EdofNet(max_dilation=7, device=device)
    elif args.model == 'UNet':
        from unet_model import UNet
        net = UNet(n_channels=3, n_classes=3)

    if args.train:
        #train_loss, val_loss = train(net=net, optimizer = optimizer, train_data_loader=train_data_loader, test_data_loader=test_data_loader, device=device, num_epochs=args.epochs,first_epoch = epoch2continue+1)
        train_loss, val_loss = train(net=net, train_data_loader=train_data_loader, test_data_loader=test_data_loader,
                                     device=device, args=args)
        _plot_fig(train_loss, val_loss, args.model+'-Losses')
        save_model(net, epoch=args.epochs, name=args.experiment_name)

if __name__ == '__main__':
    main()
