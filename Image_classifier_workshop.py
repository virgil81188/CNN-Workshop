import numpy as np
import torch
from torch import nn
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import onnx
import onnxruntime as onnxrt
import torchvision

class ImagesDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.examples = []
        self.classes = []
    def load(self, path : str = ""):
        print("Loading dataset...", end="", flush = True)
        if not os.path.isdir(path):
            print("Failed! Invalid path!")
            return
        #create classes list from folders in dataset
        self.classes = ['other']
        for root, dirs, files in os.walk(path):
            if len(dirs) == 0:
                self.classes.append(os.path.split(root)[1])
        #make sure we don't depend on file structure traversal order
        self.classes = self.classes[0:1] + sorted(self.classes[1:])

        #load the actual dataset
        for root, dirs, files in os.walk(path):
            if len(dirs) == 0:
                for filename in files:
                    e = {'path':os.path.join(root,filename), 'x':cv2.imread(os.path.join(root,filename), flags=cv2.IMREAD_COLOR), 'y':self.classes.index(os.path.split(root)[1])}
                    self.examples.append(e)
        print("Done! Classes = " + str(self.classes))
            
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index : int):
        #we switch from the openCV HWC format to the torch CHW format
        #also perform some additional preprocessing for normalizing the data (optional, but usually leads to better results)
        X = np.transpose(self.examples[index]['x'], (2,0,1)).astype(np.float32)/255 - 0.5
        Y = self.examples[index]['y']
        return {'x' : X, 'y' : Y, 'path':self.examples[index]['path']}

class ClasifierNet(nn.Module):
    def __init__(self, nr_classes : int = 2):
        '''
        Number of output classes
        '''
        super().__init__()
        # Key points when designing a conv net:
        #  - receptive field (stride, dilated conv, conv kernel size)
        #  - depth (padding, stride)
        #  - model_size (depth, conv kernel size)
        #  - capacity (depth, conv kernel size)
        #  - gradient flow

        self.nr_classes = nr_classes
        self.conv_layers = nn.Sequential(
                                #make some convs here
                                #...
                                #...
                                #make a final conv that can be used further to predict one number for each output class
                                #Hint: the input image size varies, so here you should be smart about the number of channels
                                #...
        #make a final computation that outputs exactly 'nr_classes' real numbers. We call these logits
        #Hint: the input image size varies, it's time to kill the spatial dimensions
        #self.logits = ...
        #self.compute_probs = ...

    def forward(self, x):
        #logits = ...
        #make sure we have the right dimensions going forward
        logits = logits.view((-1, self.nr_classes))
        #compute class probabilities
        #class_probabilities = ...
        return class_probabilities, logits

class Trainer(object):
    def __init__(self):
        #just some logging and ploting stuff
        self.log = {'L':np.zeros((0,2)), 'dt':np.zeros((0)), 'train_acc':np.zeros((0,2)), 'dev_acc':np.zeros((0,2))}
        plt.ion()
        self.figure = plt.figure(figsize=(16,8))
        fig_gridspec = self.figure.add_gridspec(4,2)
        self.L_axes = self.figure.add_subplot(fig_gridspec[0:3,0])
        self.acc_axes = self.figure.add_subplot(fig_gridspec[0:3,1])
        self.dt_axes = self.figure.add_subplot(fig_gridspec[3,:])
        self.L_plot, = self.L_axes.plot(self.log['L'][:,0], self.log['L'][:,1], color='orange', label="Loss")
        self.dt_plot, = self.dt_axes.plot(self.log['dt'][:], self.log['dt'][:], color='blue', label="step duration (ms)")
        self.train_acc_plot, = self.acc_axes.plot(self.log['train_acc'][:,0], self.log['train_acc'][:,1], color='orange', label="train accuracy")
        self.dev_acc_plot, = self.acc_axes.plot(self.log['dev_acc'][:,0], self.log['dev_acc'][:,1], color='red', label="dev accuracy")
        self.L_axes.legend()
        self.L_axes.grid(True)
        self.acc_axes.legend()
        self.acc_axes.grid(True)
        self.dt_axes.legend()
        self.dt_axes.grid(True)
        plt.show()
    
    def train(self, net : torch.nn.Module, ds_train : Dataset, ds_dev : Dataset, steps : int = 10000, device = None):
        '''
        Trains the given net on the ds_train dataset for the number os steps given
        Evaluation is performed on the ds_dev dataset.
        '''

        #choose GPU if available, otherwise default to CPU
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #make dataset iterators for train set and dev set
        dl_train = DataLoader(ds_train, shuffle=True)
        dl_dev = DataLoader(ds_dev)
        
        #create optimizer, used later for making an update step to the model
        #optimizer = ...

        current_step = 0
        dt = 0
        while current_step < steps:
            correct = 0
            avg_loss = 0
            avg_dt = 0
            #train one epoch
            #put network in training mode
            net.train()
            for e in dl_train:
                frame_clock = time.perf_counter()
                if current_step >= steps:
                    break
                # prepare training example: get features X and targets Y
                X = e['x'].to(device) # input features
                Y = e['y'].to(device) # true label
                #forward pass
                #...
                #compute loss
                #L = ...
                #compute gradients using backprop
                #...
                #...
                #use gradients to make a small change in the model... hopefully for the better
                #...
                
                #compute some metrics: accuracy (= nr_correct_predictions / nr_predictions), average loss, average train step time
                #compute prediction
                predicted_class = int(np.argmax(prediction.detach().cpu()[0,:]))
                #add up correct predictions for accuracy computations
                correct += int(predicted_class == int(e['y'][0]))
                #add up the loss of this example to the total loss of the epoch
                avg_loss += L.item()
                #compute the duration of one train step and add it up to the total epoch time
                dt = time.perf_counter() - frame_clock
                frame_clock = time.perf_counter()
                avg_dt += dt

                #next step
                current_step += 1

            #compute averages from accumulated values
            train_acc = correct / len(ds_train)
            avg_dt = avg_dt / len(ds_train)
            avg_loss = avg_loss / len(ds_train)

            #eval dev accuracy
            #put network in eval mode
            net.eval()
            correct = 0
            for e in dl_dev:
                #get data
                X = e['x'].to(device)
                Y = e['y'].to(device)
                #make inference
                #...
                #compute predicted class
                predicted_class = int(np.argmax(prediction.detach().cpu()[0,:]))
                #accumulate correct results for accuracy computation
                correct += int(predicted_class == int(e['y'][0]))

            dev_acc = correct / len(ds_dev)

            # log stuff
            self.log['L'] = np.concatenate([self.log['L'], np.array([(current_step-1, avg_loss)])], axis=0)
            self.log['dt'] = np.concatenate([self.log['dt'], [avg_dt * 1000]])
            self.log['train_acc'] = np.concatenate([self.log['train_acc'], np.array([(current_step, train_acc)])], axis=0)
            self.log['dev_acc'] = np.concatenate([self.log['dev_acc'], np.array([(current_step, dev_acc)])], axis=0)
            self.draw_logs()
            print("Step = {0:10}   Train accuracy = {1:.4f}    Dev accuracy = {2:.4f}   avg Loss = {3:.7f}".format(current_step, train_acc, dev_acc, avg_loss))

    def draw_logs(self):
        #update dt data and plot limits
        self.dt_plot.set_xdata(np.arange(0, len(self.log['dt'][:])))
        self.dt_plot.set_ydata(self.log['dt'][:])
        self.dt_axes.set_xlim([0, len(self.log['dt'][:])])
        self.dt_axes.set_ylim([0, 2.0 * np.percentile(self.log['dt'][:], 95)]) #don't care about spikes

        #update both train and dev accuracy data and plot limits
        if len(self.log['train_acc'][:,0]) > 0:
            self.train_acc_plot.set_xdata(self.log['train_acc'][:,0])
            self.train_acc_plot.set_ydata(self.log['train_acc'][:,1])
            self.dev_acc_plot.set_xdata(self.log['dev_acc'][:,0])
            self.dev_acc_plot.set_ydata(self.log['dev_acc'][:,1])
            self.acc_axes.set_xlim([0, np.max([np.max(self.log['train_acc'][:,0]), np.max(self.log['dev_acc'][:,0])])])
            self.acc_axes.set_ylim([-0.1, 1.1])

        #update loss data and plot limits
        self.L_plot.set_xdata(self.log['L'][:,0])
        self.L_plot.set_ydata(self.log['L'][:,1])
        self.L_axes.set_xlim([0,np.max(self.log['L'][:,0])])
        self.L_axes.set_ylim([0,1.5 * np.percentile(self.log['L'][:,1], 99)])

        self.figure.canvas.draw()
        plt.pause(0.001) #allow a pass through the window event queue so that the window remains responsive
                
    

if __name__=="__main__":
    ds_train = ImagesDataset()
    ds_train.load(r".\dataset\train")

    ds_dev = ImagesDataset()
    ds_dev.load(r".\dataset\dev")

    net = ClasifierNet(len(ds_train.classes))
    print(net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    trainer = Trainer()
    trainer.train(net, ds_train, ds_dev, steps = 10000, device=device)