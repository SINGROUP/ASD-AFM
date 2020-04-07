
import os
import sys
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import Sequence
from keras.callbacks import Callback

def _calc_plot_dim(n, f=0.3):
    rows = max(int(np.sqrt(n) - f), 1)
    cols = 1
    while rows*cols < n:
        cols += 1
    return rows, cols

class DataGenerator(Sequence):

    def __init__(self, data_path):
        self.batch_ids = glob.glob(os.path.join(data_path, 'batch_*.npz'))

    def apply_preprocessing(self, batch):

        X, Y = batch
        
        add_norm(X)
        add_noise(X, c=0.1)
        rand_shift_xy(X, c=0.02)
        add_cutout(X, n_holes=5)
        
        Y = [Y[:,:,:,i] for i in range(Y.shape[-1])]
        minimum_to_zero(Y)

        return X,Y

    def load_batch(self, index):
        file_path = os.path.join(self.batch_ids[index])
        batch_object = np.load(file_path)
        batch_data = (batch_object['arr_0'], batch_object['arr_1'])
        return batch_data

    def __len__(self):
        return len(self.batch_ids)

    def __getitem__(self, index):
        batch = self.load_batch(index)
        return self.apply_preprocessing(batch)
        
class HistoryPlotter(Callback):

    def __init__(self, log_path, plot_path, loss_labels):
        self.log_path = log_path
        self.plot_path = plot_path
        self.loss_labels = ['Total_weighted'] + loss_labels
        self.read_log()
        super(HistoryPlotter, self).__init__()

    def read_log(self):
        self.losses = []
        self.val_losses = []
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                f.readline()
                for line in f:
                    line = line.split(',')
                    lt = []
                    lv = []
                    for i in range(len(self.loss_labels)):
                        lt.append(float(line[1+i]))
                        lv.append(float(line[1+len(self.loss_labels)+i]))
                    self.losses.append(lt)
                    self.val_losses.append(lv)

    def on_epoch_end(self, epoch, logs):
        lt = [logs['loss']]
        lv = [logs['val_loss']]
        for label in self.loss_labels[1:]:
            lt.append(logs[label+'_loss'])
            lv.append(logs['val_'+label+'_loss'])
        self.losses.append(lt)
        self.val_losses.append(lv)
        self.plot()

    def plot(self, show=False):
        x = range(1, len(self.losses)+1)
        n_rows, n_cols = _calc_plot_dim(len(self.loss_labels), f=0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4*n_rows))
        for i, (label, ax) in enumerate(zip(self.loss_labels, axes.flatten())):
            ax.semilogy(x, np.array(self.losses)[:,i])
            ax.semilogy(x, np.array(self.val_losses)[:,i])
            ax.legend(['Training', 'Validation'])
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title(label)
        fig.tight_layout()
        plt.savefig(self.plot_path)
        if show:
            plt.show()
        else:
            plt.close()

def make_prediction_plots(preds, true=None, losses=None, descriptors=None, outdir='./predictions/', start_ind=0, verbose=1):
    
    if true is None:
        rows = 1
    else:
        rows = 2
        if not isinstance(true, list):
            true = [true]
    if not isinstance(preds, list):
        preds = [preds]
    if descriptors is not None:
        if len(descriptors) != len(preds):
            raise ValueError('len(descriptors) = %d and len(preds) = %d do not match' % (len(descriptors), len(preds)))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cols = len(preds)
    
    if losses is not None and losses.ndim < 1:
        losses = np.expand_dims(losses, axis=0)

    img_ind = start_ind
    for i in range(preds[0].shape[0]):
        
        fig, axes = plt.subplots(rows, cols)
        fig.set_size_inches(6*cols, 5*rows)

        if cols == 1:
            axes = np.expand_dims(axes, axis=1)

        if losses is not None and losses.ndim < 2:
            losses = np.expand_dims(losses, axis=1)

        for j in range(cols):
            
            p = preds[j][i]
            if true is not None:
                t = true[j][i]
                ax = axes[:, j]
                vmax = np.concatenate([p,t]).flatten().max()
                vmin = np.concatenate([p,t]).flatten().min()
            else:
                ax = [axes[j]]
                vmax = p.flatten().max()
                vmin = p.flatten().min()
                
            title1 = ''
            title2 = ''
            cmap = cm.viridis
            if descriptors is not None:
                descriptor = descriptors[j]
                title1 += descriptor+' Prediction'
                title2 += descriptor+' Reference'
                if descriptor == 'ES':
                    vmax = max(abs(vmax), abs(vmin))
                    vmin = -vmax
                    cmap = cm.coolwarm
            if losses is not None:
                title1 += '\nMSE = '+'{:.2E}'.format(losses[i,j])

            im1 = ax[0].imshow(p, vmax=vmax, vmin=vmin, cmap=cmap, origin='lower')
            if true is not None:
                im2 = ax[1].imshow(t, vmax=vmax, vmin=vmin, cmap=cmap, origin='lower')

            if title1 != '':
                ax[0].set_title(title1)
                if true:
                    ax[1].set_title(title2)

            for axi in ax:
                pos = axi.get_position()
                pos_new = [pos.x0, pos.y0, 0.8*(pos.x1-pos.x0), pos.y1-pos.y0]
                axi.set_position(pos_new)
            
            pos1 = ax[0].get_position()
            if true is not None:
                pos2 = ax[1].get_position()
                c_pos = [pos1.x1+0.1*(pos1.x1-pos1.x0), pos2.y0, 0.08*(pos1.x1-pos1.x0), pos1.y1-pos2.y0]
            else:
                c_pos = [pos1.x1+0.1*(pos1.x1-pos1.x0), pos1.y0, 0.08*(pos1.x1-pos1.x0), pos1.y1-pos1.y0]
            cbar_ax = fig.add_axes(c_pos)
            fig.colorbar(im1, cax=cbar_ax)

        save_name = outdir+str(img_ind)+'_pred.png'
        plt.savefig(save_name)
        plt.close()
        
        if verbose > 0: print('Prediction saved to '+save_name)
        img_ind += 1

def make_input_plots(Xs, outdir='./predictions/', start_ind=0, constant_range=True, cmap=cm.viridis, verbose=1):

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not isinstance(Xs, list):
        Xs = [Xs]

    img_ind = start_ind
    for i in range(Xs[0].shape[0]):

        for j in range(len(Xs)):

            x = Xs[j][i]
            rows, cols = _calc_plot_dim(x.shape[-1])
            fig = plt.figure(figsize=(3.2*cols,2.5*rows))
            
            vmax = x.max()
            vmin = x.min()
            for k in range(x.shape[-1]): 
                fig.add_subplot(rows,cols,k+1)
                if constant_range:
                    plt.imshow(x[:,:,k], cmap = cmap, vmin=vmin, vmax=vmax, origin="lower")
                else:
                    plt.imshow(x[:,:,k], cmap = cmap, origin="lower")
                plt.colorbar()
         
            save_name = outdir+str(img_ind)+'_input'
            if len(Xs) > 1:
                save_name += str(j+1)
            save_name += '.png'
            plt.savefig(save_name)
            plt.close()

            if verbose > 0: print('Input image saved to '+save_name)

        img_ind += 1

def calculate_losses(model, true, preds=None, X=None):

    import keras.backend as K

    if preds is None and X is None:
        raise ValueError('preds and X cannot both be None')
    
    if preds is None:
        preds = model.predict_on_batch(X)

    if not isinstance(true, list):
        true = [true]
    if not isinstance(preds, list):
        preds = [preds]
    
    losses = np.zeros((true[0].shape[0], len(true)))
    for i, (t, p) in enumerate(zip(true, preds)):
        t = K.variable(t)
        p = K.variable(p) 
        loss = model.loss_functions[i](t, p)
        sh = loss.shape.as_list()
        if len(sh) > 1:
            loss = K.mean(K.reshape(loss, (sh[0],-1)), axis=1)
        losses[:,i] = K.eval(loss)

    if losses.shape[1] == 1:
        losses = losses[:,0]
    if losses.shape[0] == 1 and losses.ndim == 1:
        losses = losses[0]
    
    return losses

def minimum_to_zero(Y_):
    if isinstance(Y_, list):
        Ys = Y_
    else:
        Ys = [Y_]
    for Y in Ys:
        sh = Y.shape
        for j in range(sh[0]):
            Y[j,:,] = Y[j,:,] - np.amin(Y[j,:,])

def add_noise(X_, c=0.1 ):
    if isinstance(X_, list):
        Xs = X_
    else:
        Xs = [X_]
    for X in Xs:
        sh = X.shape
        R = np.random.rand( sh[0], sh[1], sh[2], sh[3] ) - 0.5
        for j in range(sh[0]):
            for i in range(sh[3]):
                vmin = X[j,:,:,i].min()
                vmax = X[j,:,:,i].max()
                X[j,:,:,i] += R[j,:,:,i] * c*(vmax-vmin)

def add_norm(X_):
    if isinstance(X_, list):
        Xs = X_
    else:
        Xs = [X_]
    for X in Xs:
        sh = X.shape
        for j in range(sh[0]):
            for i in range(sh[3]):
                mean=np.mean(X[j,:,:,i])            
                sigma=np.std(X[j,:,:,i])          
                X[j,:,:,i]-= mean
                X[j,:,:,i]= X[j,:,:,i]/ sigma

def rand_shift_xy(X_, c=0.02):
    # c= percantage shift acording to size of image in pixels. c=0.05 ~ 5 %
    if isinstance(X_, list):
        Xs = X_
    else:
        Xs = [X_]
    for X in Xs:
        sh= X.shape
        max_y_shift=np.floor(sh[1]*c).astype(int)
        max_x_shift=np.floor(sh[2]*c).astype(int)
        for j in range(sh[0]):
            for i in range(sh[3]):    
                rand_shift_y=random.choice(np.append(np.arange(-max_y_shift,0), np.arange(1,max_y_shift+1)))   
                rand_shift_x= random.choice(np.append(np.arange(-max_x_shift,0), np.arange(1,max_x_shift+1)))  
                shift_y=abs(rand_shift_y)            
                shift_x=abs(rand_shift_x)
                a=X[j,:,:,i]
                tmp=np.zeros((sh[1]+2*shift_y,sh[2]+2*shift_x))
                tmp[shift_y:-shift_y,shift_x:-shift_x]=a
                tmp[:shift_y,shift_x:-shift_x]=a[shift_y:0:-1,:]
                tmp[-shift_y:,shift_x:-shift_x]=a[-2:-2-shift_y:-1,:]
                tmp[:,-shift_x:]=tmp[:,-2-shift_x:-2-2*shift_x:-1]
                tmp[:,:shift_x]=tmp[:,2*shift_x:shift_x:-1]
                X[j,:,:,i]=tmp[shift_y-rand_shift_y:shift_y-rand_shift_y+sh[1],shift_x-rand_shift_x:shift_x-rand_shift_x+sh[2] ] 

def add_cutout(X_, n_holes=5):
    def  get_random_eraser(input_img,p=0.2, s_l=0.001, s_h=0.01, r_1=0.1, r_2=1/0.1, v_l=0, v_h=0):
        '''        
        p : the probability that random erasing is performed
        s_l, s_h : minimum / maximum proportion of erased area against input image
        r_1, r_2 : minimum / maximum aspect ratio of erased area
        v_l, v_h : minimum / maximum value for erased area
        '''

        sh = input_img.shape
        img_h, img_w= [sh[0],sh[1]] 
        
        if np.random.uniform(0, 1) > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = 0.0

        return input_img

    if isinstance(X_, list):
        Xs = X_
    else:
        Xs = [X_]
    for X in Xs:
        sh = X.shape
        for j in range(sh[0]):
            for i in range(sh[3]):
                for attempt in range(n_holes):
                    X[j,:,:,i]=get_random_eraser(X[j,:,:,i])



