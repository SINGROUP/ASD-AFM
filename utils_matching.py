#!/usr/bin/python
# coding: utf-8
# This script is continue of error_mix.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec 
#import scipy
import scipy.ndimage as nimg
import scipy.misc 
from scipy import misc
import time
import os
import pandas as pd
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
import numpy as np

import sys


def correlationCoef( im1, im2 ):
    im1 = im1 - np.mean(im1);
    im2 = im2 - np.mean(im2);
    cs=np.sum(im1*im2)/np.sqrt( np.sum(im1**2) * np.sum(im2**2) )
    #cs*=cs; cs*=cs
    return cs 

def image_center(img):
    #finds center of mass for image
    (xa,ya)= nimg.measurements.center_of_mass(img)
    cy = np.rint(xa)
    cx = np.rint(ya)
    return cx,cy


def add_norm(X):
    # normalize and than scale to [-1,1]
    sh = X.shape
    for j in range(sh[0]):
        for i in range(sh[3]):
            mean=np.mean(X[j,:,:,i])         
            #print 'mean='+str(mean)+ 'sigma='+str(sigma)        
            sigma=np.std(X[j,:,:,i])      
            X[j,:,:,i]-= mean
            X[j,:,:,i]= X[j,:,:,i]/ sigma            
            # Then scale to [-1,1]
            tmp = np.absolute(X[j,:,:,i])
            vmax=tmp.max()
            if vmax>0:
                X[j,:,:,i] = X[j,:,:,i] / vmax

def crosscorel_2d_fft(im0,im1):
    #calculates cross correlation between two images
    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)
    renorm = 1/( np.std(f0)*np.std(f1) )
    return abs(np.fft.ifft2( f0 * f1.conjugate() ) ) * renorm

def trans_match_fft(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    ir = crosscorel_2d_fft(im0,im1)
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    return [t0, t1]


def roll2d( a , shift=(10,10) ):
    # rolls image with specified shift values
    a_ =np.roll( a, shift[0], axis=0 )
    return np.roll( a_, shift[1], axis=1 )


def create_df_model_coef_correls(filename_model,filename_exp,orient) :
    #finds correlation coefficient values for each experiment and simulated configurations
    # Load the npz model file with Camphor rotations
    print ('Expreiment '+str(orient)+ ': start to calculate correlation coefficients')
    try:    
        data= np.load(filename_model)
        X_model=data['X']
        Y_model=data['predictions'] 
        Y_model = Y_model[:,:,:,1] #spheres

        sh_mod = Y_model.shape
    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)
 
    # Load the npz experiment file with 1 Camphor rotation to work
    try:    
        data= np.load(filename_exp)
        X_exp=data['X']
        Y_exp=data['Y'] 
        Y_exp = Y_exp[1][0]  #spheres
        #Y_exp = Y_exp[0][0] #disk
        sh_exp = Y_exp.shape
    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)

    
    #crop simulated data to fit experimental data lateral size
 
    if (sh_exp[0] < sh_mod[1]):
        shift_x = int((sh_mod[1] - sh_exp[0])/2)
        shift_y = int((sh_mod[2] - sh_exp[1])/2)
        X_model = X_model[:,shift_x:shift_x+sh_exp[0],shift_y:shift_y+sh_exp[1],:]
        Y_model = Y_model[:,shift_x:shift_x+sh_exp[0],shift_y:shift_y+sh_exp[1]]

    img_exp=Y_exp
    config_angles=np.zeros([sh_mod[0]]) 
    config_correls=np.zeros([sh_mod[0]]) 
    config_shifts=np.zeros([sh_mod[0],2]) 

    for i in range(sh_mod[0]): #range(sh_mod[0]): 
        img_model=Y_model[i,:,:]
        rot_cor_coef=[]
        rot_shift_x=[]
        rot_shift_y=[]
        #fig=plt.figure(figsize=(15, 15))
        fig_ind=1
        for rot_angle in range(359):
            #for each rotation in lateral plane we check correlation to find best one
            img_rot_= nimg.rotate(img_model, rot_angle, axes=(1, 0), reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
            [t0,t1] = trans_match_fft(img_exp, img_rot_)
            img_rot  = roll2d( img_rot_, shift=(t0,t1) )

            rot_cor_coef.append(correlationCoef(img_exp,img_rot  ) )
            rot_shift_x.append(t0)
            rot_shift_y.append(t1)

        best_correl=max(rot_cor_coef)
        best_angle = rot_cor_coef.index(best_correl)
        best_shifts= (rot_shift_x[best_angle],rot_shift_y[best_angle])

        config_angles[i]=best_angle
        config_correls[i]=best_correl
        config_shifts[i]=best_shifts
        if not i%10:
            print ('   working on configuration %03i/%03i ' %(i,sh_mod[0]))

    d = {'config': range(sh_mod[0]), 'angles': config_angles,'coef_correls': config_correls,'shift_x': config_shifts[:,0],'shift_y': config_shifts[:,1]}
    df = pd.DataFrame(d,columns=['config', 'angles','coef_correls','shift_x','shift_y'])
    df=df.sort_values(by='coef_correls', ascending=False)
    out_file_path='experimental_configs_data/'+str(orient)+'orient_exp_sim_cor_values.csv'

    df.to_csv(out_file_path, index=False)
    print ('best configurations saved to' + out_file_path)

def plot_best_match_configs_one(filename_model,dir_sim_geom,dir_exp,orient, num_best): 
    # plots selected experimental configurations and set of best matched simulated configurations
    texts =['experimental AFM data','NN prediction',  'geometry','simulated AFM data']
    # Load the npz model file with Camphor rotations
    print ('Experiment '+str(orient))
    try:    
        data= np.load(filename_model)
        X_model=data['X']
        Y_true=data['Y'] 
        Y_model=data['predictions']
        Y_model = Y_model[:,:,:,1] #spheres
        Y_true = Y_true[:,:,:,1] #spheres
        sh_mod = Y_model.shape

    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)
    add_norm(X_model)

    fig = plt.figure(figsize=(2.5*6,2.5*(num_best+1)))
    plot_ind=0
    gs= gridspec.GridSpec(num_best+1, 6, wspace=0.05, hspace=0.05)
      

    filename_exp=dir_exp+str(orient)+'orient_exp.npz'
    # Load the npz experiment file with 1 Camphor rotation to work
    try:    
        data= np.load(filename_exp)
        X_exp=data['X']
        Y_exp=data['Y'] 
        Y_exp  = Y_exp[1][0]
        sh_exp = Y_exp.shape
    except KeyError as e:
        print('Could not find filename %s' % e)
    except Exception as e:
        print(e)
    add_norm(X_exp) 
    
    # plot experimental 3 afm images 
    for j in [0,5,9]:  # change scandim here
        xi = X_exp[0,:,:,j]
      
        ax = plt.Subplot(fig,gs[plot_ind])

        if j==0:
            ax.set_ylabel('experiment '+str(orient))
        if j==5:
            ax.set_xlabel('AFM data')
 
        fig.add_subplot(ax)
        vmax = xi.max()
        vmin = xi.min()      
        plt.imshow(xi,  cmap='afmhot', origin="lower",vmin=vmin-0.1*(vmax-vmin),vmax=vmax+0.1*(vmax-vmin))
        plt.xticks([])
        plt.yticks([])    
        #plt.colorbar()
        plot_ind+=1

    # plot atomic spheres maps experiment predicted
 
    ax = plt.Subplot(fig,gs[plot_ind])
    fig.add_subplot(ax)
    ax.set_ylabel("predicted vdW-Spheres")
    plt.imshow(Y_exp, origin="lower") #, cmap='jet')
    plt.xticks([])
    plt.yticks([])  

    plot_ind+=1
    plot_ind+=2

    for num_conf_best in range(num_best):

        #1. Load 3 closest relaxed model orientations to experimental from csv file
 
        df_path=dir_exp +str(orient)+'orient_exp_sim_cor_values.csv'
        df = pd.read_csv(df_path) 
        config_ind=np.int(df.iloc[num_conf_best,[0]])
        best_angle=np.int(df.iloc[num_conf_best,[1]])
        best_correl = np.float(df.iloc[num_conf_best,[2]])
        best_shift_x = np.int(df.iloc[num_conf_best,[3]])
        best_shift_y= np.int(df.iloc[num_conf_best,[4]]) 
        if (sh_exp[0] < sh_mod[1]):
            shift_x = int((sh_mod[1] - sh_exp[0])/2)
            shift_y = int((sh_mod[2] - sh_exp[1])/2)
            X_model_current = X_model[config_ind,shift_x:shift_x+sh_exp[0],shift_y:shift_y+sh_exp[1],:]
            Y_model_current = Y_model[config_ind,shift_x:shift_x+sh_exp[0],shift_y:shift_y+sh_exp[1]]
            Y_true_current = Y_true[config_ind,shift_x:shift_x+sh_exp[0],shift_y:shift_y+sh_exp[1]] 
        else:
            X_model_current = X_model[config_ind,:,:,:]
            Y_model_current = Y_model[config_ind,:,:]
            Y_true_current = Y_true[config_ind,:,:] 

        # plot model 3 afm images 
        for j in  [0,5,9]: # change scandim here
            xi = X_model_current[:,:,j]

            rot_img_= nimg.rotate(xi, best_angle, axes=(1, 0), reshape=False, output=None, order=3, mode='mirror',  cval=0.0, prefilter=True)
            rot_img = roll2d( rot_img_, shift=(best_shift_x,best_shift_y) )

            ax = plt.Subplot( fig,gs[plot_ind])
            if j==0:
                ax.set_ylabel('simulated config %01i'  %(config_ind))
            if j==5:
                ax.set_xlabel('AFM data')                    
            fig.add_subplot(ax)
            vmax = xi.max()
            vmin = xi.min()      
            plt.imshow(rot_img,  cmap='afmhot', origin="lower",vmin=vmin-0.1*(vmax-vmin),vmax=vmax+0.1*(vmax-vmin))
            
            plt.xticks([])
            plt.yticks([])     
            plot_ind+=1
 


        # plot atomic spheres model predicted
        ax = plt.Subplot( fig,gs[plot_ind])
        fig.add_subplot(ax)
        ax.set_ylabel("predicted vdW-Spheres") 
        rot_img_= nimg.rotate(Y_model_current[:,:], best_angle, axes=(1, 0), reshape=False, output=None, order=3, mode='mirror',  cval=0.0, prefilter=True)
        rot_img = roll2d( rot_img_, shift=(best_shift_x,best_shift_y) )
        plt.imshow(rot_img, origin="lower") #, cmap='jet')
        ax.set_xlabel('correlation coef.= %05f' %best_correl)    
        if plot_ind//9==0:
            ax.set_title(letters[plot_ind], fontsize = font_size)
        plt.xticks([])
        plt.yticks([])  
        plot_ind+=1

        # plot atomic spheres model reference
        ax = plt.Subplot( fig,gs[plot_ind])
        fig.add_subplot(ax)
        #ax.set_ylabel("atomic spheres [preds]",fontsize = 20)

        rot_img_= nimg.rotate(Y_true_current[:,:], best_angle, axes=(1, 0), reshape=False, output=None, order=3, mode='mirror',  cval=0.0, prefilter=True)
        rot_img = roll2d( rot_img_, shift=(best_shift_x,best_shift_y) )


        plt.imshow(rot_img, origin="lower") #, cmap='jet')
        ax.set_ylabel("reference vdW-Spheres") 

        plt.xticks([])
        plt.yticks([])  
        plot_ind+=1
   

        # plot jmol molecule structure
        xyz_fname=dir_sim_geom+'%01i_orient.png' %(config_ind) 
        xyz_unrelaxed = np.flipud(misc.imread(xyz_fname))
        sh_pov_ray = xyz_unrelaxed.shape 
        pov_scale  = np.float(sh_pov_ray[0])/np.float(sh_mod[1])

        if (sh_exp[0] < sh_mod[1]):
            shift_x_pov = int(shift_x*pov_scale)
            shift_y_pov = int(shift_y*pov_scale)
            sh_exp_pov= [int(sh_exp[0]*pov_scale),int(sh_exp[1]*pov_scale)]
            #xyz_unrelaxed = xyz_unrelaxed[-shift_x-sh_exp_scale[0]: -shift_x,-shift_y-sh_exp_scale[1]:-shift_y,:]
            xyz_unrelaxed = xyz_unrelaxed[shift_x_pov:shift_x_pov+sh_exp_pov[0],shift_y_pov:shift_y_pov+sh_exp_pov[1],:]
        #print('xyz_unrelaxed.shape=', xyz_unrelaxed.shape[2])
        xyz_shift_x = int(best_shift_x*pov_scale)
        xyz_shift_y = int(best_shift_y*pov_scale) 
        rot_img_= nimg.rotate(xyz_unrelaxed, best_angle, axes=(1, 0), reshape=False, output=None, order=3, mode='constant',  cval=0.0, prefilter=True)
        rot_img = np.zeros_like(rot_img_)
        rot_img[:,:,0] = roll2d( rot_img_[:,:,0], shift=(xyz_shift_x,xyz_shift_y) )
        rot_img[:,:,1] = roll2d( rot_img_[:,:,1], shift=(xyz_shift_x,xyz_shift_y) )
        rot_img[:,:,2] = roll2d( rot_img_[:,:,2], shift=(xyz_shift_x,xyz_shift_y) )
        rot_img[:,:,3] = roll2d( rot_img_[:,:,3], shift=(xyz_shift_x,xyz_shift_y) )
        ax = plt.Subplot(fig,gs[plot_ind])
        ax.set_ylabel("predicted geometry")
        fig.add_subplot(ax)
        plt.imshow(rot_img, origin="lower")
        plt.xticks([])
        plt.yticks([])
        
        plot_ind+=1
    
    gs.tight_layout(fig)
    plt.show() 
    
def plot_best_match_configs_all(filename_model,dir_sim_geom,dir_exp,orientations):
   
    num_best_confs = [0,0,0, 0,0]
 
    letters=['a','b','c','d','e','f','g','h','i']
    numbers=[1,2,3,4,5]
    texts =['experimental AFM data','NN prediction',  'geometry','simulated AFM data']
    font_size=50
    text_props = dict(boxstyle='round', facecolor='none', edgecolor='none')
    amount_orient=np.size(orientations)
    fig = plt.figure(figsize=(45,5*amount_orient))
    plot_ind=0
    text_ind=0
    gs= gridspec.GridSpec(amount_orient, 9, wspace=0.1, hspace=0.1)


    # Load the npz model file with Camphor rotations
    try:    
        data= np.load(filename_model)
        X_model=data['X']
        Y_true=data['Y'] 
        Y_model=data['predictions']

        Y_model = Y_model[:,:,:,1] #spheres


        sh_mod = Y_model.shape

    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)
    add_norm(X_model)

    for i in range(amount_orient):
        num_best=num_best_confs[i]
        #num_best = 0
        orient=orientations[i]
        df_path=dir_exp +str(orient)+'orient_exp_sim_cor_values.csv'
        df = pd.read_csv(df_path) 
        config_ind=np.int(df.iloc[num_best,[0]])
        best_angle=np.int(df.iloc[num_best,[1]])
        best_correl = np.float(df.iloc[num_best,[2]])
        best_shift_x = np.int(df.iloc[num_best,[3]])
        best_shift_y= np.int(df.iloc[num_best,[4]])
 
        filename_exp=dir_exp+str(orient)+'orient_exp.npz'
        # Load the npz experiment file with 1 Camphor rotation to work
        try:    
            data= np.load(filename_exp)
            X_exp=data['X']
            Y_exp=data['Y'] 
            Y_exp  = Y_exp[1][0]
 
            sh_exp = Y_exp.shape



        except KeyError as e:
            print('Could not find filename %s' % e)

        except Exception as e:
            print(e)


        add_norm(X_exp) 

        if (sh_exp[0] < sh_mod[1]):
            shift_x = int((sh_mod[1] - sh_exp[0])/2)
            shift_y = int((sh_mod[2] - sh_exp[1])/2)
            X_model_current = X_model[config_ind,shift_x:shift_x+sh_exp[0],shift_y:shift_y+sh_exp[1],:]
            Y_model_current = Y_model[config_ind,shift_x:shift_x+sh_exp[0],shift_y:shift_y+sh_exp[1]]

        else:
            X_model_current = X_model[config_ind,:,:,:]
            Y_model_current = Y_model[config_ind,:,:]


        # plot experimental 3 afm images 
        for j in [0,5,9]:  # change scandim here
            xi = X_exp[0,:,:,j]
            #print 'int(j / 5)='+str(int(j / 5))+', j % 5='+str(j % 5)      

            ax = plt.Subplot(fig,gs[plot_ind])
            fig.add_subplot(ax)
            if j==0:
                ax.set_ylabel(numbers[plot_ind//9], fontsize = font_size,rotation=0,labelpad=30)

            if plot_ind//9==0:
                ax.set_title(letters[plot_ind], fontsize = font_size)
            if i ==amount_orient-1 and j==5 :
                ax.text(-0.60, -0.1, texts[text_ind], transform=ax.transAxes, fontsize=font_size,
                horizontalalignment='left', verticalalignment='top', bbox=text_props)
                text_ind+=1


            vmax = xi.max()
            vmin = xi.min()  
            #plt.imshow(xi,  cmap='afmhot',   origin="lower",vmin=vmin,vmax=vmax)    
            plt.imshow(xi,  cmap='afmhot', origin="lower",vmin=vmin-0.1*(vmax-vmin),vmax=vmax+0.1*(vmax-vmin))
            plt.xticks([])
            plt.yticks([])    
            #plt.colorbar()
            plot_ind+=1

        # plot atomic spheres maps experiment predicted

        ax = plt.Subplot(fig,gs[plot_ind])
        fig.add_subplot(ax)
        #ax.set_ylabel("atomic spheres [preds]",fontsize = 20)
        if plot_ind//9==0:
            ax.set_title(letters[plot_ind], fontsize = font_size)
        if i ==amount_orient-1:
            ax.text(0.4, -0.1, texts[text_ind], transform=ax.transAxes, fontsize=font_size,
            horizontalalignment='left', verticalalignment='top', bbox=text_props)
            text_ind+=1
        plt.imshow(Y_exp, origin="lower") #, cmap='jet')
        plt.xticks([])
        plt.yticks([])  

        plot_ind+=1

        # plot atomic spheres model predicted
        ax = plt.Subplot( fig,gs[plot_ind])
        fig.add_subplot(ax)
        #ax.set_ylabel("atomic spheres [preds]",fontsize = 20)

        rot_img_= nimg.rotate(Y_model_current[:,:], best_angle, axes=(1, 0), reshape=False, output=None, order=3, mode='mirror',  cval=0.0, prefilter=True)
        rot_img = roll2d( rot_img_, shift=(best_shift_x,best_shift_y) )


        plt.imshow(rot_img, origin="lower") #, cmap='jet')
        #ax.set_xlabel('correlation coef.= %05f' %best_correl,fontsize = 20)    
        if plot_ind//9==0:
            ax.set_title(letters[plot_ind], fontsize = font_size)
        plt.xticks([])
        plt.yticks([])  

        plot_ind+=1


        # plot jmol molecule structure
        xyz_fname=dir_sim_geom+'%01i_orient.png' %(config_ind)
        xyz_unrelaxed = np.flipud(misc.imread(xyz_fname))
        sh_pov_ray = xyz_unrelaxed.shape 
        pov_scale  = np.float(sh_pov_ray[0])/np.float(sh_mod[1]) 
        if (sh_exp[0] < sh_mod[1]):
            shift_x_pov = int(shift_x*pov_scale)
            shift_y_pov = int(shift_y*pov_scale)
            sh_exp_pov= [int(sh_exp[0]*pov_scale),int(sh_exp[1]*pov_scale)]
            #xyz_unrelaxed = xyz_unrelaxed[-shift_x-sh_exp_scale[0]: -shift_x,-shift_y-sh_exp_scale[1]:-shift_y,:]
            xyz_unrelaxed = xyz_unrelaxed[shift_x_pov:shift_x_pov+sh_exp_pov[0],shift_y_pov:shift_y_pov+sh_exp_pov[1],:]
        #print('xyz_unrelaxed.shape=', xyz_unrelaxed.shape[2])
        xyz_shift_x = int(best_shift_x*pov_scale)
        xyz_shift_y = int(best_shift_y*pov_scale) 
        rot_img_= nimg.rotate(xyz_unrelaxed, best_angle, axes=(1, 0), reshape=False, output=None, order=3, mode='constant',  cval=0.0, prefilter=True)
        rot_img = np.zeros_like(rot_img_)
        rot_img[:,:,0] = roll2d( rot_img_[:,:,0], shift=(xyz_shift_x,xyz_shift_y) )
        rot_img[:,:,1] = roll2d( rot_img_[:,:,1], shift=(xyz_shift_x,xyz_shift_y) )
        rot_img[:,:,2] = roll2d( rot_img_[:,:,2], shift=(xyz_shift_x,xyz_shift_y) )
        rot_img[:,:,3] = roll2d( rot_img_[:,:,3], shift=(xyz_shift_x,xyz_shift_y) )
        #print'rot_img.shape=', xyz_image.shape
        #if (sh_exp[0] < sh_mod[1]):
            #rot_img = rot_img[-shift_x-sh_exp_scale[0]: -shift_x,-shift_y-sh_exp_scale[1]:-shift_y,:]
            #rot_img = rot_img[shift_x:shift_x+sh_exp_scale[0],shift_y:shift_y+sh_exp_scale[1],:]
        ax = plt.Subplot(fig,gs[plot_ind])
        #ax.set_ylabel("model config. %01i var %01i" %(model_config,config_ind),fontsize = 20)
        fig.add_subplot(ax)
        if plot_ind//9==0:
            ax.set_title(letters[plot_ind], fontsize = font_size)
        if i ==amount_orient-1:
            ax.text(0.5, -0.1, texts[text_ind], transform=ax.transAxes, fontsize=font_size,
            horizontalalignment='center', verticalalignment='top', bbox=text_props)
            text_ind+=1           
        plt.imshow(rot_img, origin="lower")
        #plt.imshow(xyz_unrelaxed)

        plt.xticks([])
        plt.yticks([])

        plot_ind+=1


        X_model_current_ = np.expand_dims(X_model_current,axis=0) 
        X_model_current = X_model_current_[0,:,:,:]
        # plot model 3 afm images 
        for j in  [0,5,9]: # change scandim here
            xi = X_model_current[:,:,j]

            rot_img_= nimg.rotate(xi, best_angle, axes=(1, 0), reshape=False, output=None, order=3, mode='mirror',  cval=0.0, prefilter=True)
            rot_img = roll2d( rot_img_, shift=(best_shift_x,best_shift_y) )

            ax = plt.Subplot( fig,gs[plot_ind])
            #if j==0:
            #    ax.set_ylabel("model config %01i var %01i"  %(model_config,config_ind) ,fontsize = 20)
            if plot_ind//9==0:
                ax.set_title(letters[plot_ind], fontsize = font_size)
            if i ==amount_orient-1 and j==5 :
                ax.text(-0.5, -0.1, texts[text_ind], transform=ax.transAxes, fontsize=font_size,
                horizontalalignment='left', verticalalignment='top', bbox=text_props)
                text_ind+=1

            fig.add_subplot(ax)
            vmax = xi.max()
            vmin = xi.min()      
            #plt.imshow(rot_img,  cmap='afmhot',   origin="lower",vmin=vmin,vmax=vmax)    
            plt.imshow(rot_img,  cmap='afmhot', origin="lower",vmin=vmin-0.1*(vmax-vmin),vmax=vmax+0.1*(vmax-vmin))

            plt.xticks([])
            plt.yticks([])     
            plot_ind+=1

    gs.tight_layout(fig)
    plt.show() 
