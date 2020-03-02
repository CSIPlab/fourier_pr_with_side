import torch
from torch import nn
from os.path import isfile,isdir
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from skimage.measure import compare_ssim as ssim
import scipy.misc as sm
import time
start_time = time.time()

def Fourier_mtx2(ngf,ndf):
    Wr=np.exp(2*np.pi*1j/ngf)
    Wc=np.exp(2*np.pi*1j/ndf)
    
    G=np.zeros((ngf*ndf,ngf*ndf))
    
    G=G+0j
    for u in range (0,ngf*ndf):
        k=np.floor(u/np.float(ndf))
        l=u-k*ndf
        for v in range (0,ngf*ndf):
            p=np.floor(v/np.float(ndf))
            q=v-p*ndf
            G[ngf*ndf-1-u,v]=(Wr**((k+1)*p))*(Wc**((l+1)*q))
    
    return G



multiple=4
#known_position=4


train_dataset='MNIST'
test_dataset='MNIST'
test_serial=np.load('/media/rakib/Data/Data/MNIST/mnist_10digit_100.npy') # 0-9 digits 10 times

test_digits=[test_serial[36],test_serial[37],test_serial[38],test_serial[39],test_serial[21]]

z_dim=20

noisy=0

sub_dim_pool=[32*32]#
sub_dim=sub_dim_pool[0]
sub_dim_mse=[]
sub_dim_psnr=[]
sub_dim_ssim=[]
sub_dim_rec=[]


#for sub_dim in sub_dim_pool:

print(sub_dim)
#    sub_dim=128#512
subtype_pool=['abs']# 'abs','linear','square','original'
train_z_type='PCA'
test_z_type='Gauss'#'Gauss'
train_epochs=500

test_epochs1=100#30
test_outer1=20#15

test_epochs2=100#100
test_outer2=200#50

test_epochs3=200#500
test_outer_epochs=500#500

train_batch_size=256
test_batch_size=1
opt='SGD'
if opt=='SGD':
    train_lr=1
elif opt=='Adam':
    train_lr=1e-3
train_alpha=200

z_norm_type='free'
x_init='random'#'power_project','spectral','random'

seed=100

random_restart=1
#if x_init=='power_project':
#  pow_it=10
#else:
#    pow_it=500
#test_pow_epochs=800
eta=0.6


train=0
test=1
mask_id=10
rotation_theta=0


test_size=1 # choose something greater than or equal to test_batcch_size (otherwise test_alpha should be normalized)
start_fig=0
#mask_ratio=0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
nz = z_dim
ngf = 32
ndf =32
if train_dataset=='MNIST' or train_dataset=='SVHN_gray' or train_dataset=='EMNIST_digit' or train_dataset=='FashionMNIST':
    nc = 1
elif train_dataset=='SVHN' or train_dataset=='CIFAR10':
    nc = 3
    
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
#            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2,1, bias=False),
#            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf),
#            nn.ReLU(True),SGD
#            # state size. (ngf) x 32 x 32
#            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

loss_subtype=[]  
loss_outer_subtype=[]
subtype='abs'
for known_position in range(1,multiple+1): 


    test_alpha=1.28e-1*np.float(train_alpha)*test_batch_size/train_batch_size#200
        
    datax=np.load('/media/rakib/Data/Data/GLO/Feature/'+test_dataset+'.npz')
    x_test=datax['x_test']
    x_test_org=x_test
    if rotation_theta!=0:
        x_rot=np.zeros((x_test.shape[0],nc,ngf,ndf))
        for i in range (0, x_test.shape[0]):
            for j in range (0, nc):
                x_rot[i,j,:,:]=rotate(x_test[i,j,:,:],rotation_theta,reshape=False)
        x_test=x_rot       

    #    dataz=np.load('/media/rakib/Data/Data/GLO/Feature/'+test_dataset+'_'+test_z_type+'_'+str(z_dim)+'.npz')
    #    z_test=dataz['z_init_test']
    if test_z_type=='Gauss':
        z_test=np.zeros((x_test.shape[0],z_dim))
        for i in range (0,z_test.shape[0]):
            np.random.seed(seed)
            z_test[i,:]=np.random.normal(loc=0.0, scale=1.0, size=(1,z_dim))
        for i in range(z_test.shape[0]):
            z_test[i] = z_test[i, :] / np.linalg.norm(z_test[i, :], 2)
    elif test_z_type=='Optimum':
        z_test=np.load( '/media/rakib/Data/Data/GLO/Results/'+train_dataset+'_'+'Gauss'+'_'+str(z_dim)+'_alpha'+str(np.float(train_alpha))+'_lr'+str(train_lr)+'_epochs'+str(200)+'wobatchnorm.npy') 
    ## Select test size
    if test_size !=-1:
#            if multiple==2:
#                x_test=np.concatenate((x_test[0:test_size,:,:,:] ,x_test[test_size:2*test_size,:,:,:]), axis=3) 
#                x_test_org=np.concatenate((x_test_org[0:test_size,:,:,:] ,x_test_org[test_size:2*test_size,:,:,:]), axis=3)
#            elif multiple>2:
#                at=sm.imread('/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/characters/zip.png')
#                at=at.reshape((nc,ngf,ndf))/255.0
        
        for i in range (0,multiple):
            if multiple>i:
                x_pos=x_test[test_digits[i],:,:,:]
                if known_position==i+1:
                    at=x_pos
                if i==0:
                    x_temp=x_pos
                else:
                    x_temp=np.concatenate((x_temp,x_pos), axis=2) 
                
#                x_temp=np.concatenate((x_temp,x_test[1,:,:,:]), axis=2)
#                x_temp=np.concatenate((x_temp,x_test[5,:,:,:]), axis=2)
        
        # @ darpa
#                at=sm.imread('/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/characters/and.png')
#                at=at.reshape((nc,ngf,ndf))/255.0
##                
#                at1=sm.imread('/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/d.png')
#                at1=at1.reshape((nc,ngf,ndf))/255.0
#                at2=sm.imread('/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/a.png')
#                at2=at2.reshape((nc,ngf,ndf))/255.0
#                at3=sm.imread('/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/r.png')
#                at3=at3.reshape((nc,ngf,ndf))/255.0
#                at4=sm.imread('/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/p.png')
#                at4=at4.reshape((nc,ngf,ndf))/255.0
#                at5=sm.imread('/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/a.png')
#                at5=at5.reshape((nc,ngf,ndf))/255.0              
#                x_temp=np.concatenate((at,at1,at2,at3,at4,at5), axis=2) 
        
        x_test=x_temp.reshape((1,nc,ngf,ndf*multiple))
        x_test_org=x_test

        
    else:
        x_test=x_test[0:test_size,:,:,:]    
        x_test_org=x_test_org[0:test_size,:,:,:]  
        z_test=z_test[0:test_size,:]


#    generator = torch.load(  '/media/rakib/Data/Data/GLO/Model/'+train_dataset+'_'+train_z_type+'_'+str(z_dim)+'_alpha'+str(train_alpha)+'_lr'+str(train_lr)+'_epochs'+str(train_epochs)+'wobatchnorm')
#    optimizer = torch.optim.SGD(generator.parameters(), train_lr)

    # Random Matrix
        # Random Matrix

    mask=Fourier_mtx2(ngf,ndf*multiple)
        

# Applying mask
    if subtype=='linear':
        x_test=2*x_test-1
        x_test_temp=np.zeros((x_test.shape[0],nc*ngf*ndf*multiple,1))
        for i in range (0, x_test.shape[0]):
            x_test_temp[i,:,:]=np.matmul(mask[:,:],x_test[i,:,:,:].flatten().reshape(1,nc*ngf*ndf*multiple,1))
        x_test=x_test_temp
    elif subtype=='abs': 
        x_test=2*x_test-1
        x_test_temp=np.zeros((x_test.shape[0],nc*ngf*ndf*multiple,1))
        for i in range (0, x_test.shape[0]):
#                x_test_temp[i,:,:]=np.abs(np.matmul(mask[:,:],x_test[i,:,:,:].flatten().reshape(1,nc*ngf*ndf*multiple,1)))
            x_test_temp[i,:,:]=np.abs(np.fft.fft2(x_test[i,0,:,:]).flatten().reshape((1,nc*ngf*ndf*multiple,1))/np.sqrt(nc*ngf*ndf*multiple))
        x_test=x_test_temp
    elif subtype=='square':
        x_test=2*x_test-1
        x_test_temp=np.zeros((x_test.shape[0],nc*ngf*ndf*multiple,1))
        for i in range (0, x_test.shape[0]):
            x_test_temp[i,:,:]=np.square(np.matmul(mask[:,:],x_test[i,:,:,:].flatten().reshape(1,nc*ngf*ndf*multiple,1)))
        x_test=x_test_temp



    # noise add
    if noisy==1:
        noise=np.random.normal(loc=0,scale=1,size=x_test.shape)
        x_test=x_test+noise

  
        
    test_size=x_test.shape[0]
    batch_no=np.int(np.ceil(test_size/test_batch_size))
    idx=np.arange(test_size)
    
    x_rec=np.zeros((x_test.shape[0],nc,ngf,ndf*multiple))
    x_rec1=np.zeros((x_test.shape[0],nc,ngf,ndf*multiple))
    
    init_loss=[]
    loss_test_outer=[]
    loss_test=[]
    meas_loss=[]
    for batch_idx in range(0,batch_no):
        x_best=np.zeros((test_batch_size,nc*ngf*ndf*multiple,1))
#            z_test_temp=z_test
        for rr in range (0,random_restart):
            #generator = torch.load(  '/media/rakib/Data/Data/GLO/Model/'+train_dataset+'_'+train_z_type+'_'+str(z_dim)+'_alpha'+str(train_alpha)+'_lr'+str(train_lr)+'_epochs'+str(train_epochs)+'wobatchnorm')
            generator1 = torch.load(  '/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/models/MNIST_32_z'+str(z_dim)+'_dcgan')
#                generator2 = torch.load(  '/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/models/MNISTfliplr_32_z'+str(z_dim)+'_dcgan')
#                generator3 = torch.load(  '/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/models/MNISTflipud_32_z'+str(z_dim)+'_dcgan')
#                generator4 = torch.load(  '/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/models/MNISTfliplrud_32_z'+str(z_dim)+'_dcgan')

    #        if batch_idx%100==0:
#            print(batch_idx)
            epoch_idx=idx    
            y=x_test[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,0].reshape(test_batch_size,nc*ngf*ndf*multiple,1)
            x_true=2*x_test_org[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:,:].reshape(test_batch_size,nc*ngf*ndf*multiple,1)-1
            
            np.random.seed(seed) # Fixing seed
#                x=np.random.randn(test_batch_size,nc*ngf*ndf*multiple,1)
            
#                y_temp=y.reshape(test_batch_size,nc,ngf,ndf*multiple)
#                y_temp=(y_temp-np.min(y_temp))/(np.max(y_temp)-np.min(y_temp))
            
#                y_temp=np.zeros((test_batch_size,nc,ngf,ndf*multiple))
            
            y_temp=np.real(np.fft.ifft2(y.reshape(ngf,ndf*multiple))*np.sqrt(nc*ngf*ndf*multiple))
            y_temp=y_temp.reshape(test_batch_size,nc,ngf,ndf*multiple)
            y_temp=(y_temp-np.min(y_temp))/(np.max(y_temp)-np.min(y_temp))
            y_temp=2*y_temp-1
            
            if known_position==1:
                x_temp=np.concatenate((2*at.reshape(1,nc,ngf,ndf)-1,y_temp[:,:,:,ndf:ndf*multiple]),axis=3)
            elif known_position<multiple:    
                x_b=y_temp[:,:,:,0:ndf*(known_position-1)]
                x_e=y_temp[:,:,:,ndf*known_position:ndf*multiple]                
                x_temp=np.concatenate((x_b,2*at.reshape(1,nc,ngf,ndf)-1,x_e),axis=3)
            else:
                x_temp=np.concatenate((y_temp[:,:,:,0:ndf*(multiple-1)],2*at.reshape(1,nc,ngf,ndf)-1),axis=3)
            x=x_temp.reshape(test_batch_size,nc*ngf*ndf*multiple,1)
            
            p=np.zeros(y.shape)
    #        p=np.sign(y)

            x_in=x
            #plt.figure()
            loss_outer_epoch=[]
            
            

            
            ## Beginning
            x_batch=np.zeros(x.shape)
            x_batch=x_batch+0j
            
            x_i=  np.imag(x_batch) 

            
            for outer_epoch in range (0, test_outer_epochs):
                if outer_epoch%50==0:
                    print(outer_epoch)
#                    if outer_epoch==0:
#                        test_middle_epochs=test_middle_epoch1
#                    else:
#                        test_middle_epochs=test_middle_epoch2
#                    for T in range(0,test_middle_epochs):
#                        for i in range (0, test_batch_size):
#                            p[i,:,:]=np.sign(np.matmul(mask[0,:,:],x[i,:,:]))
                
                if outer_epoch<test_outer1:
                    test_epochs=test_epochs1
                elif outer_epoch<test_outer2:
                    test_epochs=test_epochs2
                else:
                    test_epochs=test_epochs3   
                
                for i in range (0, test_batch_size):
#                    p[i,:,0]=np.angle(np.dot(mask,x[i,:,0]))
                    p[i,:,0]=np.angle(np.fft.fft2(x[i,:,0].reshape(ngf,ndf*multiple))/np.sqrt(nc*ngf*ndf*multiple)).reshape(ngf*ndf*multiple)


                #
                b=np.multiply(y,np.exp(1j*p))
                for i in range (0, test_batch_size):
                    A=mask#[:,:].reshape(sub_dim,nc*ngf*ndf)
#                        (sol,res,rank,sing)=np.linalg.lstsq(A, b[i].flatten())
#                        sol=x[i]+eta*np.matmul(A.conj().T/(ngf*ndf*multiple),(b[i]-np.matmul(A,(x[i]))))
                    sol=x[i]+eta*np.fft.ifft2(b[i].reshape(ngf,ndf*multiple)-np.fft.fft2(x[i].reshape(ngf,ndf*multiple))/np.sqrt(nc*ngf*ndf*multiple)).reshape(ngf*ndf*multiple,1)*np.sqrt(nc*ngf*ndf*multiple)
                    x_batch[i,:,0]=sol.reshape(nc*ngf*ndf*multiple)
#                        x_batch[i,:,0]=np.dot(np.linalg.pinv(A),b[i,:,0])
                    
                    
                    
                x_r=  np.real(x_batch)#np.abs(x_batch) 
                
                
#                    xrr=x_r
#                    for i in range(0,4096):
#                        if x_r[0,i,0]>1:
#                            xrr[0,i,0]=1
#                        elif x_r[0,i,0]<-1:
#                            xrr[0,i,0]=-1                     
#                    xrr=xrr/2+0.5
#                    sm.imsave('/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/Results/appgd_gen/7210/7210_grad'+str(outer_epoch+1)+'.png',xrr.reshape((ngf,ndf*multiple)))
#                    print(np.min(x_r))
#                    print(np.max(x_r)) 
                
#                    x_m1=np.min(x_r)
#                    x_r=x_r-x_m1
#                    x_m2=np.max(x_r)
#                    x_r=x_r/x_m2
#                    x_r=2*x_r-1
                

                

                x_rt=x_r.reshape((test_batch_size,nc,ngf,ndf*multiple))
#                x_r1= x_rt[:,:,:,0:ndf]
#                x_r2= x_rt[:,:,:,2*ndf:3*ndf]
#                x_r3= x_rt[:,:,:,3*ndf:4*ndf]
#                    x_r4= x_rt[:,:,:,4*ndf:5*ndf]
#                    x_r5= x_rt[:,:,:,5*ndf:6*ndf]
#                    x_i=  np.imag(x_batch)
#                    x_i=2*(x_i-x_m1)/x_m2-1
                
#                    
#                    if outer_epoch==0:
#                        np.savez('w_p_0',w=x,p=p)
#                    elif outer_epoch==1:
#                        np.savez('w_p_1',w=x,p=p)  
#                    elif outer_epoch==test_outer_epochs-1:
#                        np.savez('w_p_end',w=x,p=p) 
                
                loss_outer_epoch.append(np.mean((x_true-x)**2)) 
                
                

                
#                    z_batch=z_test[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]
#                    np.random.seed(seed)
#                xt1=[]
#                xt2=[]
#                xt3=[]
#                xt4=[]
#                xt5=[]
                for digit in range (0,multiple):
                    if digit!=(known_position-1):
                        xrr=x_rt[:,:,:,ndf*digit:ndf*(digit+1)]

                        for gen_id in range(1):
                            
                            if gen_id==0:
                                generator=generator1
    #                            elif gen_id==1:
    #                                generator=generator2
    #                            elif gen_id==2:
    #                                generator=generator3
    #                            elif gen_id==3:
    #                                generator=generator4
                            
                                
                                
                            if opt=='SGD':
                                optimizer = torch.optim.SGD(generator.parameters(), train_lr)
                            elif opt=='Adam':
                                optimizer = torch.optim.Adam(generator.parameters(), train_lr) 
                            
                            np.random.seed(seed)
                            z_batch= np.random.normal(loc=0.0, scale=1.0, size=(test_batch_size,z_dim))
                            for i in range(z_batch.shape[0]):
                                z_batch[i] = z_batch[i, :] / np.linalg.norm(z_batch[i, :], 2)
                                
                            
                            loss_epoch1=[]
                            for epoch in range (0,test_epochs):
                                if subtype=='linear' or subtype=='abs' or subtype=='square':
                                    x_batch_tensor=torch.cuda.FloatTensor(xrr).view(-1, ngf*ndf*nc,1)
                
                                z_batch_tensor=torch.autograd.Variable(torch.cuda.FloatTensor(z_batch).view(-1, z_dim, 1, 1),requires_grad=True)
                
                                x_hat = generator(z_batch_tensor)
                
                                loss=(x_hat.view(-1, nc* ngf*ndf,1) - x_batch_tensor).pow(2).mean()
                                loss_epoch1.append(loss.item())
                
                                optimizer.zero_grad()
                                loss.backward()        
    #                                optimizer.step()
                
                                with torch.no_grad():        
                                    z_grad = z_batch_tensor.grad.data.cuda()    
                                    z_update = z_batch_tensor - test_alpha * z_grad
                                    z_update = z_update.cpu().numpy()
                                    z_update=np.reshape(z_update,z_batch.shape)
                                    if z_norm_type=='unit_norm':
                                        for i in range(z_update.shape[0]):
                                            z_update[i,:] = z_update[i, :] / np.linalg.norm(z_update[i, :], 2)
                                    z_batch=z_update    
                                    z_test[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]=z_update
                            
                            
                            z_update_tensor=torch.autograd.Variable(torch.cuda.FloatTensor(z_update).view(-1, z_dim, 1, 1))
                            x_hat = generator(z_update_tensor)
                            x1=np.reshape(x_hat.cpu().detach().numpy(),xrr.shape)
                            
                            
                        if digit==0:
                            x_conc=x1
                        else:
                            x_conc=np.concatenate((x_conc,x1),axis=3)
                    else:
                        if digit==0:
                            x_conc=2*at.reshape(1,nc,ngf,ndf)-1
                        else:
                            x_conc=np.concatenate((x_conc,2*at.reshape(1,nc,ngf,ndf)-1),axis=3)
#                        if digit==0:
#                            xt1.append(x1)
#                        elif digit==1:
#                            xt2.append(x1)
#                        elif digit==2:
#                            xt3.append(x1)
#                        elif digit==3:
#                            xt4.append(x1)
#                        elif digit==4:
#                            xt5.append(x1)

                    
                
                
                
#                    x_conc=np.zeros((test_batch_size, nc* ngf*ndf*multiple,1))
#                    for i in range (test_batch_size):
#                        
#                        for j in range (len(xt1)):
#                            x_temp1=xt1[j]
#                            for k in range (len(xt2)):
#                                x_temp2=xt2[k]   
#                                x_temp=np.concatenate((x_temp1[i],x_temp2[i]),axis=2)
#                                x_meas=np.abs(np.fft.fft2(x_temp.reshape((ngf,ndf*multiple))).flatten().reshape((1,nc*ngf*ndf*multiple,1)))
#                                if j==0 and k==0:
#                                    meas_loss=np.mean((y[i]-x_meas)**2)
#                                    x_conc[i]=x_temp.reshape((1,nc*ngf*ndf*multiple,1))
#                                elif meas_loss>np.mean((y[i]-x_meas)**2):
#                                    meas_loss=np.mean((y[i]-x_meas)**2)
#                                    x_conc[i]=x_temp.reshape((1,nc*ngf*ndf*multiple,1))
#                x_conc=np.concatenate((xt1[0].reshape(1,nc,ngf,ndf),2*at.reshape(1,nc,ngf,ndf)-1),axis=3)
#                x_conc=np.concatenate((x_conc,xt2[0].reshape(1,nc,ngf,ndf)),axis=3)
#                x_conc=np.concatenate((x_conc,xt3[0].reshape(1,nc,ngf,ndf)),axis=3)
#                    x_conc=np.concatenate((x_conc,xt4[0].reshape(1,nc,ngf,ndf)),axis=3)
#                    x_conc=np.concatenate((x_conc,xt5[0].reshape(1,nc,ngf,ndf)),axis=3)
                x=x_conc
#                    x=np.concatenate((x1,x2),axis=3)
                    
                x=x.reshape((test_batch_size, nc* ngf*ndf*multiple,1))
                
                
                x_meas=np.abs(np.fft.fft2(x.reshape((ngf,ndf*multiple))).flatten().reshape((1,nc*ngf*ndf*multiple,1))/np.sqrt(nc*ngf*ndf*multiple))
                meas_loss.append(np.sum((y-x_meas)**2)/np.sum(y**2))
#                    sm.imsave('/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/Results/appgd_gen/7210/7210_proj'+str(outer_epoch+1)+'.png',(x/2+0.5).reshape((ngf,ndf*multiple)))
                
                
                
#                    if outer_epoch==0:
#                        np.savez('x_z_0',x=x,z=z_test)
#                    elif outer_epoch==1:
#                        np.savez('x_z_1',x=x,z=z_test) 
#                    elif outer_epoch==test_outer_epochs-2:
#                        np.savez('x_z_semi_end',x=x,z=z_test) 
#                    elif outer_epoch==test_outer_epochs-1:
#                        np.savez('x_z_end',x=x,z=z_test) 
                    
            if np.mean((x_true-x)**2)<np.mean((x_true-x_best)**2) or random_restart==1:
                x_best=x
                x_rec[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]=x.reshape(x.shape[0],nc,ngf,ndf*multiple)
                x_rec1[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]=x_batch.reshape(x_batch.shape[0],nc,ngf,ndf*multiple)
                loss_epoch_best=loss_epoch1
                loss_outer_epoch_best=loss_outer_epoch
                x_in_best=x_in
#                z_test=z_test_temp
                #loss_test .append(np.array(loss_epoch))
    #            plt.figure()
    #            plt.plot( np.array(loss_epoch))
#            plt.plot(np.array(loss_outer_epoch))
#            plt.show()

        
        init_loss.append(np.mean((x_true-x_in_best)**2)/np.mean(x_true**2))
        loss_test .append(np.array(loss_epoch_best))
        loss_test_outer.append(np.array(loss_outer_epoch_best))
#            plt.figure()
#            plt.plot(loss_outer_epoch)
#            plt.title('x update loss during testing')
#            plt.show()

#        np.savez('/media/rakib/Data/Data/GLO/icassp2019/Results/'+train_dataset+'_'+test_z_type+'_'+str(z_dim)+z_norm_type+'_alpha'+str(test_alpha)+'_epochs'+str(test_epochs)+'_'+str(test_batch_size)+'phase_retrieval_init_'+x_init+'_eta_'+str(eta)+subtype+'_subsample_'+str(sub_dim),z_test=z_test,mask=mask,x_rec=x_rec)

    loss_outer_subtype=np.array(loss_test_outer)
    loss_subtype=np.array(loss_test)




    #    x_test=x_test/2+0.5
    x_rec=x_rec/2+0.5
    x_rec1=x_rec1/2+0.5
    mse=np.mean((x_rec-x_test_org)**2)
    print(mse)
    psnr=20*np.log10((np.max(x_test_org)-np.min(x_test_org))/np.sqrt(mse))
    

    print(psnr)
    
    ssim_mnist=np.zeros(x_test_org.shape[0])
    for i in range (0,x_test_org.shape[0]):
        img1=x_test_org[i]
        img2=x_rec[i]
        if nc==3:
            img_true=np.zeros((img1.shape[1],img1.shape[2],img1.shape[0]))
            img_rec=np.zeros((img2.shape[1],img2.shape[2],img2.shape[0]))
            for chan in range (0,nc):
                img_true[:,:,chan]=img1[chan,:,:]
                img_rec[:,:,chan]=img2[chan,:,:]
            ssim_mnist[i]=ssim(img_true, img_rec,data_range=img_rec.max() - img_rec.min(), multichannel=True)
        elif nc==1:
            img_true=img1.reshape(ngf,ndf*multiple)  
            img_rec=img2.reshape(ngf,ndf*multiple)  
            ssim_mnist[i]=ssim(img_true, img_rec,data_range=img_rec.max() - img_rec.min())
    
    print(np.mean(ssim_mnist))
    sub_dim_ssim.append(np.mean(ssim_mnist))
    sub_dim_mse.append(mse)
    sub_dim_psnr.append(psnr)
    sub_dim_rec.append(x_rec)
#        print(subtype)
#        print(test_alpha)
    
    print('Initialization loss: ',np.mean(np.array(init_loss)))
    print('Initialization loss var: ',np.var(np.array(init_loss)))
    
#    for i in range (0,len(subtype_pool)):
    plt.figure()
    plt.plot(np.mean(loss_subtype,axis=0))
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction loss (Projection step) during testing')
    plt.title('Reconstruction loss (Projection step) vs epochs during testing')
    plt.figure()
    plt.plot(np.mean(loss_outer_subtype,axis=0))
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction loss (Phase update step) during testing')
    plt.title('Reconstruction loss (Phase update step) vs epochs during testing')
    plt.show()

    plt.figure()
    plt.plot(meas_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Measurement error during testing')
    plt.title('Measurement error vs epochs during testing')
    plt.show()
	

#plt.figure()
#plt.plot(np.array(sub_dim_pool),np.array(sub_dim_mse),'*-')
#plt.xlabel('Number of measurements (m)')
#plt.ylabel('Reconstruction error (per pixel)')
#plt.show()
#
#plt.figure()
#plt.plot(np.array(sub_dim_pool),np.array(sub_dim_psnr),'*-')
#plt.xlabel('Number of measurements (m)')
#plt.ylabel('PSNR')
#plt.show()
#
#plt.figure()
#plt.plot(np.array(sub_dim_pool),np.array(sub_dim_ssim),'*-')
#plt.xlabel('Number of measurements (m)')
#plt.ylabel('Mean SSIM')
#plt.show()

#x_diff=x_test_org-x_rec
#
#mm=np.mean((x_test_org-x_rec)**2,axis=(1,2,3)) 
##mmx=np.argsort(mm)
##midx=np.int(len(mmx)/2)
##figset=[mmx[0+start_fig],mmx[1+start_fig],mmx[2+start_fig],mmx[midx-1-start_fig],mmx[midx],mmx[midx+1+start_fig],mmx[-3-start_fig],mmx[-2-start_fig],mmx[-1-start_fig]]
    n=1
    figset=np.arange(n)+0*n
    
    #print(figset)
    fig_row=len(sub_dim_pool)+1
    plt.figure(figsize=(40, fig_row*2))
    
    for i in range(n):
    
        if nc==1:
#            for sublen in range(0,len(sub_dim_pool)):
                
            ax = plt.subplot(fig_row, n,i+1  )
            tempfig=x_rec
            plt.imshow(tempfig[figset[i]].reshape(ngf, ndf*multiple))
            plt.gray()
            if i==0:
                plt.ylabel('Reconstructed')
            else:             
                ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
    #            plt.title(str(sub_dim_pool[sublen ]))
                
            ax = plt.subplot(fig_row, n, i + 1+n)
            plt.imshow(x_test_org[figset[i]].reshape(ngf, ndf*multiple))
            plt.gray()
            if i==0:
                plt.ylabel('Original')
            else:            
                ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
    #        plt.title('Original')
    
        elif nc==3:
            for sublen in range(0,len(sub_dim_pool)):
                ax =plt.subplot(fig_row, n, i + 1+sublen*n)
                tempfig=sub_dim_rec[sublen]
                temp=tempfig[figset[i]]
                temp1=np.zeros((ngf, ndf,nc))
                for chan in range (0,nc):
                    temp1[:,:,chan]=temp[chan,:,:]
                plt.imshow(temp1)
    #            plt.gray()
                if i==0:
                    plt.ylabel('Reconstructed')
                else:            
                    ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
    #            plt.title(str(sub_dim_pool[sublen ]))
    
            ax = plt.subplot(fig_row, n, i + 1+fig_row*n-n)     
            temp=x_test_org[figset[i]]
            temp1=np.zeros((ngf, ndf,nc))
            for chan in range (0,nc):
                temp1[:,:,chan]=temp[chan,:,:]
            plt.imshow(temp1)
    #            plt.gray()
            if i==0:
                plt.ylabel('Original')
            else:            
                ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
    #        plt.title('Original')
    #    plt.savefig(test_dataset+'_test_rec_test_epoch_'+str(test_epochs))  
    plt.show()
    
    meas_error=np.zeros((x_rec.shape[0],1))
    for i in range (x_rec.shape[0]):
        y_org=np.abs(np.fft.fft2(x_test_org[i].reshape((ngf,ndf*multiple))).flatten().reshape((1,nc*ngf*ndf*multiple,1))/np.sqrt(nc*ngf*ndf*multiple))
        x_meas=np.abs(np.fft.fft2(x_rec[i].reshape((ngf,ndf*multiple))).flatten().reshape((1,nc*ngf*ndf*multiple,1))/np.sqrt(nc*ngf*ndf*multiple))
        meas_error[i]=np.sum((y_org-x_meas)**2)/np.sum(y_org**2)
    #plt.figure()
    #plt.hist(meas_error,bins='auto')
    #plt.title('Histogram of Normalized Measurement Error for the Reconstructions')
    #plt.xlabel('Normalized Measurement Error')
    #plt.ylabel('Number of Test Samples')
    torch.cuda.empty_cache()
     
    
    res_path='/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/Fourier_phase_retreival/Asilomar/results/'
    for i in range (0,multiple):
        if i==0:
            sub_path=str(test_digits[i])
        else:
            sub_path=sub_path+'_'+str(test_digits[i])
    if not isdir(res_path+sub_path):
        mkdir(res_path+sub_path)
    
    plt.imsave(res_path+sub_path+'/'+'org.png',x_test_org[0,0,:,:])
    plt.imsave(res_path+sub_path+'/'+'rec_known'+str(known_position)+'.png',x_rec[0,0,:,:])
    
end_time = time.time()
print('Time taken:',end_time-start_time)