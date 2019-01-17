import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pdb
from mydataset import MyDataset
from utils import initialize_weights
import torch.autograd as autograd
import cv2
from math import cos,sin

writer = SummaryWriter(log_dir='logs')
class G(nn.Module):
    #[1,1] => [64,64]
    def __init__(self, gpu_mode, init_channels=1024):
        #Layer Normalization
        super(G, self).__init__()
        self.pos = [0 for i in range(31)]
        self.pos[0] = torch.FloatTensor([-11.16231, 1.71701, 9.612973e-06]).cuda().reshape(3,1)
        self.pos[23] = torch.FloatTensor([11.23972, -1.101868, 6.513383]).cuda().reshape(3,1)
        self.pos[27] = torch.FloatTensor([11.23971, -1.101831, -6.513384]).cuda().reshape(3,1)
        self.pos[1]  = torch.FloatTensor([-12.52701, -0.008990765, -2.464731e-08]).cuda().reshape(3,1)
        
        self.pos[24] = torch.FloatTensor([-32.66669, 1.192093e-07, 0]).cuda().reshape(3,1)
        self.pos[28] = torch.FloatTensor([-32.66669, 0, 0]).cuda().reshape(3,1)
        self.pos[2]  = torch.FloatTensor([-11.69302, -0.004831314, -1.340959e-08]).cuda().reshape(3,1)
        
        self.pos[25] = torch.FloatTensor([-36.31458, -9.536743e-07, 0]).cuda().reshape(3,1)
        self.pos[29] = torch.FloatTensor([-36.31458, -4.768372e-07, -4.768372e-07]).cuda().reshape(3,1)
        self.pos[3]  = torch.FloatTensor([-7.388115, -2.221376, 6.07915e-06]).cuda().reshape(3,1)
        
        self.pos[26] = torch.FloatTensor([-8.059542, 8.277123, 0]).cuda().reshape(3,1)
        self.pos[30] = torch.FloatTensor([-8.059542, 8.277124, -4.768372e-07]).cuda().reshape(3,1)
        self.pos[4]  = torch.FloatTensor([-5.895767, 0.09526443, 0.0003307685]).cuda().reshape(3,1)
        self.pos[5]  = torch.FloatTensor([4.509056, 1.610937, 5.190379]).cuda().reshape(3,1)
        self.pos[14] = torch.FloatTensor([4.509048, 1.610966, -5.190396]).cuda().reshape(3,1)
        
        self.pos[6]  = torch.FloatTensor([-5.375591, 0, 7.629395e-06]).cuda().reshape(3,1)
        self.pos[15] = torch.FloatTensor([-5.37559,  0, 0]).cuda().reshape(3,1)
        
        self.pos[7]  = torch.FloatTensor([-18.61573, -9.536743e-07, -7.629395e-06]).cuda().reshape(3,1)
        self.pos[16] = torch.FloatTensor([-18.61573,  0, 0]).cuda().reshape(3,1)
        
        self.pos[8]  = torch.FloatTensor([-18.90229, 0, 0]).cuda().reshape(3,1)
        self.pos[17] = torch.FloatTensor([-18.9023, -1.907349e-06, 0]).cuda().reshape(3,1)
        
        self.pos[9]  = torch.FloatTensor([-1.719231, 0.5651627, -1.910677]).cuda().reshape(3,1)
        self.pos[10] = torch.FloatTensor([-5.540304, -0.2278214,-1.871119]).cuda().reshape(3,1)
        self.pos[11] = torch.FloatTensor([-5.579481, -0.4572525,-0.4858208]).cuda().reshape(3,1)
        self.pos[12] = torch.FloatTensor([-5.428627, -0.4348068,0.7903614]).cuda().reshape(3,1)
        self.pos[13] = torch.FloatTensor([-5.161129, -0.2240829,1.925165]).cuda().reshape(3,1)
        self.pos[18] = torch.FloatTensor([-1.719234, 0.5651703, 1.910675]).cuda().reshape(3,1)
        self.pos[19] = torch.FloatTensor([-5.540302, -0.2278137,1.871119]).cuda().reshape(3,1)
        self.pos[20] = torch.FloatTensor([-5.579479, -0.4572525,0.4858189]).cuda().reshape(3,1)
        self.pos[21] = torch.FloatTensor([-5.428623, -0.4348068,-0.7903633]).cuda().reshape(3,1)
        self.pos[22] = torch.FloatTensor([-5.161131, -0.2240753,-1.925165]).cuda().reshape(3,1)
        self.minx = -81.36
        self.maxx = 79.6
        self.miny = -45.0
        self.maxy = 72.1
        self.minz = -55.1
        self.maxz = 53.8
        self.gpu_mode     = gpu_mode
        self.ekernel_size = 4
        self.pad    = nn.ConstantPad2d((1, 1, 1, 1), 0) #pad on the top and bottom

        self.convt1 = nn.ConvTranspose2d(in_channels=100, out_channels=init_channels, kernel_size=(2,2), stride=1, padding=0)
        self.in1    = nn.InstanceNorm2d(init_channels)

        self.convt2 = nn.ConvTranspose2d(in_channels=init_channels, out_channels=init_channels//2, kernel_size=4, stride=2, padding=1)
        self.in2    = nn.InstanceNorm2d(init_channels//2)

        self.convt3 = nn.ConvTranspose2d(in_channels=init_channels//2, out_channels=init_channels//4, kernel_size=4, stride=2, padding=1)
        self.in3    = nn.InstanceNorm2d(init_channels//4)

        self.convt4 = nn.ConvTranspose2d(in_channels=init_channels//4, out_channels=init_channels//8, kernel_size=4, stride=2, padding=1)
        self.in4    = nn.InstanceNorm2d(init_channels//8)

        self.convt5 = nn.ConvTranspose2d(in_channels=init_channels//8, out_channels=init_channels//16, kernel_size=4, stride=2, padding=1)
        self.in5    = nn.InstanceNorm2d(init_channels//16)

        self.convt6 = nn.ConvTranspose2d(in_channels=init_channels//16, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.relu   = nn.ReLU(True)
        self.tanh    = nn.Tanh()
        initialize_weights(self)

    def getM(self, x):
        theta = (x ** 2).sum(1)**0.5
        R = x/theta.unsqueeze(1).repeat(1,3)
        last = torch.zeros((R.shape[0],3,3)).float().cuda()
        last[:,0,1] = -R[:,2]
        last[:,0,2] =  R[:,1]
        last[:,1,0] =  R[:,2]
        last[:,1,2] = -R[:,0]
        last[:,2,0] = -R[:,1]
        last[:,2,1] =  R[:,0]
        return (torch.cos(theta).unsqueeze(1).unsqueeze(1) * torch.eye(3).repeat(R.shape[0],1).reshape(R.shape[0],3,3).cuda() + (1-torch.cos(theta)).unsqueeze(1).unsqueeze(1) * R.unsqueeze(2).matmul(R.unsqueeze(1)) + torch.sin(theta).unsqueeze(1).unsqueeze(1)*last).permute(0,2,1)
        
    def getPos(self, x):
        #pdb.set_trace()
        x = x.squeeze(1) #[-1,1]
        y = x.reshape(x.shape[0]*x.shape[1],x.shape[2])

        #pdb.set_trace()
        out = torch.zeros((x.shape[0]*x.shape[1],93)).cuda() #内部18个到所有节点31个
        
        cot = 1
        M0 = self.getM(y[:,30:33]) #spine
        P0 = self.pos[0].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1) #[shape[0]*shape[1],3,1]
        M  = M0
        P  = P0
        out[:,:cot*3] = P.squeeze(2)
        for i in range(9,6,-1): #spine1 -> spine2 -> neck
            P = M.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P
            M = M.matmul(self.getM(y[:,i*3:i*3+3]))
            cot += 1
            out[:,(cot-1)*3:cot*3] = P.squeeze(2)
        
        P1 = M.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P #head
        cot+=1
        out[:,(cot-1)*3:cot*3] = P1.squeeze(2)

        M2 = M
        P2 = P
        for i in range(6,2,-1): #L clavicle => L hand
            P2 = M2.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P2
            M2 = M2.matmul(self.getM(y[:,i*3:i*3+3]))
            cot += 1
            out[:,(cot-1)*3:cot*3] = P2.squeeze(2)
        
        for i in range(5): #L Finger 0 - 4
            Pt = M2.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P2
            cot += 1
            out[:,(cot-1)*3:cot*3] = Pt.squeeze(2)
        
        P3 = P
        M3 = M
        for i in range(11,15,1):#R Clavicle -> R hand
            P3 = M3.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P3
            M3 = M3.matmul(self.getM(y[:,i*3:i*3+3]))
            cot += 1
            out[:,(cot-1)*3:cot*3] = P3.squeeze(2)
        
        for i in range(5): #R Finger 0 - 4
            Pt = M3.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P3
            cot += 1
            out[:,(cot-1)*3:cot*3] = Pt.squeeze(2)
        
        M4 = M0
        P4 = P0
        for i in range(2,-1,-1): #L Thigh -> L Foot
            P4 = M4.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P4
            M4 = M4.matmul(self.getM(y[:,i*3:i*3+3]))
            cot += 1
            out[:,(cot-1)*3:cot*3] = P4.squeeze(2)
        
        Pt = M4.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P4 #L Toe
        cot+=1
        out[:,(cot-1)*3:cot*3] = Pt.squeeze(2)

        P5 = P0
        M5 = M0
        for i in range(15,18,1): #R Thigh -> R Foot
            P5 = M5.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P5
            M5 = M5.matmul(self.getM(y[:,i*3:i*3+3]))
            cot += 1
            out[:,(cot-1)*3:cot*3] = P5.squeeze(2)
        
        Pt = M5.matmul(self.pos[cot].repeat(x.shape[1]*x.shape[0],1).reshape(-1,3,1)) + P5 #R Toe
        cot+=1
        out[:,(cot-1)*3:cot*3] = Pt.squeeze(2)
        return out.reshape(x.shape[0],x.shape[1],-1).unsqueeze(1)

    def forward(self, x):
        x = self.relu(self.in1(self.convt1(x)))
        x = self.relu(self.in2(self.convt2(x)))
        x = self.relu(self.in3(self.convt3(x)))
        x = self.relu(self.in4(self.convt4(x)))
        x = self.relu(self.in5(self.convt5(x)))
        x = self.tanh(self.convt6(x))
        #pdb.set_trace()
        x = self.getPos(x)
        return x

class D(nn.Module):
    #[93,64] => pad => [96,64]
    def __init__(self, gpu_mode, init_channels=64):
        super(D, self).__init__()
        self.gpu_mode     = gpu_mode
        self.ekernel_size = 4
        self.pad    = nn.ConstantPad2d((1, 2, 0, 0), 0) #pad on the top and bottom
        #input:[bs,1,160,320]
        self.conv1  = nn.Conv2d(in_channels=1, out_channels=init_channels, kernel_size=4, stride=2, padding=1)
        
        self.conv2  = nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=4, stride=2, padding=1)
        self.in2    = nn.InstanceNorm2d(init_channels*2)

        self.conv3  = nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=4, stride=2, padding=1)
        self.in3    = nn.InstanceNorm2d(init_channels*4)

        self.conv4  = nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=4, stride=2, padding=1)
        self.in4    = nn.InstanceNorm2d(init_channels*8)

        self.conv5  = nn.Conv2d(in_channels=init_channels*8, out_channels=init_channels*16, kernel_size=4, stride=2, padding=1)
        self.in5    = nn.InstanceNorm2d(init_channels*16)

        self.conv6  = nn.Conv2d(in_channels=init_channels*16, out_channels=1, kernel_size=(2,3), stride=1, padding=0)

        self.relu    = nn.ReLU(True)
        self.Lrelu   = nn.LeakyReLU(0.2)
        initialize_weights(self)

    def forward(self, x):
        #pdb.set_trace()
        x = self.pad(x)
        x = self.Lrelu(self.conv1(x))
        x = self.Lrelu(self.in2(self.conv2(x)))
        x = self.Lrelu(self.in3(self.conv3(x)))
        x = self.Lrelu(self.in4(self.conv4(x)))
        x = self.Lrelu(self.in5(self.conv5(x)))
        x = self.conv6(x)
        return x

def calc_gradient_penalty(netD, current_batch_size , gpu_model, real_data, fake_data, lambda_):
    #print real_data.size()
    
    alpha = torch.rand(current_batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if gpu_model else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    
    interpolates = interpolates.cuda() if gpu_model else interpolates
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if gpu_model else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.view(gradients.shape[0], -1).norm(2, 1) - 1) ** 2).mean() * lambda_ #K = 1
    return gradient_penalty

def saveImage(data, th):
    data = (data * 2.55).astype('uint8').squeeze()
    dt  = 10
    out = np.zeros((data.shape[0], data.shape[1] * dt)).astype('uint8')
    for j in range(data.shape[1]):
        for i in range(dt):
            out[:,j*dt+i] = data[:,j]
    cv2.imwrite('./results/fake_epoch_%d.png' % th, out)

def test(model):
    z_ = torch.rand((1, 100, 1, 1)).float().cuda()
    G_ = model(z_)

    x  = np.zeros(G_.shape)
    x  = G_.squeeze().cpu().detach().numpy()
    return x #one sample
    

class WGAN_GP(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.pre_train_model = args.model
        self.output_dim = 8
        self.z_dim = 100
        self.max_length = 64
        self.lambda_ = 10
        self.n_critic = 5               # the number of iterations of the critic per generator iteration

        # load dataset
        self.dataset = MyDataset()
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.iter = self.data_loader.__iter__()

        # networks init
        self.G = G(self.gpu_mode)
        self.D = D(self.gpu_mode)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.pre_train_model:
            dic_D = torch.load(self.pre_train_model)
            dic_G = torch.load(self.pre_train_model[:self.pre_train_model.rfind('D')] + 'G.pth')
            self.G.load_state_dict(dic_G['state_dict'])
            self.G_optimizer.load_state_dict(dic_G['optimizer'])
            self.D.load_state_dict(dic_D['state_dict'])
            self.D_optimizer.load_state_dict(dic_D['optimizer'])

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        print('training start!!')
        self.G.train()
        self.D.train()
        for epoch in range(self.epoch):
            self.iter = self.data_loader.__iter__()
            epoch_start_time = time.time()
            step = 0
            
            while step <= self.iter.__len__():
                for i in range(self.n_critic):
                    step += 1
                    if step > self.iter.__len__():
                        break
                    
                    x_ = self.iter.next()
                    x_ = x_.unsqueeze(1).float()
                    z_ = torch.rand((x_.shape[0], self.z_dim, 1, 1)).float()
                    if self.gpu_mode:
                        x_, z_ = x_.cuda(), z_.cuda()

                    # update D network
                    self.D_optimizer.zero_grad()
                    D_real = self.D(x_)

                    D_real_loss = -torch.mean(D_real)

                    G_ = self.G(z_)

                    D_fake = self.D(G_)
                    D_fake_loss = torch.mean(D_fake)

                    gradient_penalty = calc_gradient_penalty(self.D, z_.shape[0], self.gpu_mode, x_.data, G_.data, self.lambda_)
                    D_loss = D_real_loss + D_fake_loss + gradient_penalty
                    W_distance = -D_real_loss - D_fake_loss
                    #print('D weight conv1:', self.D.conv1.state_dict()['weight'].min().item(), ' ',self.D.conv1.state_dict()['weight'].max().item())
                    #print('D weight conv2:', self.D.conv2.state_dict()['weight'].min().item(), ' ',self.D.conv2.state_dict()['weight'].max().item())
                    #print('D weight conv3:', self.D.conv3.state_dict()['weight'].min().item(), ' ',self.D.conv3.state_dict()['weight'].max().item())
                    #print('D weight conv4:', self.D.conv4.state_dict()['weight'].min().item(), ' ',self.D.conv4.state_dict()['weight'].max().item())
                    #print('D weight conv5:', self.D.conv5.state_dict()['weight'].min().item(), ' ',self.D.conv5.state_dict()['weight'].max().item())
                    #print('D weight conv6:', self.D.conv6.state_dict()['weight'].min().item(), ' ',self.D.conv6.state_dict()['weight'].max().item())
                    
                    
                    D_loss.backward()

                    print('D_real_loss ',D_real_loss.item(),' D_fake_loss ',D_fake_loss.item(),' gradient_penalty ',gradient_penalty.item())
                    self.D_optimizer.step()

                z_ = torch.rand((32, self.z_dim, 1, 1)).float()
                z_ = z_.cuda() if self.gpu_mode else z_

                # update G network
                self.G_optimizer.zero_grad()
                G_ = self.G(z_)

                D_fake = self.D(G_)
                G_loss = -torch.mean(D_fake)
                
                G_loss.backward()
                self.G_optimizer.step()

                print("Epoch: [%2d] [%4d/%4d] Wasserstein distance: %.8f, D_loss: %.8f, G_loss: %.8f" %
                    ((epoch + 1), step, self.data_loader.dataset.__len__() // self.batch_size, W_distance.item(), D_loss.item(), G_loss.item()))
                
                writer.add_scalar('data/G_loss', G_loss, step)
                writer.add_scalar('data/D_real_loss', D_real_loss, step)
                writer.add_scalar('data/D_fake_loss', D_fake_loss, step)
                writer.add_scalar('data/gradient_penalty_loss', gradient_penalty, step)
                writer.add_scalar('data/D_loss', D_loss, step)
                writer.add_scalar('data/Wasserstein distance', W_distance, step)

            self.save('epoch_%d_WD_%.4f_Gloss_%.4f_Dloss_%.4f' % (epoch,W_distance,G_loss,D_loss))
            print('epoch over, total cost %.4fs' % float(time.time()-epoch_start_time))
        print("Training finish!... save training results")

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self, m):
        save_dir = './%s/%s' % (self.save_dir, m)
        torch.save({'state_dict': self.G.state_dict(), 'optimizer': self.G_optimizer.state_dict()}, save_dir + '_G.pth')
        torch.save({'state_dict': self.D.state_dict(), 'optimizer': self.D_optimizer.state_dict()}, save_dir + '_D.pth')

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))