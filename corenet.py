import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from utils import UWDataset, UWDataModule,init_weights

import torch
from torch import nn
from CN_Arch_details import Upsample,Downsample,Apprentice,Master
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils import reshaper, plot_loss_fig, write_loss, PSNR_loss,calculate_SNR,snr_loss, ImageTransform, calculate_PSNR,denorm_image,plot_images,psnr_loss
from focal_frequency_loss import FocalFrequencyLoss as FFL
ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class


OUTPUT_FOLDERS = [
    "plots",
    "plotsres",
    "tr_figs",
    "val_figs",
    "weights",
    "weightsmaster"
]

for folder in OUTPUT_FOLDERS:
    if os.path.exists(folder):
        shutil.rmtree(folder)


for folder in OUTPUT_FOLDERS:
    os.makedirs(folder, exist_ok=True)

for logfile in ["TrainLoss.txt", "ValLoss.txt"]:
    if os.path.exists(logfile):
        os.remove(logfile)

vsnr_sig_loss=[]




val_mas_loss_fkloss=[]





trsnr_sig_loss=[]



tr_mas_loss_fk_loss=[]


testpsnr=[]
realpsnr=[]
mean_ma_list = []
mean_psnr_list = []
mean_real_psnr_list= []



data_dir = "train_set"


# Train
transform = ImageTransform(img_size=256)
dmh = UWDataModule(data_dir,transform, batch_size=1, phase='test')
dmh.prepare_data()
trdataloader= dmh.train_dataloader()


# Val
data_dir_hq = "val_set"
dm2_hq = UWDataModule(data_dir_hq, transform, batch_size=1, phase='test')
dm2_hq.prepare_data()
valdataloader = dm2_hq.train_dataloader()





App = Apprentice().cuda()



Ma = Master().cuda()


    
num_epoch = 10000
lr=0.00001
betas=(0.5, 0.999)
Eps=3
Beta=0.05
Phi=100


App_params = list(App.parameters())


Ma_params = list(Ma.parameters())

optimizer_g = torch.optim.Adam(App_params, lr=lr, betas=betas)


optimizer_d = torch.optim.Adam(Ma_params, lr=lr*2, betas=betas)

criterion_mae = psnr_loss
criterion_mse = nn.MSELoss()
criterion_bce = nn.L1Loss()
interval=10
targetpsnr=40 
E=0.00001    
 

def validate_app_and_ma(dataloader, App, Ma, criterion_bce, criterion_mae, targetpsnr, Eps, Beta, Phi):
 app_loss_epoch = 0
 mas_loss_epoch = 0
 mas_loss_gt_epoch = 0
 mas_loss_fk_epoch = 0
 mas_loss_cor_epoch = 0
 snr_sig_epoch = 0
 snr_spec_epoch = 0
 
 App.eval()
 Ma.eval()
 
 with torch.no_grad():
     for input_img, real_img in dataloader:
         input_img = input_img.cuda()
         real_img = real_img.cuda()
         real_label = torch.ones(input_img.size()[0], 1, 1).cuda()

         fake_img = App(input_img)
         fake_img_ = fake_img.detach() 
         out_fake = Ma(fake_img)

         alpha = calculate_PSNR(real_img.detach().cpu().numpy(), fake_img.detach().cpu().numpy())
         alpha_hat = alpha / targetpsnr
         theta = calculate_PSNR(real_img.detach().cpu().numpy(), input_img.detach().cpu().numpy())
         theta_hat = theta / targetpsnr

         loss_g_bce = criterion_bce(out_fake, real_label)
         loss_g_mae = criterion_mae(fake_img, real_img)
         loss_g_dim = ffl(fake_img, real_img)
         loss_g = Eps * loss_g_bce + Beta * loss_g_mae + Phi * loss_g_dim

         out_real = Ma(real_img)
         loss_d_real = criterion_bce(out_real, real_label)
         loss_d_fake = criterion_bce(out_fake, real_label * alpha_hat)
         out_input = Ma(input_img)
         loss_m_cor = criterion_bce(out_input, real_label * theta_hat)

         loss_d = loss_d_real + loss_d_fake
         master_gt_l = targetpsnr * loss_d_real
         master_fk_l = targetpsnr * loss_d_fake
         master_tt_l = targetpsnr * loss_d
         master_cc_l = targetpsnr * loss_m_cor

         app_loss_epoch += loss_g.item()
         mas_loss_epoch += master_tt_l.item()
         mas_loss_gt_epoch += master_gt_l.item()
         mas_loss_fk_epoch += master_fk_l.item()
         mas_loss_cor_epoch += master_cc_l.item()
         snr_sig_epoch += loss_g_mae.item()
         snr_spec_epoch += loss_g_dim.item()

 return app_loss_epoch / len(dataloader), mas_loss_epoch / len(dataloader), \
        mas_loss_gt_epoch / len(dataloader), mas_loss_fk_epoch / len(dataloader), \
        mas_loss_cor_epoch / len(dataloader), snr_sig_epoch / len(dataloader), \
        snr_spec_epoch / len(dataloader)













import matplotlib.pyplot as plt

def plot_vsnr_sig_losses(vsnr_sig_hq_loss, epoch, interval, dataset_type='Train', plot_type='PSNR'):
    
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    if dataset_type == 'Train':
        x_axis = list(range(1, len(vsnr_sig_hq_loss) + 1))  # Use epochs for training
    else:
        x_axis = [i * interval for i in range(1, len(vsnr_sig_hq_loss) + 1)]  # Use intervals for validation

    axes.plot(x_axis, vsnr_sig_hq_loss, label='HQ Loss', color='b')
    axes.set_title(f'{dataset_type} - Loss')
    axes.set_xlabel('Epochs' if dataset_type == 'Train' else 'Interval')
    axes.set_ylabel('Loss')
    axes.grid(True)
    axes.legend()

  
    plt.tight_layout()

    plt.savefig(f'plots/{dataset_type}_{plot_type}.png')

    plt.show()
    plt.close()








def train_epoch(dataloader, App, Ma, criterion_bce, criterion_mae, ffl, Eps, Beta, Phi, targetpsnr, optimizer_g, optimizer_d):
    predicted = []
    inpdata = []
    GT = []
    tr_apprentice_loss = []
    tr_master_loss = []
    tr_master_loss_fk = []
    tr_master_loss_gt = []
    tr_master_loss_cor = []
    signal_snr = []
    spec_snr = []
    psnr_list = []

  

    for input_img, real_img in dataloader:
        if 0:  # check beats
            # check beats
            plt.subplot(211)
            plt.imshow(denorm_image(input_img[0, :, :, :].cpu().detach()))
            plt.title("Corrupted/Clean")
            plt.subplot(212)
            plt.imshow(denorm_image(real_img[0, :, :, :].cpu().detach()))

        input_img = input_img.cuda()
        real_img = real_img.cuda()
        real_label = torch.ones(input_img.size()[0], 1, 1).cuda()

        # Apprentice Loss
        fake_img = App(input_img).cuda()
       
        
        
    

        fake_img_ = fake_img.detach()
        out_fake = Ma(fake_img).cuda()

        alpha = calculate_PSNR(real_img.detach().cpu().numpy(), fake_img.detach().cpu().numpy())
        alpha_hat = alpha / targetpsnr

        theta = calculate_PSNR(real_img.detach().cpu().numpy(), input_img.detach().cpu().numpy())
        theta_hat = theta / targetpsnr

        loss_g_bce = criterion_bce(out_fake, real_label)
        loss_g_mae = criterion_mae(fake_img, real_img)
        loss_g_dim = ffl(fake_img, real_img)  # calculate focal frequency loss
        loss_g = Eps * loss_g_bce + Beta * loss_g_mae + Phi * loss_g_dim

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # Master Loss
        out_real = Ma(real_img)
        loss_d_real = criterion_bce(out_real, real_label)
        out_fake = Ma(fake_img_)
      

        loss_d_fake = criterion_bce(out_fake, real_label * alpha_hat)
        out_input = Ma(input_img)

   
        loss_d = loss_d_real + loss_d_fake 
        master_gt_l = targetpsnr * loss_d_real
        master_fk_l = targetpsnr * loss_d_fake
        master_tt_l = targetpsnr * loss_d

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

    return {
        "predicted": predicted,
        "inpdata": inpdata,
        "GT": GT,
        "tr_apprentice_loss": tr_apprentice_loss,
        "tr_master_loss": tr_master_loss,
        "tr_master_loss_fk": tr_master_loss_fk,
        "tr_master_loss_gt": tr_master_loss_gt,
        "tr_master_loss_cor": tr_master_loss_cor,
        "signal_snr": signal_snr,
        "spec_snr": spec_snr,
        "psnr_list": psnr_list,
    }






for e in range(1, num_epoch):
    print("Epoch: "+str(e))
    App.train()
   
    Ma.train()
    results = train_epoch(
            dataloader=trdataloader,
            App=App,
           
            Ma=Ma,
            criterion_bce=criterion_bce,
            criterion_mae=criterion_mae,
            ffl=ffl,
            Eps=Eps,
            Beta=Beta,
            Phi=Phi,
            targetpsnr=targetpsnr,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d
        )
  
    
    
 
    
    if e % 1 == 0:
        
        App.eval()
       
        Ma.eval()
      

        app_loss_hq, mas_loss_hq, val_mas_loss_gt_hq, val_mas_loss_fk_hq, val_mas_loss_cor_hq, vsnr_sig_hq, vsnr_spec_hq = \
            validate_app_and_ma(trdataloader, App, Ma, criterion_bce, criterion_mae, targetpsnr, Eps, Beta, Phi)


        trsnr_sig_loss.append(vsnr_sig_hq)
        tr_mas_loss_fk_loss.append(val_mas_loss_fk_hq)
      
        
        plot_vsnr_sig_losses(trsnr_sig_loss, e, interval=10, dataset_type='Train',plot_type="PSNR")
        plot_vsnr_sig_losses(tr_mas_loss_fk_loss, e, interval=1, dataset_type='Train',plot_type="Master_Generated")

    if e % interval == 0:
        App.eval()
      
        Ma.eval()

        app_loss_hq, mas_loss_hq, val_mas_loss_gt_hq, val_mas_loss_fk_hq, val_mas_loss_cor_hq, vsnr_sig_hq, vsnr_spec_hq = \
            validate_app_and_ma(valdataloader, App, Ma, criterion_bce, criterion_mae, targetpsnr, Eps, Beta, Phi)

       

        vsnr_sig_loss.append(vsnr_sig_hq)
        val_mas_loss_fkloss.append(val_mas_loss_fk_hq)
       
        
        plot_vsnr_sig_losses(vsnr_sig_loss,  e, interval=10, dataset_type='Val',plot_type="PSNR")
        plot_vsnr_sig_losses(val_mas_loss_fkloss,  e, interval=10, dataset_type='Val',plot_type="Master_Generated")


            
        torch.save(App.state_dict(), 'weights/hq_weights_'+str(e)+'_.pth')
       
        torch.save(Ma.state_dict(), 'weights/master_weights_'+str(e)+'_.pth')
            
