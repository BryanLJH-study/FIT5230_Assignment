import torch
import os
from torch import optim, nn
from DeepImageFingerprinting.modules.utils import *
import DeepImageFingerprinting.modules.model as model
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import crop
import torch.nn.functional as F
from torch.utils.data import Subset
from PIL import Image
import numpy as np

# CNN Model for Denoising
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

        self._initialize_weights()

        self.prnu = None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.dncnn(x)
        return out

    def denoise(self, x):
        x = F.pad(x, (10, 10, 10, 10))
        res = self.dncnn(x)[:, :, 10:-10, 10:-10]

        if res.size()[2] != self.prnu.size()[2] or res.size()[3] != self.prnu.size()[3]:
            return res
        else:
            return res - self.prnu
        
# Class for training the denoising CNN model (a pretrained one will be used)
class TrainerDnCNN(nn.Module):
    def __init__(self, hyperparams):
        super(TrainerDnCNN, self).__init__()

        self.device = hyperparams['Device']
        self.init_lr = hyperparams['LR']
        self.ch_o = hyperparams['Out. Channel']
        self.m = hyperparams['Margin']
        self.batch_size = hyperparams['Batch Size']
        self.crop_size = hyperparams['Crop Size']
        self.depth = hyperparams['Depth']
        self.crop_b = hyperparams['Crop Batch']

        self.train_loss = []

        self.train_mean_r = []
        self.train_mean_f = []

        self.test_mean_r = []
        self.test_mean_f = []

        self.denoiser = DnCNN(self.ch_o, self.depth).to(self.device)
        self.optimizer = optim.AdamW(self.denoiser.parameters(), lr=self.init_lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.998)

        self.loss_fun = nn.MSELoss(reduction='sum')
        self.loss_bce = nn.BCELoss(reduction='none')


def load_model(trainer, path, device):
    if device.type == 'cpu':
        data_dict = torch.load(path, map_location=torch.device('cpu'))
    else:
        data_dict = torch.load(path)

    try:
        trainer.unet.load_state_dict(data_dict['G state'])
        trainer.train_loss = data_dict['Train G Loss']

        return len(trainer.train_loss)
    except:
        return data_dict
    
# Loading the denoiser
import os
root = Path("dncnn")

def load_denoiser(device: str, trainable:bool=False):
    os.chdir('./DeepImageFingerprinting')
    denoiser_prnu_np = np.load(str(root / r"clean_real.npy"), allow_pickle=True)
    trainer = load_model(TrainerDnCNN, Path("dncnn/chk_2000.pt"), device)
    os.chdir('../')
    model = trainer.denoiser.to(device)

    denoiser_prnu = torch.tensor(denoiser_prnu_np.transpose((2, 0, 1))).to(device).unsqueeze(0)
    model.prnu = denoiser_prnu

    if not trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model

# Helper function to calculate distance between two points
def distance(arr, mu_a, mu_b):
    dist_arr2a = np.sqrt(((arr - mu_a) ** 2)).reshape((-1, 1))
    dist_arr2b = np.sqrt(((arr - mu_b) ** 2)).reshape((-1, 1))
    return np.concatenate((dist_arr2a, dist_arr2b), axis=1)

# Helper function to initiate dummy noise model for the first training step
def init_dummy(bs, noise_type, img_dims, ch_n, var=0.1):
    if noise_type == 'uniform':
        img = var * torch.rand((bs, ch_n, img_dims[0], img_dims[1]))
    elif noise_type == 'normal':
        img = var * torch.randn((bs, ch_n, img_dims[0], img_dims[1]))
    elif noise_type == 'mesh':
        assert ch_n == 2
        X, Y = np.meshgrid(np.arange(0, img_dims[1]) / float(img_dims[1] - 1),
                        np.arange(0, img_dims[0]) / float(img_dims[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        img = torch.tensor(meshgrid).unsqueeze(0).type(torch.float)

    elif noise_type == 'special':
        X, Y = np.meshgrid(np.arange(0, img_dims[1]) / float(img_dims[1] - 1),
                        np.arange(0, img_dims[0]) / float(img_dims[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        img = torch.tensor(meshgrid).unsqueeze(0).type(torch.float)
        img = torch.cat((img, torch.ones((1, 1, img_dims[0], img_dims[1]))), dim=1)
    return img

# Deep Image Fingerprinting model using UNet
class DIFModel(nn.Module):
    def __init__(self, hyperparams):
        super(DIFModel, self).__init__()

        self.device = hyperparams['Device']
        self.init_lr = hyperparams['LR']
        self.ch_i = hyperparams['Inp. Channel']
        self.ch_o = hyperparams['Out. Channel']
        self.arch = hyperparams['Arch.']
        self.depth = hyperparams['Depth']
        self.concat = np.array(hyperparams['Concat'])
        self.m = hyperparams['Margin']
        self.batch_size = hyperparams['Batch Size']
        self.alpha = hyperparams['Alpha']

        self.train_loss = []
        self.train_corr_r = None
        self.train_corr_f = None

        self.test_loss = []
        self.test_corr_r = []
        self.test_corr_f = []
        self.test_labels = []

        self.noise_type = hyperparams['Noise Type']
        self.noise_std = hyperparams['Noise STD']
        self.noise_channel = hyperparams['Inp. Channel']
        self.crop_size = hyperparams['Crop Size']

        d_h, n_h, d_w, n_w = calc_even_size(self.crop_size, self.depth)
        self.crop_size = (n_h - d_h, n_w - d_w)

        self.noise = None
        self.denoiser = load_denoiser(self.device)
        self.unet = model.Unet(self.device, self.ch_i, self.ch_o, self.arch,
                               activ='leak', depth=self.depth, concat=self.concat).to(self.device)
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.init_lr)
        self.loss_mse = nn.MSELoss()

        self.init_train()

    def train_step(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.unet.train()
        self.optimizer.zero_grad()

        residuals = self.denoiser.denoise(images).detach()
        alpha = (1 - self.alpha) * torch.rand((len(images), 1, 1, 1)).to(self.device) + self.alpha
        residuals = alpha * residuals

        f_mean = residuals[~labels].mean(0, keepdims=True)
        r_mean = residuals[labels].mean(0, keepdims=True)
        residuals = torch.cat((residuals, f_mean, r_mean), dim=0)

        dmy = self.prep_noise().to(self.device)
        out = self.unet(dmy).repeat(len(images) + 2, 1, 1, 1)

        corr = self.corr_fun(out, residuals)
        loss = self.loss_contrast(corr[:-2].mean((1, 2, 3)), labels).mean() / self.m

        loss.backward()
        self.optimizer.step()

        if self.fingerprint is None:
            self.fingerprint = out[0:1].detach()
        else:
            self.fingerprint = self.fingerprint * 0.99 + out[0:1].detach() * (1 - 0.99)

        corr = self.corr_fun(self.fingerprint.repeat(len(images), 1, 1, 1), residuals[:-2]).mean((1, 2, 3))

        self.train_loss.append(loss.item())

        if self.train_corr_r is None:
            self.train_corr_r = [corr[labels].mean().item()]
            self.train_corr_f = [corr[~labels].mean().item()]
        else:
            corr_r = corr[labels]
            corr_f = corr[~labels]
            self.train_corr_r.append(corr_r.mean().item())
            self.train_corr_f.append(corr_f.mean().item())

    def test_model(self, test_loader):
        self.reset_tests()
        self.calc_centers()

        fingerprint = self.fingerprint.to(self.device)
        fingerprint.repeat((self.batch_size, 1, 1, 1))

        with torch.no_grad():
            for images, labels in iter(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                residuals = self.denoiser.denoise(images).float()

                corr = self.corr_fun(fingerprint, residuals)
                loss = self.loss_contrast(corr.mean((1, 2, 3)), labels) / self.m
                mean_corr = corr.mean((1, 2, 3))

                self.test_loss = self.test_loss + loss.tolist()
                self.test_labels = self.test_labels + labels.tolist()
                
                labels = labels.bool()
                
                if self.test_corr_r is None:
                    self.test_corr_r = mean_corr[labels].cpu().numpy()
                    self.test_corr_f = mean_corr[~labels].cpu().numpy()
                else:
                    self.test_corr_r = np.append(self.test_corr_r, mean_corr[labels].cpu().numpy(), axis=0)
                    self.test_corr_f = np.append(self.test_corr_f, mean_corr[~labels].cpu().numpy(), axis=0)

    def predict_image(self, image):
        self.reset_tests()
        self.calc_centers()

        fingerprint = self.fingerprint.to(self.device)
        fingerprint.repeat((self.batch_size, 1, 1, 1))

        with torch.no_grad():
            image = image.to(self.device)
            residual = self.denoiser.denoise(image).float()

            corr = self.corr_fun(fingerprint, residual)
            mean_corr = corr.mean((1, 2, 3))
            dist = distance(mean_corr[0].cpu().numpy(), self.mu_real, self.mu_fake)

            predicted_class = np.argmin(dist, axis=1)

        return predicted_class, dist

    def norm_val(self, arr):
        return (arr - arr.mean((1, 2, 3)).view(-1, 1, 1, 1)) / (arr.std((1, 2, 3)).view(-1, 1, 1, 1) + 1e-8)

    def init_train(self, n=1):
        self.noise = init_dummy(n, self.noise_type, self.crop_size, self.noise_channel)
        self.fingerprint = None

    def prep_noise(self, var=-1):
        if var == -1:
            return self.noise + torch.randn_like(self.noise.detach()) * self.noise_std
        else:
            return self.noise + torch.randn_like(self.noise.detach()) * var

    def corr_fun(self, out, target):
        out = self.norm_val(out)
        target = self.norm_val(target)

        return out * target

    def loss_contrast(self, corrs, labs):
        n = len(corrs) // 2
        corr_a = corrs[:n]
        lab_a = labs[:n]

        corr_b = corrs[n:]
        lab_b = labs[n:]

        sim_label = torch.bitwise_xor(lab_a, lab_b).type(torch.float64)
        corr_delta = torch.sqrt(((corr_a - corr_b) ** 2))
        loss = sim_label * (self.m - corr_delta) + (1. - sim_label) * corr_delta

        return relu(loss)

    def reset_tests(self):
        self.test_corr_r = None
        self.test_corr_f = None

        self.test_loss = []
        self.test_labels = []

    def produce_fingerprint(self):
        with torch.no_grad():
            out = self.fingerprint[0]
            
        return out.cpu().numpy().transpose((1, 2, 0))

    def show_fingerprint(self):
        finger = self.produce_fingerprint()
        finger = 0.5 * finger + 0.5

        plt.figure(figsize=(4, 4))

        plt.imshow(finger)
        plt.axis(False)
        plt.title('Fingerprint')

        plt.show()

        dct_finger = produce_spectrum(finger)
        dct_finger = (dct_finger - dct_finger.min()) / (dct_finger.max() - dct_finger.min())

        plt.figure(figsize=(4, 4))

        plt.imshow(dct_finger, 'bone')
        plt.axis(False)
        plt.title('Fingerprint FFT')

        plt.show()

    def plot_loss(self):
        plt.scatter(np.arange(1, len(self.train_loss) + 1), self.train_loss, s=3, label='Loss', c='g')
        plt.xlabel('Batch Index')
        plt.ylabel('Mean Sample Loss')
        plt.title('Train Loss')

        plt.grid(True)
        plt.ylim([0., 1.0])
        plt.legend(fontsize=12)
        plt.tight_layout()

        plt.show()

    def plot_corr(self):
        plt.scatter(np.arange(len(self.train_corr_r)), self.train_corr_r, s=3,
                    label='Real Corr.', c='g')
        plt.scatter(np.arange(len(self.train_corr_f)), self.train_corr_f, s=3,
                    label='AI Corr.', c='r')
        plt.xlabel('Batch Index')
        plt.ylabel('Mean Sample Corr.')
        plt.title('Train Correlation')

        plt.grid(True)
        plt.legend(fontsize=12)

        plt.show()

    def calc_centers(self):
        self.mu_real = np.mean(self.train_corr_r[-20:])
        self.mu_fake = np.mean(self.train_corr_f[-20:])

    def calc_distance(self):
        dist_real = distance(self.test_corr_r, self.mu_real, self.mu_fake)
        dist_fake = distance(self.test_corr_f, self.mu_real, self.mu_fake)

        return dist_fake, dist_real

    def calc_accuracy(self):
        dist_real = distance(self.test_corr_r, self.mu_real, self.mu_fake)
        dist_fake = distance(self.test_corr_f, self.mu_real, self.mu_fake)

        class_real = np.argmin(dist_real, axis=1) == 1
        class_fake = np.argmin(dist_fake, axis=1) == 0

        acc_real = class_real.sum() / len(class_real)
        acc_fake = class_fake.sum() / len(class_fake)

        return acc_fake, acc_real
        
    def save_stats(self, path):
        self.calc_centers()

        data_dict = {'Fingerprint': self.fingerprint,
                     'Train Real': self.train_corr_r,
                     'Train Fake': self.train_corr_f,
                     'Loss': self.train_loss}

        torch.save(data_dict, path)

    def load_stats(self, path):
        print(path)
        if self.device.type == 'cpu':
            data_dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            data_dict = torch.load(path)

        self.train_loss = data_dict['Loss']
        self.train_corr_r = data_dict['Train Real']
        self.train_corr_f = data_dict['Train Fake']
        self.fingerprint = data_dict['Fingerprint']

    # Function to predict image based on previously trained model
def predict_image(image, pretrained_model):
    #image = Image.open(image_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop_hyper)
    ])

    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256))
    ])

    image_tensor = transform_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)
    height = 2
    aspect_ratio = image_tensor.shape[1] / image_tensor.shape[0]
    width = height * aspect_ratio

    plt.figure(figsize=(width, height))
    plt.imshow(image)
    plt.axis('off')

    preprocess_image = preprocess(image)
    preprocess_image = preprocess_image.unsqueeze(0)

    prediction_class, prediction_dist = pretrained_model.predict_image(preprocess_image)
    real_dist = round(prediction_dist[0][1], 4)
    ai_dist = round(prediction_dist[0][0], 4)
    if prediction_class[0] == 1:
        #print("Prediction: Real Image", "\nDistance to Real Center:", real_dist, "\nDistance to AI Center:", ai_dist)
        return 1
    else:
        #print("Prediction: AI Generated Image", "\nDistance to Real Center:", real_dist, "\nDistance to AI Center:", ai_dist)
        return 0

def crop_hyper(image):
    img_size = (image.shape[1], image.shape[2])
    crop_w, crop_h = (256, 256)
    top = (img_size[0] - crop_h) // 2
    left = (img_size[1] - crop_w) // 2
    return crop(image, top, left, crop_h, crop_w)