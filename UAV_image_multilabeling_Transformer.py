##### Transformer Network for multi-label classification for UAV Imagery

import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.nn import Sequential, Linear
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
from skimage import io
from sklearn.metrics import confusion_matrix
import cv2
import albumentations as albu
import csv, pandas
import spacy
import matplotlib.pyplot as plt
import re
import timeit
from PIL import Image

from transformer_util_multilabel import SupConLoss, SimclrCriterion, F1_Loss
from transformer_util_multilabel import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoderLayer,TransformerDecoderLayer,TransformerDecoder
from Metrics_Multilabel import Accuracy_Multilabel, metrics_function

from deit.models import deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
from deit.models import deit_tiny_distilled_patch16_224, deit_base_distilled_patch16_224, deit_base_distilled_patch16_384
from pytorch_pretrained_vit import ViT


start = timeit.default_timer()

### Check the accuracy of the multilabel model
def Check_accuracy_Multilabel(Model, loader,num_classes):
    Model.eval()
    with torch.no_grad():
        scores = []
        ytrue = []
        for batch_idx, batch_test in enumerate(loader):
            ytrue.append(batch_test['label'].detach().numpy())  # .to(device=device)
            ytest=batch_test['label'].to(device=device) 
            x = batch_test['image'].to(device=device)
            x_aug = batch_test['sim_clr'].to(device=device)
            x_mask = batch_test['cut_out'].to(device=device)
            output = Model(x, x)
            out=(output[0]+output[1])/2
            scores.append(torch.sigmoid(out).detach().cpu().numpy())
    ytrue = np.array(ytrue)
    ytrue = ytrue.squeeze(axis=1)
    scores = np.array(scores)
    scores = scores.squeeze(axis=1)
    ypred = np.zeros((scores.shape[0], num_classes))
    ypred[scores > 0.5] = 1
    metrics_function(ytrue, ypred)
    return 0

### Define the data augumentation methods
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def get_augmentation_onetype():
    train_transform = [

        albu.RandomRotate90(p=1),
        albu.OneOf(
            [
                albu.IAASharpen(p=0.5),
                albu.Blur(blur_limit=3, p=0.5),
                albu.MotionBlur(blur_limit=3, p=0.5),
            ],
            p=0.9,
        )

    ]
    return albu.Compose(train_transform)


def get_augmentation():
    train_transform = [

        albu.RandomRotate90(p=1),
        albu.HorizontalFlip(p=1),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


# Read Multilabel datasets: Trento + Civezzano
class Trento_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            images_dir,
            masks_dir, tfms, augmentation,img_dim=224
    ):
        self.dataset = torchvision.datasets.ImageFolder(images_dir)
        self.wb = pandas.read_csv(masks_dir)
        self.Cut_out=Cutout(8,50)
        self.img_dim=img_dim

    def __getitem__(self, i):
        # read data
        image = tfms(self.dataset[i][0])
        data = {"image": np.array(self.dataset[i][0])}
        augmented1 = augmentation(**data)
        augmented1 = tfms(transforms.ToPILImage(mode='RGB')(augmented1['image']))
        augmented2 = simclr_transform(self.dataset[i][0])
        ### Get name:
        _,file_name = os.path.split(self.dataset.imgs[i][0])
        #file_name,_=os.path.splitext(file_name)
        idx_file = (self.wb[self.wb['images'] == file_name])
        number=re.findall('\d+', idx_file.iloc[0][1])

        label=[]
        for jk in range(len(number)):
            label.append(int(number[jk]))
        label=np.array(label).astype('uint8')
        results = dict()
        results['image_orig'] = cv2.resize((np.array(self.dataset[i][0])), (self.img_dim, self.img_dim))
        results['image'] = image
        results['augmented1'] = augmented1
        results['sim_clr'] = augmented2
        results['cut_out'] = self.Cut_out(image)
        results['label'] = label
        return results

    def __len__(self):
        return len(self.dataset)


# Read Multilabel dataset: AID
class AID_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            images_dir,
            masks_dir, tfms, augmentation,img_dim=224
    ):
        self.dataset = torchvision.datasets.ImageFolder(images_dir)
        self.wb = pandas.read_csv(masks_dir)
        self.img_dim=img_dim
        self.Cut_out=Cutout(1,150)

    def __getitem__(self, i):
        # read data
        image = tfms(self.dataset[i][0])
        data = {"image": np.array(self.dataset[i][0])}
        augmented1 = augmentation(**data)
        augmented1 = tfms(transforms.ToPILImage(mode='RGB')(augmented1['image']))
        augmented2 = simclr_transform(self.dataset[i][0])
        ### Get name:
        _,file_name = os.path.split(self.dataset.imgs[i][0])
        file_name,_=os.path.splitext(file_name)
        ### Get file
        idx_file=(self.wb[self.wb['IMAGE\LABEL'] == file_name])
        label=[]
        for jk in range(1, len(idx_file.iloc[0])):
            label.append(idx_file.iloc[0][jk])
        label = np.array(label)
        results = dict()
        results['image_orig'] = cv2.resize((np.array(self.dataset[i][0])), (self.img_dim, self.img_dim))
        results['image'] = image
        results['augmented1'] = augmented1
        results['sim_clr'] = augmented2
        results['cut_out']=self.Cut_out(image)
        results['label'] = label
        return results

    def __len__(self):
        return len(self.dataset)

# Read Multilabel dataset: Merced
class Merced_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            images_dir,
            masks_dir, tfms, augmentation,img_dim=224
    ):
        self.dataset = torchvision.datasets.ImageFolder(images_dir)
        self.wb = pandas.read_csv(masks_dir)
        self.img_dim=224
        self.Cut_out=Cutout(1,150)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        # read data
        image = tfms(self.dataset[i][0])
        data = {"image": np.array(self.dataset[i][0])}
        augmented1 = augmentation(**data)
        augmented1 = tfms(transforms.ToPILImage(mode='RGB')(augmented1['image']))
        augmented2 = simclr_transform(self.dataset[i][0])
        ### Get name:
        _,file_name = os.path.split(self.dataset.imgs[i][0])
        res = int(''.join(filter(lambda i: i.isdigit(), file_name)))
        label = []
        idx_image = self.wb.iloc[i][0]
        for jk in range(1, len(self.wb.iloc[i])):
            label.append(self.wb.iloc[i][jk])

        label = np.array(label)
        results=dict()
        results['image_orig'] = cv2.resize((np.array(self.dataset[i][0])), (self.img_dim, self.img_dim))
        results['image']=image
        results['augmented1'] = augmented1
        results['sim_clr'] = self.Cut_out(augmented2)
        results['cut_out']=self.Cut_out(image)
        results['label'] = label

        return results


### Weighted fusion
class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs, epsilon=1e-4):
        super(LinearWeightedAvg, self).__init__()
        self.weights = nn.Parameter(torch.div(torch.ones(n_inputs), n_inputs))
        self.epsilon = epsilon

    def forward(self, inputs):
        w = torch.nn.ReLU()(self.weights)
        res = 0
        for emb_idx, emb in enumerate(inputs):
            res += emb * w[emb_idx] * inputs[emb_idx]
        res = res / (torch.sum(w) + self.epsilon)

        return res

## define my transformer encoder
class Transformer_Encoder_module(nn.Module):
    def __init__(self, num_classes=100, num_queries=100, aux_loss=False, dim_feat=192,image_size=256):
        super(Transformer_Encoder_module, self).__init__()

        #Use ViT or Deit encoder
        #self.encoder=ViT('B_16_imagenet1k', pretrained=True,num_classes=num_classes,image_size=image_size)
        self.encoder=deit_base_distilled_patch16_224(pretrained=True)

        ### Decoder
        Decoder_layer1 = TransformerDecoderLayer(d_model=768, dim_feedforward=192, nhead=8, dropout=0.0)
        
        self.transformerDecoder1 = TransformerDecoder(Decoder_layer1,num_layers=1)
        self.transformerDecoder2 = nn.MultiheadAttention(768, 8)

        self.class_encoder1=nn.Linear(dim_feat, num_classes, bias=True)
        self.class_encoder2=nn.Linear(dim_feat, num_classes, bias=True)

        self.class_decoder1 = nn.Sequential(
                        nn.LayerNorm(dim_feat),
                        nn.Linear(dim_feat, num_classes, bias=True),
                                  )

        self.class_decoder2 = nn.Sequential(
            nn.LayerNorm(dim_feat),
            nn.Linear(dim_feat, num_classes, bias=True),
        )

    def forward(self, x1, x2):

        _,_,hidden1 = self.encoder(x1)
        _,_,hidden2 = self.encoder(x2)

        h_encoder_main=self.class_encoder1(hidden1[-1][:,0])
        h_encoder_distil=self.class_encoder1(hidden2[-1][:,1])

        h_decoder_aux_im1 = self.transformerDecoder1(hidden2[-1][:, 2:].permute(1, 0, 2),hidden1[-1][:, 2:].permute(1, 0, 2))
        h_decoder_aux_im1 = h_decoder_aux_im1.permute(1, 0, 2)
        h_decoder1 = self.class_decoder1(torch.mean(h_decoder_aux_im1, 1))

        return h_encoder_main, h_encoder_distil, h_decoder1




# Define main code here
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes=17
    
    ##### Batchsize * accumulation_steps = Batchsize 50 
    batch_size = 5
    accumulation_steps = 10  
    #####
    
    train_size = 70*21 #np.int(2100 / 2)
    val_size = 10 * 21  # np.int(2100 / 2)
    num_epochs = 20
    shape_image=224
    dim_feat=768

    ### Change dataset here: Merced: 1, AID: 2,  Civezzano: 3, Trento: 4
    dataset=3      


    tfms = transforms.Compose([
        transforms.Resize((shape_image, shape_image)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    simclr_transform = transforms.Compose([
        # get a set of data augmentation transformations as described in the SimCLR paper.
        transforms.Resize((shape_image, shape_image)),
        #transforms.transforms.ColorJitter(brightness=0.5),
        # transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.RandomC(224,224),
        transforms.Normalize(0.5, 0.5),
    ])


    # Augmentation if used
    augmentation = get_augmentation_onetype()

    ## Get my dataset: have added text and image name for checking purposes
    
    ####### Merced dataset
    if dataset == 1:
        num_classes = 17
        Multilabel_RS = Merced_Dataset("./data/UCMerced_LandUse/images",
                                "./data/UCMerced_LandUse/LandUse_Multilabeled.csv", tfms, augmentation,img_dim=shape_image)
        Class_tokens = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral',
                    'court', 'dock', 'field', 'grass', 'mobile home park', 'pavement',
                    'sand', 'sea', 'ship', 'tanks', 'trees', 'water']

        # Split dataset into training and test
        test_size = len(Multilabel_RS) - train_size - val_size
        torch.manual_seed(0)
        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(Multilabel_RS,
                                                                             [train_size, test_size, val_size])

    ####### AID dataset
    if dataset==2:
        num_classes = 17

        train_dataset = AID_Dataset("./data/AID_multilabel/images_tr",
                                "./data/AID_multilabel/Multilabel.csv", tfms, augmentation,img_dim=shape_image)
        val_dataset=train_dataset

        test_dataset = AID_Dataset("./data/AID_multilabel/images_test",
                               "./data/AID_multilabel/Multilabel.csv", tfms, augmentation,img_dim=shape_image)

        Class_tokens = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral',
                    'court', 'dock', 'field', 'grass', 'mobile home park', 'pavement',
                    'sand', 'sea', 'ship', 'tanks', 'trees', 'water']

    ####### Civezzano dataset
    if dataset==3:    
        num_classes = 14  
        #train_size=1000

        train_dataset = Trento_Dataset("./data/Civezzano/Train_images/",
                                       "./data/Civezzano/Train_images/Train_label_Civezzano.csv", tfms, augmentation)

        test_dataset = Trento_Dataset("./data/Civezzano/Test_images/",
                                      "./data/Civezzano/Test_images/Test_label_Civezzano.csv", tfms, augmentation)

        val_dataset=train_dataset

        Class_tokens = ['asphalt', 'grass', 'tree', 'vineyard', 'low vegetation', 'car', 'blue roof',
                        'white roof', 'dark roof', 'solar pannel', 'building facade', 'soil', 'gravel', 'rocks']

    ####### Trento dataset
    if dataset == 4:  
        num_classes = 13 
        # train_size=1000

        test_dataset = Trento_Dataset("./data/Trento/Train_images/",
                                       "./data/Trento/Train_images/Train_label_Trento.csv", tfms, augmentation)

        train_dataset = Trento_Dataset("./data/Trento/Test_images/",
                                      "./data/Trento/Test_images/Test_label_Trento.csv", tfms, augmentation)

        val_dataset = train_dataset

        Class_tokens = ['asphalt', 'grass', 'tree', 'vineyard', 'pedestrian crossing', 'person', 'car',
                       'dark roof', 'red roof', 'solar pannel', 'building facade', 'soil', 'shadow']



    ### Define the backbone
    ## Define the Transformer model
    Transformer_Multilabel = Transformer_Encoder_module(num_classes=num_classes, num_queries=num_classes, dim_feat=dim_feat,image_size=shape_image)
    Transformer_Multilabel.to(device=device)


    ## Set data loaders for train and test
    ## Batchsize, can be larger the choice of 25 is due memory issues
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    ## Specify criterion
    Criterion2=nn.BCEWithLogitsLoss(reduction='mean').cuda()

    optimizer = optim.Adam(Transformer_Multilabel.parameters(), lr=3e-4)  ### Can be set variable
    #optimizer = torch.optim.SGD(Transformer_Multilabel.parameters(), lr=0.001,momentum=0.9)  
    #optimizer = AdaBelief(Transformer_Multilabel.parameters(), lr=1e-4) 


    optimizer.zero_grad()
    for epoch in range(20):

        losses = []
        # Train:
        counter=0
        for batch_idx, batch_samples in enumerate(trainloader):
            train_label = batch_samples['label'].to(torch.float)
            train_batch = batch_samples['image'].to(device=device)
            train_batch_aug1 = batch_samples['augmented1'].to(device=device)
            train_batch_aug2 = batch_samples['sim_clr'].to(device=device)
            train_batch_aug3 = batch_samples['cut_out'].to(device=device)
            train_label = train_label.to(device=device) ####+(1-train_label.to(device=device))*0.1

            Sc1,Sc2,Sc3 = Transformer_Multilabel(train_batch, train_batch_aug3)
            loss = Criterion2(Sc1, train_label) +Criterion2(Sc2, train_label) +Criterion2(Sc3, train_label)

            loss = loss / accumulation_steps
            losses.append(loss.item())
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:  # Wait for several backward steps
               optimizer.step()  # Now we can do an optimizer step
               optimizer.zero_grad()

        if epoch%5 == 0:
            testloader=DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
            Check_accuracy_Multilabel(Transformer_Multilabel, testloader,num_classes=num_classes)
            Transformer_Multilabel.train()


        training_loss = np.average(losses)
        print('epoch: \t', epoch, '\t [Loss   : {:.3f}]'.format(training_loss))

    #### Compute accuracy
    print("Final accuracy.........")
    testloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    Check_accuracy_Multilabel(Transformer_Multilabel,testloader,num_classes=num_classes)



    stop = timeit.default_timer()
    print('Time: ', stop - start)

    #### Save model
    torch.save(Transformer_Multilabel.state_dict(),"./data/Civezzano/Civezzano_Model_with_augmentation_Text_trial_0.pth.tar")

