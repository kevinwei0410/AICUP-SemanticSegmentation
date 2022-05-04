import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision import models
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision.models import vgg16
import torch
import torch.nn.functional as F
from torch import nn
import imageio
import random
from sklearn.model_selection import train_test_split
# import Image
#ignore warning 
import warnings
warnings.simplefilter('ignore')

#weight path
weight_path = "./checkpoint/"

#Training data path
Training_data_path = "./p2_data/train"
Val_data_path = "./p2_data/validation"


#debug -- print all elements in array
np.set_printoptions(threshold = np.inf)

#clear up cache, or the CUDA memroy would out of limit
torch.cuda.empty_cache()

#reproducible
torch.manual_seed(1) 

#training hyperparameter
MINIBATCH_SIZE = 4
NUM_EPOCH = 30
learning_rate = 0.001
weight_decay = 1e-5
gpu_id = 0
gpu_paralell = False

#reproducible
class Reproducible():
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        
train_x_path = './SEG_Train_Datasets/Train_Images/'
train_y_path = './Image_Label/'

class DataPreprocessing():
    def __init__(self):
        print("data loading...")
        train_x_name_list = sorted(list(set(os.listdir(train_x_path))))
        train_y_name_list = sorted(list(set(os.listdir(train_y_path))))
        
        
        training_image_array_x, training_image_array_y = self.ReadImage(train_x_name_list, train_y_name_list)
        
        
        # self.training_image_array_x = training_image_array_x[:1000]
        # self.training_image_array_y = training_image_array_y[:1000]
        # self.val_image_array_x = training_image_array_x[1000:len(training_image_array_x)]
        # self.val_image_array_y = training_image_array_y[1000:len(training_image_array_y)]
        
        
        self.training_image_array_x, self.val_image_array_x, self.training_image_array_y, self.val_image_array_y = train_test_split(training_image_array_x,training_image_array_y, train_size=0.8)
        
        # self.training_image_array_x, self.training_image_array_y = self.ReadImage(training_name_list, Training_data_path)  
        # self.val_image_array_x, self.val_image_array_y = self.ReadImage(val_name_list, Val_data_path)
        
        print(len(self.training_image_array_x), len(self.val_image_array_x), len(self.training_image_array_y), len(self.val_image_array_y))
        
        
        # self.training_image_array_y = self.LabelProcessing(self.training_image_array_y)
        # self.val_image_array_y = self.LabelProcessing(self.val_image_array_y)


        #self.DataArgumentation()  
        
        
       #  torchvision_transform = transforms.Compose([
       #  transforms.ToPILImage(),
       # # transforms.Resize(512),
       #  transforms.ToTensor(),
       #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       #  ])
        
        print(np.array(self.training_image_array_x).shape)
        
        # self.training_image_array_x = torchvision_transform(self.training_image_array_x)
        # self.training_image_array_y = torchvision_transform(self.training_image_array_y)
        
        self.training_image_array_x = torch.tensor(self.training_image_array_x)
        self.training_image_array_y = torch.tensor(self.training_image_array_y)
        
        

        print("data loaded. ")
        
        
    def ReadImage(self, train_x_name_list, train_y_name_list):
        image_x = []
        image_y = []
        for _, i in enumerate(train_x_name_list):
            if i.endswith(".jpg"):
                img = imageio.imread(os.path.join(train_x_path,i))
                img2 = img.copy()
                img2.resize((512, 512, 3))
                image_x.append(img2.transpose(2,0,1))
        for _, i in enumerate(train_y_name_list):        
            if i.endswith(".jpg"):
                # temp = np.load(os.path.join(train_y_path,i))
                # print(temp)
                img = imageio.imread(os.path.join(train_y_path,i))
                img2 = img.copy()
                
                # print(img2.shape)
                
                img2.resize((512, 512))
                image_y.append(img2)
                #print(image_y)
        
        return image_x, image_y
    
    
    def LabelProcessing(self, img_array):
        
        masks = np.empty((len(img_array), 512, 512))
        
        for i , mask in enumerate(img_array):

            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
            masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
            masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
            masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
            masks[i, mask == 2] = 3  # (Green: 010) Forest land 
            masks[i, mask == 1] = 4  # (Blue: 001) Water 
            masks[i, mask == 7] = 5  # (White: 111) Barren land 
            masks[i, mask == 0] = 6  # (Black: 000) Unknown 
        
        return masks
    

    def DataArgumentation(self):
        torchvision_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1)
        ])
        
        
        self.training_image_array_x = torch.tensor(self.training_image_array_x)
        self.training_image_array_y = torch.tensor(self.training_image_array_y)
        
        argumentation_data_x = torch.tensor(torchvision_transform(self.training_image_array_x))
        argumentation_data_y = torch.tensor(torchvision_transform(self.training_image_array_y))
        

        
        self.Training_data_image_array = torch.cat((self.training_image_array_x, argumentation_data_x), 0)
        self.Training_data_label_array = torch.cat((self.training_image_array_y, argumentation_data_y), 0)
        
        


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
    

class fcn32s(nn.Module):
    def __init__(self, pretrained = True):
        super(fcn32s, self).__init__()
        self.fcn32 = vgg16(pretrained=True)
        self.fcn32.classifier = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels = 4096,out_channels =  4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels = 4096, out_channels = 7, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(7, 7, 64 , 32 , 0, bias=False),
        )
    def  forward (self, x) :        
        feature = self.fcn32.features(x)
        x = self.fcn32.classifier(feature)
        return x
    

class Training():
    def __init__(self):
        
        #load model
        #self.main_model = model.FCNs()
        self.main_model  = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True, aux_loss=False)
        self.main_model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        #self.main_model.aux_classifier[-1] = nn.Conv2d(256, 7, kernel_size = (1, 1), stride=(1, 1))
        
        print(self.main_model)
        #self.main_model = model.fcn8s(7)
        
        #gpu check
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            if gpu_paralell:
                self.main_model = torch.nn.DataParallel(self.main_model).cuda()
            else:
                torch.cuda.set_device(gpu_id)
        
        #load data, and transfer to gpu mode
        processing = DataPreprocessing()
        
        self.temp = processing.val_image_array_y

        
        self.torch_train_dataset = Data.TensorDataset(torch.tensor(processing.training_image_array_x).type(torch.FloatTensor), torch.tensor(processing.training_image_array_y).type(torch.LongTensor))
        self.torch_val_dataset = Data.TensorDataset(torch.tensor(processing.val_image_array_x).type(torch.FloatTensor), torch.tensor(processing.val_image_array_y).type(torch.LongTensor))
        
        self.training_data_loader = Data.DataLoader(
        dataset=self.torch_train_dataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
        )
        
        self.validation_data_loader = Data.DataLoader(
        dataset=self.torch_val_dataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
        )
            
        
    def InverseTransform(self, array):
        n_masks = len(array)
        masks_RGB = np.empty((n_masks, 512, 512, 3))
        for i, p in enumerate(array):
            masks_RGB[i, p == 0] = [0,255,255]
            masks_RGB[i, p == 1] = [255,255,0]
            masks_RGB[i, p == 2] = [255,0,255]
            masks_RGB[i, p == 3] = [0,255,0]
            masks_RGB[i, p == 4] = [0,0,255]
            masks_RGB[i, p == 5] = [255,255,255]
            masks_RGB[i, p == 6] = [0,0,0]
        masks_RGB = masks_RGB.astype(np.uint8)
        return masks_RGB
    
    
    def mean_iou_score(self, pred, labels):
        mean_iou = 0
        for i in range(1):
            tp_fp = np.sum(pred == i)
            tp_fn = np.sum(labels == i)
            tp = np.sum((pred == i) * (labels == i))
            iou = tp / (tp_fp + tp_fn - tp)
            mean_iou += iou / 1
            #print('class #%d : %1.5f'%(i, iou))
        #print('\nmean_iou: %f\n' % mean_iou)

        return mean_iou
        
        
        
    def main(self):
        
        print("Start to train...")
        
        #optimization setup
        optimizer = torch.optim.AdamW(self.main_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay = weight_decay)
        #optimizer = torch.optim.SGD(self.main_model.parameters(), lr= 1e-4, weight_decay = 5e-5, momentum = 0.9)
        loss_function = nn.BCELoss()
        
        
        best_IOU = 0.0
        training_loss_save_array = []
        val_loss_save_array = []
        val_IOU_save_array = []
        for epoch in range(1, NUM_EPOCH + 1):

            self.main_model.train() #turn to training mode
            training_running_loss, validation_running_loss = 0.0, 0.0
            for step, (batch_x, batch_y) in enumerate(self.training_data_loader):
                
               # batch_x = torch.squeeze(batch_x)
                
            #training process
                optimizer.zero_grad()
                if self.use_gpu:
                    self.main_model.cuda()
                    loss_function.cuda()
                    batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
                
                
                #torch.squeeze(batch_x)
                pred = self.main_model(batch_x)['out']
                pred_probability = F.log_softmax(pred, dim = 1)
                # print(pred_probability.shape)
                # print(batch_y.shape)
                training_loss = loss_function(torch.squeeze(pred_probability), batch_y)
                
                print(batch_y.cpu().detach().numpy())
                
                training_loss.backward() 
                optimizer.step() 
                
                training_running_loss += training_loss.item()
            
            
            train_loss = training_running_loss/len(self.training_data_loader)
            training_loss_save_array.append(train_loss)
            print('epoch: {}, training loss: {}'.format(epoch, train_loss))
            
            
            
            # validation process
            SAVE_PREDICT = []
            
            self.main_model.eval()  # not to do dropout, and BN
            if epoch % 1 == 0:
                for step, (val_x, val_y) in enumerate(self.validation_data_loader):
                    if self.use_gpu:
                        val_x, val_y = Variable(val_x.cuda()), Variable(val_y.cuda())
                        
                    pred = self.main_model(val_x)['out']
                    pred_probability = F.log_softmax(pred, dim = 1)
                    
                    val_loss = loss_function(torch.squeeze(pred_probability), val_y)
                    if self.use_gpu:
                        pred_probability = pred_probability.cpu().detach().numpy()
                    validation_running_loss += val_loss.item()
                
                    pred_mask = np.argmax(pred_probability, axis = 1)
                    
                    SAVE_PREDICT.extend(pred_mask)
                    
                    #validation IOU
                
                
                mean_iou = self.mean_iou_score(np.array(SAVE_PREDICT).astype(int),  np.array(self.temp).astype(int))
                        
                val_loss = validation_running_loss/len(self.validation_data_loader)
                val_loss_save_array.append(val_loss)
                val_IOU_save_array.append(mean_iou)
                print('epoch: {}, validation loss: {}, mean IOU: {}'.format(epoch, val_loss, mean_iou))
            
            #check point save
            #if best_IOU < mean_iou:
            #   best_IOU = mean_iou
            torch.save(self.main_model.state_dict(), weight_path + str(epoch) +"_weight.pt")

            
        # self.PlotLossCurve(training_loss_save_array, val_loss_save_array)
        # self.PlotIOUCurve(val_IOU_save_array)



if __name__ == '__main__':
    SemanticSegmentation = Training()
    SemanticSegmentation.main()
    
    
    