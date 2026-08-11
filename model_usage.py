import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches

class BasicBlock(nn.Module):
    def __init__(self,input_channels,output_channels,kernel):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels,output_channels,kernel,padding='same',bias=False)

        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(output_channels)

    def forward(self,x: torch.Tensor):
        x = self.conv1(x)

        #x = self.relu(x)
        x = self.batch_norm(x)

        return x
    
class BasicModel(nn.Module):
    def __init__(self,Grid: int = 5,Bboxes: int = 2,output_channels: int = 1024,input_resolution: dict={'w':900,'h':900},**kwargs):
        super().__init__()
        self.output_channels = output_channels
        self.conv_layers = self._make_conv_layers()

        self.lin1 = nn.Linear(output_channels,928)
        self.lin2 = nn.Linear(928,(Grid**2)*(5*Bboxes)) # (G*G,5*Bboxes)
        self.relu = nn.ReLU()

    def _make_conv_layers(self,):
        layers_list = [BasicBlock(3,3,7),nn.MaxPool2d(5)] + [BasicBlock(3,6,5),nn.MaxPool2d(5)]
        layers_list += [BasicBlock(6,12,5),BasicBlock(12,6,3)] * 3 + [BasicBlock(6,128,3),nn.MaxPool2d(4)]
        layers_list += [BasicBlock(128,256,3),BasicBlock(256,self.output_channels,3),nn.AdaptiveMaxPool2d(1)]

        result = nn.Sequential(*layers_list)
        return result

    def forward(self,x: torch.Tensor,conf=.7):
        B,_,_,_ = x.shape

        x = self.conv_layers(x)
        x = x.reshape(B,self.output_channels)

        x = self.lin1(x)
        x = self.lin2(x)

        x = self.relu(x)

        return x

def iou(t1: torch.Tensor,t2: torch.Tensor):
    # x y w h
    x1,y1 = torch.max(t1[...,0:1],t2[...,0:1]),torch.max(t1[...,1:2],t2[...,1:2])
    x2,y2 = torch.min(t1[...,0:1]+t1[...,2:3],t2[...,0:1]+t2[...,2:3]),torch.min(t1[...,1:2]+t1[...,3:],t2[...,1:2]+t2[...,3:])
    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    union = t1[...,2:3] * t1[...,3:] + t2[...,2:3] * t2[...,3:] - intersection
    return intersection/union

def non_max_supression(bboxes,conf_threshold=0.5,iou_threshold=0.5):
    bboxes = [box for box in bboxes if box[0] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_bbox = bboxes.pop(0)

        bboxes = [box for box in bboxes if iou(torch.tensor(chosen_bbox[1:]),torch.tensor(box[1:])) < iou_threshold]
        bboxes_after_nms.append(chosen_bbox)
    return bboxes_after_nms


def prepare_preds(pred: torch.Tensor, Grid=5):
    pred = pred.squeeze(0)
    pred = pred.reshape(Grid,Grid,10)

    pred = pred.to('cpu')
    bboxes_1 = pred[...,1:5]
    bboxes_2 = pred[...,6:]

    scores = torch.cat(
        [pred[...,0].unsqueeze(0),pred[...,5].unsqueeze(0)],
        dim=0)
    best_bbox = scores.argmax(0).unsqueeze(-1)
    best_bbox = bboxes_1 * (1-best_bbox) + bboxes_2 * best_bbox

    cell_indices = torch.arange(Grid).repeat(Grid,1).unsqueeze(-1)

    x = 1/Grid * (best_bbox[..., :1] + cell_indices)
    y = 1/Grid * (best_bbox[..., 1:2] + cell_indices.permute(1,0,2))
    w_h = 1/Grid * best_bbox[...,2:4]

    best_conf = torch.max(pred[...,0],pred[...,5]).unsqueeze(-1)
    result = torch.cat([best_conf,x,y,w_h],dim=-1)

    return result

def cellboxes2bboxes(pred,Grid=5):
    converted_pred = prepare_preds(pred).reshape(Grid*Grid,-1)

    bboxes = []
    for i in range(Grid*Grid):
        bboxes.append([x.item() for x in converted_pred[i]])
    return bboxes

def draw_bboxes():
    ...
def get_bboxes(image,model,conf_threshold=.5,iou_threshold=.5,Grid=5,device='cuda'):
    '''
    model = torch.nn.Module --> (1,Grid,Grid,Bboxes*5)
    image = (1080,1920,3)
    conf_threshold - erase values that lower than conf_threshold
    iou_threshold - bboxes between which iou lower than that threshold will be save
    Grid - grid of image (Grid X Grid)
    '''
    image = image.permute(2,0,1)
    image = image.unsqueeze(0)

    image = image.to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(image)    

    pred = cellboxes2bboxes(pred,Grid)
    pred = non_max_supression(pred,conf_threshold,iou_threshold)
    
    return [x[1:] for x in pred],[x[0] for x in pred]

def plot_image(image, boxes):

    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        upper_left_x = box[0]
        upper_left_y = box[1]
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()