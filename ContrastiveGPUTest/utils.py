import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def categorisor(i,encoder,kmeans):
    with torch.no_grad():   # could remove and use detach later?   

        i=encoder(i.unsqueeze(0)).float()
        out= kmeans.predict(i.to('cpu'))
    return out

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox, size,device):
    #print(size)
    img_w, img_h = 224,224
    b = box_cxcywh_to_xyxy(out_bbox).to(device)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=torch.device(device))#.to(device,non_blocking=True)
    return b

def getArea(bbox,im):
    imArea=224*224
    x1,y1,x2,y2=bbox.unpack(-1)
    x=x2-x1
    y=y2-y1
    bboxArea=x*y
    return bboxArea/imArea
