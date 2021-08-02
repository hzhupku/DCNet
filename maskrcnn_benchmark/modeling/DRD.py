import torch
import torch.nn.functional as F
import torch.nn as nn

class DenseRelationDistill(nn.Module):

    def __init__(self, indim, keydim, valdim, dense_sum=False):
        super(DenseRelationDistill,self).__init__()
        #self.key_q = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        #self.value_q = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.key_t = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.value_t = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.sum = dense_sum
        if self.sum:
            self.key_q0 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.value_q0 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.key_q1 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.value_q1 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.key_q2 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.value_q2 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.key_q3 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.value_q3 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.key_q4 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.value_q4 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
            self.bnn0 = nn.BatchNorm2d(256)
            self.bnn1 = nn.BatchNorm2d(256)
            self.bnn2 = nn.BatchNorm2d(256)
            self.bnn3 = nn.BatchNorm2d(256)
            self.bnn4 = nn.BatchNorm2d(256)
            self.combine = nn.Conv2d(512,256,kernel_size=1,padding=0,stride=1)
   
    def forward(self,features,attentions):
        features = list(features)
        if isinstance(attentions,dict):
            for i in range(len(attentions)):
                if i==0:
                    atten = attentions[i].unsqueeze(0)
                else:
                    atten = torch.cat((atten,attentions[i].unsqueeze(0)),dim=0)
            attentions = atten.cuda()
        output = []
        h , w = attentions.shape[2:]
        ncls = attentions.shape[0]       
        key_t = self.key_t(attentions)   
        val_t = self.value_t(attentions) 
        for idx in range(len(features)):
            feature = features[idx]
            bs = feature.shape[0]       
            H , W = feature.shape[2:]
            feature = F.interpolate(feature,size=(h,w),mode='bilinear',align_corners=True)
            key_q = eval('self.key_q'+str(idx))(feature).view(bs,32,-1) 
            val_q = eval('self.value_q'+str(idx))(feature)   
            for i in range(bs):
                kq = key_q[i].unsqueeze(0).permute(0,2,1)   
                vq = val_q[i].unsqueeze(0)                  
       
                p = torch.matmul(kq,key_t.view(ncls,32,-1))   
                p = F.softmax(p,dim=1)

                val_t_out = torch.matmul(val_t.view(ncls,128,-1),p).view(ncls,128,h,w)  
                for j in range(ncls):
                    if(j==0):
                        final_2 = torch.cat((vq,val_t_out[j].unsqueeze(0)),dim=1)
                    else:
                        final_2 += torch.cat((vq,val_t_out[j].unsqueeze(0)),dim=1)
                if(i==0):
                    final_1 = final_2
                else:
                    final_1 = torch.cat((final_1,final_2),dim=0)
            final_1 = F.interpolate(final_1,size=(H,W),mode='bilinear',align_corners=True)
            if self.sum:
                final_1 = eval('self.bnn'+str(idx))(final_1)
          
            output.append(final_1)
        

        if self.sum:
            for i in range(len(output)):
                output[i] = self.combine(torch.cat((features[i],output[i]),dim=1))
        output = tuple(output)
        
        return output