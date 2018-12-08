import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class PixelwiseLinearCombination(nn.Module):
    def __init__(self, c_dim = 7, image_size = 256):
        super(PixelwiseLinearCombination, self).__init__()
        #init = torch.ones(3,1)*0.5
        #self.c = Parameter(init)
        #self.c.data.uniform_(0, 1)
        #print(self.c.requires_grad)
        self.c_dim = c_dim
        self.num_c = 3
        self.image_size = image_size
        self.fc_1 = nn.Linear(self.c_dim, 512)
        self.fc_2 = nn.Linear(512, 2*self.num_c*self.image_size)

    def forward(self, A, B, c_trg):
        assert A.size() == B.size(), "Input must of same dimensions"
        #assert self.c.size(0) == A.size(1), "num_chanel must be 3"
        #channel = A.size(1)
        #print(type(self.c + torch.ones(3,1)))
        #print((self.c + torch.ones(3,1)).requires_grad)
        #return self.c + torch.ones(3,1)
        #print(A.size())
        #out = torch.zeros_like(A)
        #print("out",out.size())
        #print(self.c.data)
        #print(out.requires_grad)
        #w = F.leaky_relu(self.fc_1(c_trg), 0.2)
        w = F.leaky_relu(self.fc_1(c_trg), 0.2)
        w = F.sigmoid(self.fc_2(w))
        w = w.view(A.size(0), 2, self.num_c, self.image_size)
        wl = w[:,0,:].squeeze(1)
        wr = w[:,1,:].squeeze(1)
        #wl = wl.view(A.size(0), self.num_c).unsqueeze(2).unsqueeze(3).expand(A.size())
        #wr = wr.view(B.size(0), self.num_c).unsqueeze(2).unsqueeze(3).expand(B.size())
        wl = wl.view(A.size(0), self.num_c, self.image_size).unsqueeze(3).expand(A.size())
        wr = wr.view(B.size(0), self.num_c, self.image_size).unsqueeze(3).expand(B.size())
        #wl = wl.view(A.size(0), self.image_size, self.image_size).unsqueeze(1).expand(A.size())
        #wr = wr.view(B.size(0), self.image_size, self.image_size).unsqueeze(1).expand(B.size())
        return F.tanh(torch.add(torch.mul(wl, A), torch.mul(wr,B)))
        #for i in range(channel):
            #print(out[:,i,:].size())
            #out[:,i,:] = self.c[i]*A[:,i,:] + (1-self.c[i])*B[:,i,:]

        #print(out.requires_grad)
        #return F.tanh(out)
        #return out

def findIndex(row_vector, search_from_0):
    if search_from_0 == True:
        for i in range(0, row_vector.size(0), 1):
            if row_vector[i] == 1:
                return i
    else:
        for i in range(row_vector.size(0)-1, -1, -1):
            if row_vector[i] == 1:
                return i

def divideMultistyleLabel(c_trg):
    # c_trg: batch_size * dim
    num_dim = c_trg.size(1)
    c_trg_1 = torch.zeros_like(c_trg)
    c_trg_2 = torch.zeros_like(c_trg)
    for i in range(c_trg.size(0)):
        if sum(c_trg[i]) == 2:
            c_trg_1[i][findIndex(c_trg[i], True)] = 1
            c_trg_2[i][findIndex(c_trg[i], False)] = 1
        else:
            print("label unrecognizable")
            exit(1)
    return c_trg_1, c_trg_2

if __name__ == "__main__":
    """
    A = torch.rand(3,5,5) * 5
    B = torch.rand(3,5,5) * 10
    print(A, B)
    print(A.requires_grad, B.requires_grad)
    T = torch.rand(3,5,5) * 7.5
    #T = torch.ones(3,1)*5
    loss = nn.MSELoss()
    affine = AffineCombination()
    print(affine.parameters())
    for param in affine.parameters():
         print(param.data, param.requires_grad, param.size())
    optimizer = optim.Adam(affine.parameters())
    optimizer.zero_grad()
    out = affine(A, B)
    print(out)
    loss_a = loss(out, T)
    print(loss_a)
    loss_a.backward()
    for param in affine.parameters():
         print(param.grad)
    optimizer.step()
    for param in affine.parameters():
         print(param.data)
    affine.weightClamp(0, 1)
    for param in affine.parameters():
         print(param.data)
    out = affine(torch.zeros(3,5,5), torch.ones(3,5,5))
    print(out)
    """

    # test divideMultistyleLabel
    a= torch.Tensor([[0, 1, 1, 0, 0],[0, 0, 0, 0, 1],[0, 1, 0, 0, 1],[1,0,0,0,0]])
    print(a)
    a1, a2 = divideMultistyleLabel(a)
    print(a1)
    print(a2)
