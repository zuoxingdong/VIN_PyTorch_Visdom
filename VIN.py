import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def attention(tensor, params):
    """Attention model for grid world
    """
    S1, S2, args = params
    
    num_data = tensor.size()[0]

    # Slicing S1 positions
    slice_s1 = S1.expand(args.imsize, 1, args.ch_q, num_data)
    slice_s1 = slice_s1.permute(3, 2, 1, 0)
    q_out = tensor.gather(2, slice_s1).squeeze(2)
    
    # Slicing S2 positions
    slice_s2 = S2.expand(1, args.ch_q, num_data)
    slice_s2 = slice_s2.permute(2, 1, 0)
    q_out = q_out.gather(2, slice_s2).squeeze(2)
    
    return q_out

class VIN(nn.Module):
    """Value Iteration Network architecture"""
    def __init__(self, args):
        super(VIN, self).__init__()
        
        # First hidden Conv layer
        self.conv_h = nn.Conv2d(in_channels=args.ch_i, 
                                out_channels=args.ch_h, 
                                kernel_size=3, 
                                stride=1, 
                                padding=(3 - 1)//2, # SAME padding: (F - 1)/2
                                bias=True)
        # Conv layer to generate reward image
        self.conv_r = nn.Conv2d(in_channels=args.ch_h, 
                                out_channels=1, 
                                kernel_size=3, 
                                stride=1, 
                                padding=(3 - 1)//2, # SAME padding: (F - 1)/2
                                bias=False)
        # q layers in VI module
        self.conv_q = nn.Conv2d(in_channels=2, # stack [r, v] -> 2 channels
                                out_channels=args.ch_q, 
                                kernel_size=3, 
                                stride=1, 
                                padding=(3 - 1)//2, # SAME padding: (F - 1)/2
                                bias=False)
        # Final fully connected layer
        self.fc1 = nn.Linear(in_features=args.ch_q, # After attention model -> Q(s, .) for q layers
                             out_features=8, # 8 available actions
                             bias=False)
        
        # Record grid image, reward image and its value images for each VI iteration
        self.grid_image = None
        self.reward_image = None
        self.value_images = []
        
    def forward(self, X, S1, S2, args, record_images=False):
        # Get reward image from observation image
        h = self.conv_h(X)
        r = self.conv_r(h)
        
        if record_images: # TODO: Currently only support single input image
            # Save grid image in Numpy array
            self.grid_image = X.data[0].cpu().numpy() # cpu() works both GPU/CPU mode
            # Save reward image in Numpy array
            self.reward_image = r.data[0].cpu().numpy() # cpu() works both GPU/CPU mode
        
        # Initialize value map (zero everywhere)
        v = torch.zeros(r.size())
        # Move to GPU if necessary
        v = v.cuda() if X.is_cuda else v
        # Wrap to autograd.Variable
        v = Variable(v)
        
        # K-iterations of Value Iteration module
        for _ in range(args.k):
            rv = torch.cat([r, v], 1) # [batch_size, 2, imsize, imsize]
            q = self.conv_q(rv)
            v, _ = torch.max(q, 1) # torch.max returns (values, indices)
            
            if record_images: 
                # Save single value image in Numpy array for each VI step
                self.value_images.append(v.data[0].cpu().numpy()) # cpu() works both GPU/CPU mode
        
        # Do one last convolution
        rv = torch.cat([r, v], 1) # [batch_size, 2, imsize, imsize]
        q = self.conv_q(rv)
        
        # Attention model
        q_out = attention(q, [S1.long(), S2.long(), args])
        
        # Final Fully Connected layer
        logits = self.fc1(q_out)
        
        return logits
