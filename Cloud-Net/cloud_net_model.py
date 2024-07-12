
import torch
import torch.nn as nn
import torch.nn.functional as F

def same_padding(Hin, Win, Hout, Wout, kernel_size, stride=1, dilation=1):
    """
    Given the input tensor dimensions (Hin, Win), kernel size, stride and dilation,
    this function computes the padding required to keep the spatial dimensions
    of the input tensor the same as the output tensor.
    
    Args:
    
    Returns:
        A tuple of two integers, padding_0 and padding_1
    """
    padding_0 = ((Hout - 1) * stride + dilation * (kernel_size[0] - 1) + 1 - Hin) // 2
    padding_1 = ((Wout - 1) * stride + dilation * (kernel_size[1] - 1) + 1 - Win) // 2
    return padding_0, padding_0, padding_1, padding_1


class BNRelu(nn.Module):
    def __init__(self, channels):
        super(BNRelu, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        return F.relu(self.bn(x))

# class ContrArm(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(ContrArm, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels//2, 1)
#         self.bn_relu = BNRelu(out_channels)
#         self.bn_relu1x1 = BNRelu(out_channels//2)
        
#     def forward(self, x):
#         x1 = self.bn_relu(self.conv1(x))
#         x1 = self.bn_relu(self.conv2(x1))
        
#         x2 = self.bn_relu1x1(self.conv1x1(x))
        
#         x = torch.cat((x, x2), dim=1)
#         x = self.bn_relu(x1 + x)
#         return F.relu(x)

class ContrArm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_rows=192, input_cols=192):
        super(ContrArm, self).__init__()
        # Calculate padding for same effect
        padding1 = tuple((k - 1) // 2 for k in kernel_size)
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding1)
        self.bn_relu1 = BNRelu(in_channels)
        
        padding2 = tuple((k - 1) // 2 for k in kernel_size)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding2)
        self.bn_relu2 = BNRelu(out_channels)
        
        # Adjust kernel size for conv3
        kernel_size3 = (kernel_size[0]-2, kernel_size[1]-2)
        padding3 = tuple((k - 1) // 2 for k in kernel_size3)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size3, padding=padding3)
        self.bn_relu3 = BNRelu(out_channels // 2)
        
        self.kernel_size = kernel_size 
        
    def forward(self, input_tensor):
        # x = F.pad(input_tensor, same_padding(input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[2], input_tensor.shape[3],self.kernel_size), 'constant', 0)
        x = self.conv1(input_tensor)
        x = self.bn_relu1(x)
        
        x = self.conv2(x)
        x = self.bn_relu2(x)
        
        x1 = self.conv3(input_tensor)
        x1 = self.bn_relu3(x1)
        
        # Concatenate x and x2 along the channels dimension
        x2 = torch.cat((input_tensor, x1), dim=1)  # Assuming concatenation along channels
        
        # Add x1 and x2
        x = torch.add(x, x2)
        
        # Apply ReLU activation
        x = F.relu(x)
        
        return x

class ImprvContrArm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ImprvContrArm, self).__init__()
    
        # Calculate padding for same effect for conv1, conv2, and conv3
        padding = tuple((k - 1) // 2 for k in kernel_size)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Adjust kernel size for conv4 and calculate padding
        kernel_size_adj = (kernel_size[0] - 2, kernel_size[1] - 2)
        padding_adj = tuple((k - 1) // 2 for k in kernel_size_adj)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 2, kernel_size_adj, padding=padding_adj)
        self.bn4 = nn.BatchNorm2d(out_channels // 2)
        
    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x0 = self.conv2(x)
        x0 = self.bn2(x0)
        x0 = F.relu(x0)
        
        x = self.conv3(x0)
        x = self.bn3(x)
        x = F.relu(x)
        
        x1 = self.conv4(inp)
        x1 = self.bn4(x1)
        x1 = F.relu(x1)

        
        x1 = torch.cat([inp, x1], dim=1)
        
        # Too convoluted right now to implement corretly
        # x2 = self.conv4(x0)
        # x2 = self.bn4(x2)
        # x2 = F.relu(x2)

        # x2 = torch.cat([x2, inp], dim=1)
        
        x = torch.add(x, x1)
        # x = torch.add(x, x2)
        x = F.relu(x)
        
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bridge, self).__init__()
        # Calculate padding for same effect for conv1 and conv2
        padding = tuple((k - 1) // 2 for k in kernel_size)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout(p=0.15)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust kernel size for conv3 and calculate padding
        kernel_size_adj = (kernel_size[0] - 2, kernel_size[1] - 2)
        padding_adj = tuple((k - 1) // 2 for k in kernel_size_adj)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size_adj, padding=padding_adj)
        self.bn3 = nn.BatchNorm2d(out_channels // 2)


    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(x)

        x1 = self.conv3(inp)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)

        x1 = torch.cat([inp, x1], dim=1)

        x = torch.add(x, x1)
        x = F.relu(x)

        return x


class ConvBlockExpPath(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlockExpPath, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        # self.bn_relu = BNRelu(out_channels)
        padding = tuple((k - 1) // 2 for k in kernel_size)  # Calculate padding for 'same' effect

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn_relu = BNRelu(out_channels)
        
    def forward(self, x):
        x = self.bn_relu(self.conv1(x))
        x = self.bn_relu(self.conv2(x))
        return x

class ConvBlockExpPath3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlockExpPath3, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        # self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        # self.bn_relu = BNRelu(out_channels)
        # Assuming kernel_size is a tuple (kernel_height, kernel_width)
        padding = tuple((k - 1) // 2 for k in kernel_size)  # Calculate padding for 'same' effect

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn_relu = BNRelu(out_channels)
        
    def forward(self, x):
        x = self.bn_relu(self.conv1(x))
        x = self.bn_relu(self.conv2(x))
        x = self.bn_relu(self.conv3(x))
        return x

class AddBlockExpPath(nn.Module):
    def __init__(self):
        super(AddBlockExpPath, self).__init__()
        
    def forward(self, x1, x2, x3):
        x = torch.add(x1,x2) 
        x = torch.add(x, x3)
        return F.relu(x)

class ImproveFFBlock4(nn.Module):
    def __init__(self):
        super(ImproveFFBlock4, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.pool3 = nn.MaxPool2d(kernel_size=(8, 8))
        self.pool4 = nn.MaxPool2d(kernel_size=(16, 16))
        
    def forward(self, input_tensor1, input_tensor2, input_tensor3, input_tensor4, pure_ff):
        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = torch.concat([x1, input_tensor1], axis=1)
        x1 = self.pool1(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = torch.concat([x2, input_tensor2], axis=1)
        x2 = self.pool2(x2)

        for ix in range(7):
            if ix == 0:
                x3 = input_tensor3
            x3 = torch.concat([x3, input_tensor3], axis=1)
        x3 = self.pool3(x3)

        for ix in range(15):
            if ix == 0:
                x4 = input_tensor4
            x4 = torch.concat([x4, input_tensor4], axis=1)
        x4 = self.pool4(x4)

        x = torch.add(x1, x2)
        x = torch.add(x, x3)
        x = torch.add(x, x4)
        x = torch.add(x, pure_ff)
        x = F.relu(x)
        return x
    
class ImproveFFBlock3(nn.Module):
    def __init__(self):
        super(ImproveFFBlock3, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.pool3 = nn.MaxPool2d(kernel_size=(8, 8))
        
    def forward(self, input_tensor1, input_tensor2, input_tensor3, pure_ff):
        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = torch.concat([x1, input_tensor1], axis=1)
        x1 = self.pool1(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = torch.concat([x2, input_tensor2], axis=1)
        x2 = self.pool2(x2)

        for ix in range(7):
            if ix == 0:
                x3 = input_tensor3
            x3 = torch.concat([x3, input_tensor3], axis=1)
        x3 = self.pool3(x3)


        x = torch.add(x1, x2)
        x = torch.add(x, x3)
        x = torch.add(x, pure_ff)
        x = F.relu(x)
        return x
    

class ImproveFFBlock2(nn.Module):
    def __init__(self):
        super(ImproveFFBlock2, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.pool3 = nn.MaxPool2d(kernel_size=(8, 8))
        
    def forward(self, input_tensor1, input_tensor2, pure_ff):
        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = torch.concat([x1, input_tensor1], axis=1)
        x1 = self.pool1(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = torch.concat([x2, input_tensor2], axis=1)
        x2 = self.pool2(x2)


        x = torch.add(x1, x2)
        x = torch.add(x, pure_ff)

        x = F.relu(x)
        return x
class ImproveFFBlock1(nn.Module):
    def __init__(self):
        super(ImproveFFBlock1, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.pool3 = nn.MaxPool2d(kernel_size=(8, 8))
        
    def forward(self, input_tensor1, pure_ff):
        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = torch.concat([x1, input_tensor1], axis=1)
        x1 = self.pool1(x1)

        x = torch.add(x1, pure_ff)
        x = F.relu(x)
        return x



class ModelArch(nn.Module):
    def __init__(self, input_rows=192, input_cols=192, num_of_channels=4, num_of_classes=1):
        super(ModelArch, self).__init__()
        self.conv1 = nn.Conv2d(num_of_channels, 16, (3, 3), padding=1)
        
        self.contr_arm1 = ContrArm(16, 32, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.contr_arm2 = ContrArm(32, 64, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.contr_arm3 = ContrArm(64, 128, (3, 3))
        self.pool3 = nn.MaxPool2d((2, 2))
        
        self.contr_arm4 = ContrArm(128, 256, (3, 3))
        self.pool4 = nn.MaxPool2d((2, 2))
        
        # TODO: change to imprv 
        self.imprv_contr_arm = ImprvContrArm(256, 512, (3, 3))
        self.pool5 = nn.MaxPool2d((2, 2))
        
        self.bridge = Bridge(512, 1024, (3, 3))
    
        
        # CHeck what this actually does
        self.convT7 = nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=2)
        self.improve_ff_block4 = ImproveFFBlock4()
        self.conv_block_exp_path3 = ConvBlockExpPath3(1024, 512, (3, 3))
        self.add_block_exp_path3 = AddBlockExpPath()
        
        self.convT8 = nn.ConvTranspose2d(512, 256,  kernel_size=(2, 2), stride=2)
        self.improve_ff_block3 = ImproveFFBlock3()
        self.conv_block_exp_path2 = ConvBlockExpPath(512, 256, (3, 3))
        self.add_block_exp_path2 = AddBlockExpPath()
        
        self.convT9 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=2)
        self.improve_ff_block2 = ImproveFFBlock2()
        self.conv_block_exp_path1 = ConvBlockExpPath(256, 128, (3, 3))
        self.add_block_exp_path1 = AddBlockExpPath()
        
        self.convT10 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=2)
        self.improve_ff_block1 = ImproveFFBlock1()
        self.conv_block_exp_path0 = ConvBlockExpPath(128, 64, (3, 3))
        self.add_block_exp_path0 = AddBlockExpPath()
        
        self.convT11 = nn.ConvTranspose2d(64, 32,  kernel_size=(2, 2), stride=2)
        self.conv_block_exp_path = ConvBlockExpPath(64, 32, (3, 3))
        self.add_block_exp_path = AddBlockExpPath()

        
        self.conv12 = nn.Conv2d(32, num_of_classes, (1, 1))
        
    def forward(self, x):

        print(f'Input: {x.shape}')
        conv1 = F.relu(self.conv1(x))
        print(f'Conv1: {conv1.shape}')

        conv1 = self.contr_arm1(conv1)
        print(f'After contr_arm1: {conv1.shape}')

        pool1 = self.pool1(conv1)
        print(f'After pool1: {pool1.shape}')

        conv2 = self.contr_arm2(pool1)
        print(f'After contr_arm2: {conv2.shape}')
        pool2 = self.pool2(conv2)
        print(f'After pool2: {pool2.shape}')

        conv3 = self.contr_arm3(pool2)
        print(f'After contr_arm3: {conv3.shape}')
        pool3 = self.pool3(conv3)
        print(f'After pool3: {pool3.shape}')

        conv4 = self.contr_arm4(pool3)
        print(f'After contr_arm4: {conv4.shape}')
        pool4 = self.pool4(conv4)
        print(pool4.shape)

        conv5 = self.imprv_contr_arm(pool4)
        print(f'Conv5: {conv5.shape}')
        
        pool5 = self.pool5(conv5)
        print(f'After pool5: {pool5.shape}')
        # Match up to here

        conv6 = self.bridge(pool5)
        print(f'After bridge: {conv6.shape}')

        convT7 = self.convT7(conv6)
        print(f'After convT7: {convT7.shape}')
        prevup7 = self.improve_ff_block4(conv4, conv3, conv2, conv1, conv5)
        # Check that this right dim
        print(f'After improve_ff_block4: {prevup7.shape}')
        up7 = torch.cat((convT7, prevup7), dim=1)
        print(f'After cat with convT7 and prevup7: {up7.shape}')
        conv7 = self.conv_block_exp_path3(up7)
        print(f'After conv_block_exp_path3: {conv7.shape}')
        conv7 = self.add_block_exp_path3(conv7, conv5, convT7)
        print(f'After add_block_exp_path3: {conv7.shape}')

        convT8 = self.convT8(conv7)
        print(f'After convT8: {convT8.shape}')
        prevup8 = self.improve_ff_block3(conv3, conv2, conv1, conv4)
        print(f'After improve_ff_block3: {prevup8.shape}')
        up8 = torch.cat((convT8, prevup8), dim=1)
        print(f'After cat with convT8 and prevup8: {up8.shape}')
        conv8 = self.conv_block_exp_path2(up8)
        print(f'After conv_block_exp_path2: {conv8.shape}')
        conv8 = self.add_block_exp_path2(conv8, conv4, convT8)
        print(f'After add_block_exp_path2: {conv8.shape}')

        convT9 = self.convT9(conv8)
        print(f'After convT9: {convT9.shape}')
        prevup9 = self.improve_ff_block2(conv2, conv1, conv3)
        print(f'After improve_ff_block2: {prevup9.shape}')
        up9 = torch.cat((convT9, prevup9), dim=1)
        print(f'After cat with convT9 and prevup9: {up9.shape}')
        conv9 = self.conv_block_exp_path1(up9)
        print(f'After conv_block_exp_path1: {conv9.shape}')
        conv9 = self.add_block_exp_path1(conv9, conv3, convT9)
        print(f'After add_block_exp_path1: {conv9.shape}')

        convT10 = self.convT10(conv9)
        print(f'After convT10: {convT10.shape}')
        prevup10 = self.improve_ff_block1(conv1, conv2)
        print(f'After improve_ff_block1: {prevup10.shape}')
        up10 = torch.cat((convT10, prevup10), dim=1)
        print(f'After cat with convT10 and prevup10: {up10.shape}')
        conv10 = self.conv_block_exp_path0(up10)
        print(f'After conv_block_exp_path0: {conv10.shape}')
        conv10 = self.add_block_exp_path0(conv10, conv2, convT10)
        print(f'After add_block_exp_path0: {conv10.shape}')

        convT11 = self.convT11(conv10)
        print(f'After convT11: {convT11.shape}')
        up11 = torch.cat((convT11, conv1), dim=1)
        print(f'After cat with convT11 and conv1: {up11.shape}')
        conv11 = self.conv_block_exp_path(up11)
        print(f'After conv_block_exp_path: {conv11.shape}')
        conv11 = self.add_block_exp_path(conv11, conv1, convT11)
        print(f'After add_block_exp_path: {conv11.shape}')

        x13 = self.conv12(conv11)
        print(f'Final output: {x13.shape}')

        # Upsample the tensor to (12, 1, 192, 192)
        return x13


model = ModelArch()
