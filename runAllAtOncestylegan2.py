# TRAINS - Example of Pytorch and matplotlib integration and reporting
#
#"""
#Neural Transfer Using PyTorch
#=============================
#**Author**: `Alexis Jacq <https://alexis-jacq.github.io>`_
#**Edited by**: `Winston Herring <https://github.com/winston6>`_
#Introduction
#------------
#This tutorial explains how to implement the `Neural-Style algorithm <https://arxiv.org/abs/1508.06576>`__
#developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
#Neural-Style, or Neural-Transfer, allows you to take an image and
#reproduce it with a new artistic style. The algorithm takes three images,
#an input image, a content-image, and a style-image, and changes the input
#to resemble the content of the content-image and the artistic style of the style-image.
#.. figure:: /_static/img/neural-style/neuralstyle.png
#   :alt: content1
#"""
#
######################################################################
# Underlying Principle
# --------------------
#
# The principle is simple: we define two distances, one for the content
# (:math:`D_C`) and one for the style (:math:`D_S`). :math:`D_C` measures how different the content
# is between two images while :math:`D_S` measures how different the style is
# between two images. Then, we take a third image, the input, and
# transform it to minimize both its content-distance with the
# content-image and its style-distance with the style-image. Now we can
# import the necessary packages and begin the neural transfer.
#
# Importing Packages and Selecting a Device
# -----------------------------------------
# Below is a  list of the packages needed to implement the neural transfer.
#
# -  ``torch``, ``torch.nn``, ``numpy`` (indispensables packages for
#    neural networks with PyTorch)
# -  ``torch.optim`` (efficient gradient descents)
# -  ``PIL``, ``PIL.Image``, ``matplotlib.pyplot`` (load and display
#    images)
# -  ``torchvision.transforms`` (transform PIL images into tensors)
# -  ``torchvision.models`` (train or load pre-trained models)
# -  ``copy`` (to deep copy the models; system package)

from __future__ import print_function
import sys
sys.path.append('/home/robustness_lib')
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import io
import json
import time

import glob


from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
from trains import Task


task = Task.init(project_name='stylegan', task_name='pytorch stylegan onescript', task_type=Task.TaskTypes.testing)

stylepic=sys.argv[1]
contentpic=sys.argv[2]


#bringing in robustness from red8top reporting

from robustness import datasets, model_utils, constants, helpers



######################################################################
# Loading the Images
# ------------------
#
# Now we will import the style and content images. The original PIL images have values between 0 and 255, but when
# transformed into torch tensors, their values are converted to be between
# 0 and 1. The images also need to be resized to have the same dimensions.
# An important detail to note is that neural networks from the
# torch library are trained with tensor values ranging from 0 to 1. If you
# try to feed the networks with 0 to 255 tensor images, then the activated
# feature maps will be unable sense the intended content and style.
# However, pre-trained networks from the Caffe library are trained with 0
# to 255 tensor images.
#
#
# .. Note::
#     Here are links to download the images required to run the tutorial:
#     `picasso.jpg <https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ and
#     `dancing.jpg <https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     Download these two images and add them to a directory
#     with name ``images`` in your current working directory.
device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
# desired size of the output image
imsize = 512 if ch.cuda.is_available() else 128  # use small size if no gpu


def image_loader(image_name):
  loader = transforms.Compose([
      transforms.Resize(imsize),  # scale imported image
      transforms.ToTensor()])  # transform it into a torch tensor

  image = Image.open(image_name)
  # fake batch dimension required to fit network's input dimensions
  image = loader(image).unsqueeze(0)[:, :3, :, :]
  return image.to(device, ch.float)

style_img = image_loader(stylepic)
content_img = image_loader(contentpic)


#assert style_img.size() == content_img.size(), \
#    "we need to import style and content images of the same size"

######################################################################
# Now, let's create a function that displays an image by reconverting a
# copy of it to PIL format and displaying the copy using
# ``plt.imshow``. We will try displaying the content and style images
# to ensure they were imported correctly.

def imshow(tensor, title=None):
  unloader = transforms.ToPILImage()  # reconvert into PIL image

  image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
  image = image.squeeze(0)      # remove the fake batch dimension
  image = unloader(image)
  plt.imshow(image)
  if title is not None:
      plt.title(title)
  plt.pause(0.001) # pause a bit so that plots are updated

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


######################################################################
# Importing the Model/Prepping pretrained
# -------------------
#
# Now we need to import a pre-trained neural network. We will use a 19
# layer VGG network like the one used in the paper.
#
# PyTorch's implementation of VGG is a module divided into two child
# ``Sequential`` modules: ``features`` (containing convolution and pooling layers),
# and ``classifier`` (containing fully connected layers). We will use the
# ``features`` module because we need the output of the individual
# convolution layers to measure content and style loss. Some layers have
# different behavior during training than evaluation, so we must set the
# network to evaluation mode using ``.eval()``.
#
#cnn = models.vgg19(pretrained=True).features.to(device).eval()

######################################################################
# Additionally, VGG networks are trained on images with each channel
# normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
# We will use them to normalize the image before sending it into the network.
#

#cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
#cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
#class Normalization(nn.Module):
#    def __init__(self, mean, std):
#        super(Normalization, self).__init__()
#        # .view the mean and std to make them [C x 1 x 1] so that they can
#        # directly work with image Tensor of shape [B x C x H x W].
#        # B is batch size. C is number of channels. H is height and W is width.
#        self.mean = torch.tensor(mean).view(-1, 1, 1)
#        self.std = torch.tensor(std).view(-1, 1, 1)
#
#    def forward(self, img):
#        # normalize img
#        return (img - self.mean) / self.std
##

#ALTERNATIVELY do it Robustness_lib's way
dataset = datasets.RestrictedImageNet('')

model_kwargs = {
    'arch': 'resnet50',
    'dataset': dataset,
    'resume_path': '/home/jesse/Documents/robustness_lib/RestrictedImageNet.pt',
    'state_dict_path': 'model',
    'parallel': False
}

# Robust ResNet
model, ckpt = model_utils.make_and_restore_model(**model_kwargs)
robust_resnet = model.model

# Regular ResNet
reg_resnet = copy.deepcopy(robust_resnet)
new_params = reg_resnet.state_dict()
partial_params = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
del partial_params['fc.bias']
del partial_params['fc.weight']
new_params.update(partial_params)
reg_resnet.load_state_dict(new_params)

# VGG
vgg = models.vgg19(pretrained=True).features

#ORIGINAL VERSION DOC
######################################################################
# Loss Functions
# --------------
# Content Loss
# ~~~~~~~~~~~~
#
# The content loss is a function that represents a weighted version of the
# content distance for an individual layer. The function takes the feature
# maps :math:`F_{XL}` of a layer :math:`L` in a network processing input :math:`X` and returns the
# weighted content distance :math:`w_{CL}.D_C^L(X,C)` between the image :math:`X` and the
# content image :math:`C`. The feature maps of the content image(:math:`F_{CL}`) must be
# known by the function in order to calculate the content distance. We
# implement this function as a torch module with a constructor that takes
# :math:`F_{CL}` as an input. The distance :math:`\|F_{XL} - F_{CL}\|^2` is the mean square error
# between the two sets of feature maps, and can be computed using ``nn.MSELoss``.
#
# We will add this content loss module directly after the convolution
# layer(s) that are being used to compute the content distance. This way
# each time the network is fed an input image the content losses will be
# computed at the desired layers and because of auto grad, all the
# gradients will be computed. Now, in order to make the content loss layer
# transparent we must define a ``forward`` method that computes the content
# loss and then returns the layer's input. The computed loss is saved as a
# parameter of the module.
#

class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


######################################################################
# .. Note::
#    **Important detail**: although this module is named ``ContentLoss``, it
#    is not a true PyTorch Loss function. If you want to define your content
#    loss as a PyTorch Loss function, you have to create a PyTorch autograd function
#    to recompute/implement the gradient manually in the ``backward``
#    method.

######################################################################
# Style Loss
# ~~~~~~~~~~
#
# The style loss module is implemented similarly to the content loss
# module. It will act as a transparent layer in a
# network that computes the style loss of that layer. In order to
# calculate the style loss, we need to compute the gram matrix :math:`G_{XL}`. A gram
# matrix is the result of multiplying a given matrix by its transposed
# matrix. In this application the given matrix is a reshaped version of
# the feature maps :math:`F_{XL}` of a layer :math:`L`. :math:`F_{XL}` is reshaped to form :math:`\hat{F}_{XL}`, a :math:`K`\ x\ :math:`N`
# matrix, where :math:`K` is the number of feature maps at layer :math:`L` and :math:`N` is the
# length of any vectorized feature map :math:`F_{XL}^k`. For example, the first line
# of :math:`\hat{F}_{XL}` corresponds to the first vectorized feature map :math:`F_{XL}^1`.
#
# Finally, the gram matrix must be normalized by dividing each element by
# the total number of elements in the matrix. This normalization is to
# counteract the fact that :math:`\hat{F}_{XL}` matrices with a large :math:`N` dimension yield
# larger values in the Gram matrix. These larger values will cause the
# first layers (before pooling layers) to have a larger impact during the
# gradient descent. Style features tend to be in the deeper layers of the
# network so this normalization step is crucial.
#

#def gram_matrix(input):
#    a, b, c, d = input.size()  # a=batch size(=1)
#    # b=number of feature maps
#    # (c,d)=dimensions of a f. map (N=c*d)
#
#    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
#
#    G = torch.mm(features, features.t())  # compute the gram product
#
#    # we 'normalize' the values of the gram matrix
#    # by dividing by the number of element in each feature maps.
#    return G.div(a * b * c * d)


######################################################################
# Now the style loss module looks almost exactly like the content loss
# module. The style distance is also computed using the mean square
# error between :math:`G_{XL}` and :math:`G_{SL}`.
#

#class StyleLoss(nn.Module):
#
#    def __init__(self, target_feature):
#        super(StyleLoss, self).__init__()
#        self.target = gram_matrix(target_feature).detach()
#
#    def forward(self, input):
#        G = gram_matrix(input)
#        self.loss = F.mse_loss(G, self.target)
#        return input

#ROBUSTNESS_LIB STYLE TRANSFER
class ContentLoss(nn.Module):

    def __init__(self, feature_maps,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.feature_maps = {key: val.detach() for key, val in feature_maps.items()}

    def forward(self, input, layer_name):
        return F.mse_loss(input[layer_name], self.feature_maps[layer_name])

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = ch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
  
class StyleLoss(nn.Module):

    def __init__(self, feature_maps):
        super(StyleLoss, self).__init__()
        self.feature_maps = {key: gram_matrix(val).detach() for key, val in feature_maps.items()}

    def forward(self, input, layer_name):
        G = gram_matrix(input[layer_name])
        return F.mse_loss(G, self.feature_maps[layer_name])




#ORIGINAL DOCUMENTATION
######################################################################
# A ``Sequential`` module contains an ordered list of child modules. For
# instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
# Conv2d, ReLU...) aligned in the right order of depth. We need to add our
# content loss and style loss layers immediately after the convolution
# layer they are detecting. To do this we must create a new ``Sequential``
# module that has content loss and style loss modules correctly inserted.
#

# desired depth layers to compute style/content losses :
#content_layers_default = ['conv_4']
#style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


#BACK TO ROBUSTNESSLIB
		
class ResNetStyleTransferModel(nn.Module):
  """This class is used to wrap a ResNet model for style transfer"""
  def __init__(self, model, mean, std):
    super(ResNetStyleTransferModel, self).__init__()
    self.normalize = helpers.InputNormalize(mean, std)
    self.model = model  # Resnet50 model
   
  def forward(self, x):
    layers = {}
    
    x = self.normalize(x)
    
    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)
    
    x = self.model.layer1(x)
    layers['conv_1'] = x
    x = self.model.layer2(x)
    layers['conv_2'] = x
    x = self.model.layer3(x)
    layers['conv_3'] = x
    x = self.model.layer4(x, fake_relu=True)
    layers['conv_4'] = x
    
    return layers		
		
class VGGStyleTransferModel(nn.Module):
  """This class is used to wrap a VGG model for style transfer"""
  def __init__(self, model, mean, std):
    super(VGGStyleTransferModel, self).__init__()
    self.normalize = helpers.InputNormalize(mean, std)
    self.model = model  # VGG model

  def forward(self, x):
    layers = {}
    
    x = self.normalize(x)
    
    i=0
    for layer in self.model.children():
      if isinstance(layer, nn.Conv2d):
          i += 1
          name = 'conv_{}'.format(i)
      elif isinstance(layer, nn.ReLU):
          name = 'relu_{}'.format(i)
          # The in-place version doesn't play very nicely with the ContentLoss
          # and StyleLoss we insert below. So we replace with out-of-place
          # ones here.
          layer = nn.ReLU(inplace=False)
      elif isinstance(layer, nn.MaxPool2d):
          name = 'pool_{}'.format(i)
          layer = nn.AvgPool2d(kernel_size=2, stride=2)
      elif isinstance(layer, nn.BatchNorm2d):
          name = 'bn_{}'.format(i)
      else:
          raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
      x = layer(x)
      
      if isinstance(layer, nn.Conv2d):
        layers[name] = x
      if i == 13:  # This is the last layer we usually use for style transfer
        break
    return layers



	######################################################################
# Gradient Descent
# ----------------
#
# As Leon Gatys, the author of the algorithm, suggested `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__, we will use
# L-BFGS algorithm to run our gradient descent. Unlike training a network,
# we want to train the input image in order to minimise the content/style
# losses. We will create a PyTorch L-BFGS optimizer ``optim.LBFGS`` and pass
# our image to it as the tensor to optimize.
#
def get_input_optimizer(input_img, opt='Adam'):
    # this line to show that input is a parameter that requires a gradient
    if opt == 'Adam':
      optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01)
    elif opt == 'LBFGS':
      optimizer = optim.LBFGS([input_img.requires_grad_()])
    else:
      raise RuntimeError('Unrecognized optimizer: {}'.format(opt))
    return optimizer
######################################################################
# Finally, we must define a function that performs the neural transfer. For
# each iteration of the networks, it is fed an updated input and computes
# new losses. We will run the ``backward`` methods of each loss module to
# dynamicaly compute their gradients. The optimizer requires a "closure"
# function, which reevaluates the modul and returns the loss.
#
# We still have one final constraint to address. The network may try to
# optimize the input with values that exceed the 0 to 1 tensor range for
# the image. We can address this by correcting the input values to be
# between 0 to 1 each time the network is run.
#


def style_transfer(st_model, content_img, style_img, start_from_content=True,
                   n_iters=[0, 2000], style_weight=1e9, content_weight=1,
                   content_layers=['conv_3'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4'],
                   opt='Adam',
                   verbose=True,
                   ):
  st_model.eval().cuda()
  
  if isinstance(content_img, str):
    content_img = image_loader(content_img)
  
  if isinstance(style_img, str):
    style_img = image_loader(style_img)
  
  content_feature_maps = st_model(content_img)
  content_feature_maps = {key: val.detach() for key, val in content_feature_maps.items()}
  
  style_feature_maps = st_model(style_img)
  style_feature_maps = {key: val.detach() for key, val in style_feature_maps.items()}
  
  content_loss_func = ContentLoss(content_feature_maps)
  style_loss_func = StyleLoss(style_feature_maps)
  
  if start_from_content:
    input_img = content_img.clone()
  else:
    input_img = ch.randn(content_img.data.size(), device=device)
    
  optimizer = get_input_optimizer(input_img, opt=opt)
  
  images = []
  run = [0]
  start_time = time.time()
  while run[0] <= n_iters[-1]:
    def closure():
      # correct the values of updated input image
      input_img.data.clamp_(0, 1)

      optimizer.zero_grad()
      input_feature_maps = st_model(input_img)
      
      style_score = 0
      content_score = 0
      edition = 1

      for sl in style_layers:
          _l = style_loss_func(input_feature_maps, sl) * style_weight
          style_score += _l
          #print(_l)
      for cl in content_layers:
          content_score += content_loss_func(input_feature_maps, cl) * content_weight
          #print(content_score)

      #style_score *= style_weight
      #content_score *= content_weight

      loss = style_score + content_score
      loss.backward()
      
      if True:
          print("run {}:".format(run))
          print('Style Loss : {:4f} Content Loss: {:4f} time elapsed: {} seconds'.format(
              style_score.item(), content_score.item(), time.time() - start_time))
          if verbose:
            plt.figure(figsize=(8, 8))
            imshow(input_img, title='run {}:'.format(run))
            
          unloader = transforms.ToPILImage()  # reconvert into PIL image
          image = input_img.clone().cpu()  # we clone the tensor to not do changes on it
          image = image.squeeze(0)      # remove the fake batch dimension
          image = unloader(image)
          imagename = "/content/testrun/" + str(edition)
#          imagefile = open("%s" % imagename, "wb")
          image.save("{0}".format(imagename), 'jpeg')
          images.append(image)
      edition += 1
      run[0] += 1

      return style_score + content_score

    optimizer.step(closure)

  # a last correction...
  input_img.data.clamp_(0, 1)

  return input_img, images
  
  
out = style_transfer(
    ResNetStyleTransferModel(robust_resnet, dataset.mean, dataset.std), 
    style_img, content_img, style_weight=1e9, content_weight=0.5,
    start_from_content=True, n_iters=[0, 50, 100], opt='LBFGS')
