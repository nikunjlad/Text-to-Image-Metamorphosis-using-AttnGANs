
# Text-to-Image Metamorphosis using GANs &nbsp; ![](https://img.shields.io/badge/release-v1.0-orange)

![](https://img.shields.io/badge/license-MIT-blue) &nbsp;<br>

## Architecture &nbsp;
<img src="https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/assets/framework.png">

## Description &nbsp; 
[![](https://img.shields.io/badge/GitHub-taoxugit-red)](https://github.com/davidstap/AttnGAN) &nbsp;
[![](https://img.shields.io/badge/arXiv-AttnGAN-lightgrey)](https://arxiv.org/abs/1711.10485)

Generative Adversarial Networks are used heavily for generating synthetic data and for upsampling an unbalanced dataset. However, it has more to it and one of it's application can be observed in this repository. Text-to-Image Metamorphosis is translation of a text to an Image. Essentially, it is inverse of Image Captioning. In Image Captioning, given an image, we develop a model to generate a caption for it based on the underlying scene. Text-to-Image Metamorphosis, generates an image from a corresponding text by understanding the language semantics. Various works have been done in this domain, the most notable being developing an Attentional GAN model to develop images given a local word feature vector and a global sentence vector. Currently we have only worked on AttnGAN. Further up, we intend to implement [MirrorGAN](https://arxiv.org/abs/1903.05854), an extension of AttnGAN to generate images from sentences and reconstruct the sentences from the generated image, so as to see how similar are the input and output sentences. Concretely, we would like a input and output sentences to be as close to each other (like a mirror) so as to conclude, the underlying generation is close to ground truth.

## Dependencies &nbsp;
![](https://img.shields.io/badge/python-3.6-yellowgreen) &nbsp; ![](https://img.shields.io/badge/install%20with-pip-orange)

You can either create a virtualenv or a docker container for running the project. We went about creating a virtualenv <br>
Necessary libraries include but not limited to the following and are installed using pip.

python==3.6.5 <br>
numpy=>=1.18.1 <br>
pandas>=1.0.1 <br>
nltk>=3.4.5 <br>
torch>=1.4.0 <br>
torchvision>=0.5.0 <br>

For an entire list of libraries for this project refer the [requirements.txt](https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/requirements.txt) file. <br>

## System Information &nbsp; 
![](https://img.shields.io/badge/Ubuntu-18.04-blueviolet) 
![](https://img.shields.io/badge/Google%20-Cloud-yellow)
![](https://img.shields.io/badge/NVidia-TeslaT4-red)

Developed and configured this project on MAC using PyCharm IDE and trained the model on Google Cloud using NVidia Tesla T4 GPUs <br>

The data lies in the data directory. A download the bird.zip file and place it in the data directory in this repository. Following are the files in the bird.zip file.

1. <b>/birds/CUB_200_2011/</b> : It contains an images directory which holds 200 bird class directories. The bird dataset has 200 classes of birds each class has ~60 images of birds.
2. <b>/birds/CUB_200_2011/bounding_boxes.txt</b> : file which contains bounding box information of the birds in the true images. The annotations are in [top-left-x, top-left-y, width, height] format.
3. <b>/birds/CUB_200_2011/classes.txt</b> : file containing names of all the 200 classes.
4. <b>/birds/CUB_200_2011/image_class_labels.txt</b> : file containing class labels. First 60 lines belong to image 1, next 60 lines belong to image 2 and so and so forth.
5. <b>/birds/CUB_200_2011/images.txt</b> : file containing all the image names in the dataset. this is about ~12000 images, 60 images per class.

## Attention Maps &nbsp;

<img src="https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/assets/attention_maps0.png">

## How to run? &nbsp;

Pre-train DAMSM models:
  - For bird dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0`
  - For coco dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 1`
 
- Train AttnGAN models:
  - For bird dataset: `python main.py --cfg cfg/bird_attn2.yml --gpu 2`
  - For coco dataset: `python main.py --cfg cfg/coco_attn2.yml --gpu 3`

- `*.yml` files are example configuration files for training/evaluation our models.

Run `python main.py --cfg cfg/eval_bird.yml --gpu 1` to generate examples from captions in files listed in "./data/birds/example_filenames.txt". Results are saved to `DAMSMencoders/`. 
- Change the `eval_*.yml` files to generate images from other pre-trained models. 
- Input your own sentence in "./data/birds/example_captions.txt" if you wannt to generate images from customized sentences. 

## Results &nbsp;

Given the following 3 statements to the model, we get some realistic bird images as results.

1. a red bird has a white belly and short beak. <br>
<img src="https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/assets/bird1.png" width="600" height="600">
2. a bird has long white wings. <br>
<img src="https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/assets/bird2.png" width="600" height="600">
3. a green bird has white wings and long tail. <br>
<img src="https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/assets/bird3.png" width="600" height="600">

## Developers &nbsp;

[![](https://img.shields.io/badge/Nikunj-Lad-yellow)](https://github.com/nikunjlad)
