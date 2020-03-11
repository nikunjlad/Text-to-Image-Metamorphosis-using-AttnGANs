
# Text-to-Image Metamorphosis &nbsp; ![](https://img.shields.io/badge/release-v1.0-orange)

![](https://img.shields.io/badge/license-MIT-blue) &nbsp;


## Description &nbsp; 
[![](https://img.shields.io/badge/GitHub-taoxugit-red)](https://github.com/taoxugit/AttnGAN) &nbsp;
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

For an entire list of libraries for this project refer the [requirements.txt](http://gitlab.uiius.com/uvision/texture3d/blob/master/requirements.txt) file. <br>

## System Information &nbsp; ![](https://img.shields.io/badge/Ubuntu-18.04-blueviolet)

Developed and configured this project on MAC using PyCharm IDE and trained the model on Google Cloud using NVidia Tesla T4 GPU
Data lies in the following path:

/mnt/data/texture3D/images/ : directory containing images each of 1024x1024 dimension. &nbsp;
/mnt/data/texture3D/iuvs/ : directory containing IUV map of each person in an image. If image contains multiple persons, multiple images exist. For ex: If an image named test12.png containing multiple persons exists in /images directory, then IUV maps of names iuv12_0.png, iuv_1.png, etc are generated here. &nbsp;
/mnt/data/texture3D/output/ : directory where the .obj and pickle files are exported. &nbsp;


## Docker setup

The entire project is run inside a docker container named <b>tex</b>. The [docker](http://gitlab.uiius.com/uvision/texture3d/tree/master/docker) directory in the project has the [Dockerfile](http://gitlab.uiius.com/uvision/texture3d/blob/master/docker/Dockerfile) as well as scripts to [build](http://gitlab.uiius.com/uvision/texture3d/blob/master/docker/build.sh) a docker image and [run](http://gitlab.uiius.com/uvision/texture3d/blob/master/docker/run.sh) a docker container.

#### Building docker image 

The docker image is built using command:

```bash
docker build -t tex2shape:latest .

```

In the above command, <b>tex2shape</b> is name of the docker image while <b>latest</b> is the tag assigned to the image. We use the <b>-t</b> command line argument for The image can be of any name of user interest and the tags essentially distinguish each image from other (e.g. v1.0, v2.0 or latest as in the above case). The period at the end indicates to take the Dockerfile existing in the current directory.

Type,

```bash
docker images
``` 
to find the image which was just built.

#### Running a docker container 

To run a docker container use command:

```bash
docker run --gpus '"device=0"' --name tex --rm --privileged -v /mnt/interns/nikunj/data/texture3D/:/mnt/data -v /home/nikunj/texture3D:/app -p 5000:5000 -e="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" --net host -it tex2shape:latest bash
```

In above command, the data directory from <b>/mnt/interns/nikunj/data/texture3D/</b> on our local computer to <b>/mnt/data</b> directory inside the docker container. Likewise, we map our code directory from <b>/home/nikunj/texture3D</b> on our local machine to <b>/app</b> directory inside the docker. The above built image tex2shape:latest is used to launch the docker containter using interactive mode (specified by the -it option). <b>tex</b> is name of docker container and this docker is run on a gpu. The GPUs needs to be mentioned with their id's (0 in our case). Use <b>nvidia-smi</b> to get the GPU id.

## How to run?

1. Run build.sh to build docker image.
2. Run run.sh to start docker container. Make sure to enable X11 forwarding between docker and host machine. Run xhost local:docker prior to running the container.
3. Check if inside /app directory. Inside /app directory, execute run.sh and enter the image and IUV map image names for generating 3D  models. An obj file and pickle file is saved in the output directory.

## Developers

[![](https://img.shields.io/badge/Nikunj-Lad-yellow)](https://github.com/nikunjlad)