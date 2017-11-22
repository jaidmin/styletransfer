# styletransfer

instructions: 

docker: 
you can use docker for this project, execute the following steps:

git clone --recursive https://github.com/tensorflow/haskell.git tensorflow-haskell
cd tensorflow-haskell
IMAGE_NAME=tensorflow/haskell:v0
docker build -t $IMAGE_NAME docker
# TODO: move the setup step to the docker script.
stack --docker --docker-image=$IMAGE_NAME setup
stack --docker --docker-image=$IMAGE_NAME build
stack --docker --docker-image=$IMAGE_NAME exec styletransfer-exe "obama448.jpg vangogh448.jpg 3000


if you dont want to use docker you have to install the following dependencies:

1. libprotobuf (the devel version)
2. libtensorflow (cpu or gpu depending on whether you want to use your cpu or gpu)

beware that in order to use the gpu you need to have cuda 8 and cudnn (recommended version 5.1) installed






