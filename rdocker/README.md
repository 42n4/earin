## M$ RevoScaleR Rstudio Docker with libraries and scripts of Cichosz book: https://www.amazon.com/Data-Mining-Algorithms-Explained-Using/dp/111833258X and [my lectures of data mining](https://github.com/pwasiewi/earin)

[![Docker Pulls](https://img.shields.io/docker/pulls/42n4/rstudio.svg)](https://hub.docker.com/r/42n4/rstudio/)
[![Docker Stars](https://img.shields.io/docker/stars/42n4/rstudio.svg)](https://hub.docker.com/r/42n4/rstudio/)

### You can pull my docker with all R packages (about 1h with a good internet connection)
```
docker pull 42n4/rstudio   #pull my docker (about 7G in tar)
```
### or make it in three steps (about 1 hour on i7) - you can add your packages to Dockerfile:
#### 1
```
git clone https://github.com/pwasiewi/dokerz.git
cd dokerz/r-ml
docker build --rm -t 42n4/r-ml .
```
#### 2
```
cd ../rstudio-ml
docker build --rm -t 42n4/rstudio-ml .
```
#### 3
```
cd ../rstudio
docker build --rm -t 42n4/rstudio .
docker login 				#docker-hub-user-login and pass to hub.docker.com
docker push 42n4/rstudio 	#send to docker-hub-user/docker-name
```
### Docker offline copy
```
docker save 42n4/rstudio > ~/docker42n4.tar 
docker load < ~/docker42n4.tar
```
### Install 7zip. Unpack all data in a directory data with 8x from a directory bin. 
#### execute in a directory data: 
```
cd data; 
sh AirOnTime87to12.xdf.sh 
sh wgetyellowdatasample201606.sh
../bin/8x census-income.tar.7z
../bin/8x covandcommunities.tar.7z
../bin/8x retailchurn.tar.7z
../bin/8x yellownycmap.tar.7z
cd ..
```
## Start the docker in Linux with exact username and its user id and group id (login: guest pass: rstudio)
docker run -d -p 8787:8787 -e USER=$(whoami) -e USERID=$(id -u) -e GROUPID=$(id -g) -v $(pwd):/home/$(whoami) --name=rstudio 42n4/rstudio

#### If you want the opencl support for AMD, you need to install amd drivers and add '-v /sys/dev:/sys/dev --device=/dev/dri  --device=/dev/kfd' to the docker command:
docker run -d -p 8787:8787 -v /sys/dev:/sys/dev --device=/dev/dri  --device=/dev/kfd -e USERID=$(id -u) -e GROUPID=$(id -g) -e USER=$(whoami) -v $(pwd):/home/$(whoami) --name=rstudio 42n4/rstudio

#### If you want the opencl support for NVIDIA, you need to install nvidia-docker2 package from the NVidia PPA (https://github.com/NVIDIA/nvidia-docker) and add '--runtime=nvidia' to the docker command.

## Start the docker in MSWindows (Docker for Windows) with Linux containers enabled and Powershell and shared disk c: in docker settings (login: rstudio pass: rstudio)
![Screen](https://github.com/pwasiewi/dokerz/raw/master/rstudio/linux_docker_in_windows10.png)
```
docker run -d -p 8787:8787 --name=rstudio --restart=always -v c:/Users/Piotr/remote:/home/rstudio 42n4/rstudio
```
### To turn off hyper-v in powershell after removing docker for windows in order to use e.g. Virtualbox, Vmware
```
Disable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
```
### or
```
dism.exe /Online /Disable-Feature:Microsoft-Hyper-V-All
```
### To turn on hyper-v in powershell
```
dism.exe /Online /Enable-Feature:Microsoft-Hyper-V /All
```
### In order to use VirtualBox with Docker use Docker ToolBox: https://www.docker.com/products/docker-toolbox, but it is deprecated.

### Docker for Windows always use hyper-v disabling other vm providers

##### https://forums.docker.com/t/volume-mounts-in-windows-does-not-work/10693/141
##### https://coderwall.com/p/2rpbba/docker-create-a-bridge-and-shared-network

### Run bash in the docker
```
docker run -it 42n4/rstudio /bin/bash
```
### Run bash in the active docker
```
docker exec -it rstudio /bin/bash
```
### Get or update the docker
```
docker pull 42n4/rstudio
```
### Stop and remove the running docker
```
docker rm -f rstudio
```
### or just use
```
bin/docker_remove
```
### After building a docker clean your hd from unwanted temporary image layers
```
bin/docker_clean
```
## Alternatives for RSTUDIO e.g. https://github.com/jupyter/docker-stacks
#### Examples in https://github.com/ipython-books/cookbook-2nd-code
```
docker run -it --rm -p 8888:8888 -p 4040:4040 -e NB_USER=$(whoami) -e NB_UID=$(id -u) -e NB_GID=$(id -g) -v $(pwd):/home/jovyan/workspace jupyter/all-spark-notebook

#Jupyter with graphical gui
docker run -it --rm -p 8888:8888 -p 4040:4040 -e JUPYTER_ENABLE_LAB=yes -e NB_USER=$(whoami) -e NB_UID=$(id -u) -e NB_GID=$(id -g)  -v $(pwd):/home/jovyan/work jupyter/all-spark-notebook
```
