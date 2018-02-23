### For building in a new folder with a Dockerfile in it (you can add your packages to it)

docker build --rm -t 42n4/rstudio .

https://github.com/rocker-org/rocker/wiki/Using-the-RStudio-image

https://github.com/rocker-org/rocker-versioned/tree/master/rstudio

### Start the docker in Linux with exact username and its user id and group id (login: guest pass: rstudio)

docker run -d -p 8787:8787 --name=rstudio -e USER=guest -e USERID=1001 -e GROUPID=100 -v $(pwd):/home/guest 42n4/rstudio

### Start the docker in MSWindows (Docker for Windows) with Linux containers enabled and Powershell and shared disk c: in docker settings (login: rstudio pass: rstudio)
https://github.com/pwasiewi/dokerz/blob/master/rstudio/linux_docker_in_windows10.png

docker run -d -p 8787:8787 --name=rstudio --restart=always -v c:/Users/Piotr/remote:/home/rstudio 42n4/rstudio

### To turn off hyper-v in powershell after removing docker for windows in order to use e.g. Virtualbox, Vmware

Disable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
### or
dism.exe /Online /Disable-Feature:Microsoft-Hyper-V-All

### To turn on hyper-v in powershell
dism.exe /Online /Enable-Feature:Microsoft-Hyper-V /All

### In order to use VirtualBox with Docker use Docker ToolBox: https://www.docker.com/products/docker-toolbox

### Docker for Windows always use hyper-v disabling other vm providers

##### https://forums.docker.com/t/volume-mounts-in-windows-does-not-work/10693/141
##### https://coderwall.com/p/2rpbba/docker-create-a-bridge-and-shared-network

### Run bash in the docker

docker exec -it rstudio /bin/bash

### Update the docker

docker pull 42n4/rstudio

### Stop the running docker

docker rm rstudio -f
