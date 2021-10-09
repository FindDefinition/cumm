## How to debug manylinux build

```docker run --rm -it -e PLAT=manylinux2014_x86_64 -v `pwd`:/io -v $HOME:/myhome scrin/manylinux-cuda:cu114-devel bash```