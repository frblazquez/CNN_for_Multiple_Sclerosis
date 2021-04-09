# DeepHealth - CNNs for Multiple Sclerosis lesion detection

* **Author:** Francisco Javier Blázquez Martínez
* **Director:** Prof. David Atienza Alonso
* **Supervisors:** Dr. Arman Iranfar, Dr. Marina Zapater, Dr. Tomás Teijeiro Campo


### DeepHealth libraries
These are esentially two C++ libraries, first is EDDL with the core functionalities for distributed Deep Learning and secondly ECVL, for computer vision and image management. These libraries have also python wrappers.

| [EDDL](https://github.com/deephealthproject/eddl) | [ECVL](https://github.com/deephealthproject/ecvl) | [PyEDDL](https://github.com/deephealthproject/pyeddl) | [PyECVL](https://github.com/deephealthproject/pyecvl) |
| :------------- | :---------- | :----------- | :----------- |
| [EDDL docs](https://deephealthproject.github.io/eddl/index.html) | [ECVL docs](https://deephealthproject.github.io/ecvl/master/index.html) | [PyEDDL docs](https://deephealthproject.github.io/pyeddl) | [PyECVL docs](https://deephealthproject.github.io/pyecvl/index.html) |


### Docker installation 

[Installation](https://github.com/deephealthproject/docker-libs) and [configuration](https://docs.docker.com/engine/install/linux-postinstall/):
```
> sudo apt install docker.io
> sudo groupadd docker               
> sudo usermod -aG docker $USER
> newgrp docker 
> docker pull dhealth/pylibs:latest
```

Check DeepHealth libraries installation and version with docker:
```
> docker run -it --rm -v /mnt/vol/data:/data:ro dhealth/pylibs /usr/bin/python3
>>> import pyeddl
>>> pyeddl.VERSION
(should print the version)
>>> import pyecvl
>>> pyecvl.VERSION
(should print the version)
```

Start DeepHealth's docker image with sync folder:
```
> docker run -it --mount target=<dst_folder>,src=<src_folder>,type=bind --rm dhealth/pylibs
```

### DOUBTS

[Slack channel](https://app.slack.com/client/TKCHB0BME)


### C4SCIENCE

[Project repository](https://c4science.ch/diffusion/11226/) \
[Multiple Sclerosis image segmentation](https://c4science.ch/diffusion/10390/) \
[DeepHealth Seizure Detection](https://c4science.ch/diffusion/9868/) 

Clone with SSH key: 
```
git clone git@c4science.ch:/diffusion/11226/.git --config core.sshCommand="ssh -i <path_to_private_key>"
git config core.sshCommand="ssh -i <path_to_private_key>"
```

### ESL Server

Rules:
1. _/home_ is only supposed to store small configuration files. 
2. _/scrap/users/yourname_ is for any fast-access local data for the experiments or to install local packages with conda or pip. 
3. _/shares/eslfiler1/home/yourname_ to store data for which you need backup (network filesystem, much slower than /scrap). 
4. _/shares/eslfiler1/scratch_ is a large network volume without backup which may be used to store intermediate results.

Connect to the server:

```
> # It's necessary to be connected to EPFL's VPN
> ssh blazquez@eslsrv12.epfl.ch 
```

Copy file from local to server:
```
> scp -r <localfile> blazquez@eslsrv12.epfl.ch:<path_in_server>
```

Setup the running enviroment:
```
> conda create --prefix /scrap/users/blazquez/.conda/envs/pyeddl
> conda activate /scrap/users/blazquez/.conda/envs/pyeddl
> conda install -c dhealth pyeddl-gpu
```

Optional configuration
```
> create dataset folder in /scrap/users/blazquez
```

Execution benchmarks without GPU:
- My laptop: 0.1254 secs/batch
- Server:    7.6003 secs/batch



### Global vision documentation

Image segmentation: \
https://github.com/mrgloom/awesome-semantic-segmentation \
https://missinglink.ai/guides/computer-vision/image-segmentation-deep-learning-methods-applications/ \
https://tuatini.me/practical-image-segmentation-with-unet/ 

Multiple Sclerosis MRI segmentation: \
https://portal.fli-iam.irisa.fr/msseg-challenge/overview      (MICCAI MSSEG Challenge) \
https://portal.fli-iam.irisa.fr/msseg-challenge/workshop-day  (MICCAI MSSEG Docs and results) 

https://github.com/sergivalverde/nicMSlesions			(https://arxiv.org/pdf/1805.12415.pdf) \
https://github.com/marianocabezas/miccai_challenge2016	(http://marianocabezas.github.io/miccai_challenge2016/) \ 
https://github.com/Fjaviervera/MS-challenge-2016		(https://www.overleaf.com/project/5c986fad25c3584dbc987293) 

https://github.com/deephealthproject/use-case-pipelines/tree/3rd_hackathon \
https://github.com/IntelAI/unet 

### Datasets

[MICCAI 2016](http://miccai2016.org/en/) \
[ISBI](http://brainiac2.mit.edu/isbi_challenge/) \
[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) \
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) 

### References

I want to distinguish between general references and papers which are the theoretical basis for the whole project.


### Troubleshooting

* **Error allocating memory**

Allow to allocate more memory to the processes, [link](https://discuss.pynq.io/t/runtimeerror-failed-to-allocate-memory/1773)
```
# Restart after executing this
echo 1 > /proc/sys/vm/overcommit_memory
```

Allow Jupyter Notebook to allocate bigger memory blocks [link](https://stackoverflow.com/questions/57948003/how-to-increase-jupyter-notebook-memory-limit)
```
# Start jupyter with bigger buffer size or modify this variable in its configuration file
jupyter notebook --NotebookApp.max_buffer_size=your_value
```

* **<Layer>.output NoneType instead of Tensor**

EDDL models created in an inner scope are automatically removed when the environment is deleted if these are not referenced anymore.

```
# Out of this scope the whole net might be removed even if we are still referencing some of its layers
def encoder1(in_):
    vgg19_net = vgg19(in_, include_top=False)
    # or
    vgg19_net = eddl.Model([in_],[out])    
    ...
```

In words of the EDDL developers: \
*"Yes, eddl Net objects destroy internal objects when they are deleted. There have been some changes in memory management after eddl 0.8.3a (corresponding to pyeddl 0.12.0), so that could be the reason why your sample worked in pyeddl 0.12.0."*

* **RuntimeError: [CUDA ERROR]: out of memory (2)**

* **Segmentation fault (core dumped)**

