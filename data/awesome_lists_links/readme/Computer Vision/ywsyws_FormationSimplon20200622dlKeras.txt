# FormationSimplon20200622dlKeras

## Note to set up docker
#### to build an image
```docker build -t <image name> .```

#### to run a container
```docker container run -it --name <container name> -p 8004:5000 -v "$(PWD):/workspace" -d <image name>```
e.g.: docker container run (--rm) -it --name "azure-ml" -p 8004:5000 -v "$(PWD):/workspace" (-d) jupyter-python
explaination:  	       containers name /  hostPC port: service port in container / $(PWD) = projects directory

#### to execute the container in bash
```docker exec -it <container name> bash```

#### start jupyter notebook in the containter
```jupyter notebook --port=5000 --NotebookApp.password='' --NotebookApp.token='' --no-browser --ip=0.0.0.0 --allow-root```

#### launch jupyter notebook in navigator
localhost:8004

## Consignes


Comprendre les transformers : 
http://jalammar.github.io/illustrated-gpt2/#part-3-beyond-language-modeling
http://jalammar.github.io/illustrated-transformer/

https://arxiv.org/abs/2005.12872
https://github.com/facebookresearch/detr