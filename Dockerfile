
############################################################
# Based on Ubuntu
############################################################

################## BASE IMAGE ######################

FROM ubuntu:18.04

################## METADATA ######################


################## PYTHON 3.7 ####################
RUN apt-get update && apt-get install -y build-essential \
    && apt-get install -y gcc wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b

RUN conda init bash \
    && . ~/.bashrc \
    && conda create --name pssr python=3.7 \ 
    && conda activate pssr \
    && pip install utils numpy \
    && pip install spacy>=2.0.18 \
    && pip install fastai==1.0.55 \ 
    && pip install tifffile libtiff czifile scikit-image \
    && pip uninstall -y torch torchvision \
    && conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

################# LABEL #############################
LABEL software="PSSR"

############### MAIN CMD ###################
ADD inference.py /data/
ADD PSSR/ /data/

ENV PYTHONPATH "${PYTHONPATH}:/data/"

WORKDIR /data/

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pssr", "python3", "/data/inference.py"]
