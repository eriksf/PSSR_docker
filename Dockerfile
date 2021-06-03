############################################################
# Based on continuumio/miniconda3
############################################################


################## BASE IMAGE ######################
FROM continuumio/miniconda3:4.9.2


################## BUILD TOOLS #####################
RUN apt-get update && apt-get install -y build-essential vim-tiny \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


################## METADATA ######################
RUN . ~/.bashrc \
    && conda create --name pssr python=3.7 \
    && echo "conda activate pssr" >> ~/.bashrc \
    && conda activate pssr \
    && conda install --yes numpy \
       spacy>=2.0.18 \
       tifffile \
       libtiff \
       scikit-image \
    && conda install --yes pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch \
    && pip install libtiff czifile fastai==1.0.55 \
    && conda clean -afy \
    && python -c "import libtiff; pass"

################# LABEL #############################
LABEL software="PSSR"

############### MAIN CMD ###################
COPY inference.py /data/
COPY PSSR/ /data/
COPY entrypoint.sh /usr/local/bin
ENV PYTHONPATH "${PYTHONPATH}:/data/"
WORKDIR /data/

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
