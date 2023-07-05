# PSSR_docker

## Build
`git clone --recurse-submodules https://github.com/eriksf/PSSR_docker.git`  
or  
`git clone https://github.com/eriksf/PSSR_docker.git`  
`git submodule update --init --recursive`  
and  
`docker build -t eriksf/pssr:<version> .`  

## Current usage
`docker run -v ${DIR_WITH_TIF}:/data/test/PSSR/stats -v ${DIR_MODELS}:/data/test/PSSR/models eriksf/pssr:0.0.0 ${PWD} ${MODEL_FILE_NAME} ${SIZE}`

### Example
`docker run -v ${DIR_WITH_TIF}:/data/test/PSSR/stats -v ${DIR_MODELS}:/data/test/PSSR/models eriksf/pssr:0.0.0 . PSSR_for_EM_1024.pkl 1024`

`singularity exec --cleanenv --nv pssr_0.0.0.sif /usr/local/bin/entrypoint.sh python /data/inference.py $WORKDIR $modelf $size`
