# PSSR_docker

## Current usage
`docker run -v ${DIR_WITH_TIF}:/data/test/PSSR/stats -v ${DIR_MODELS}:/data/test/PSSR/models jawon/pssr:0.0.0 ${PWD} ${MODEL_FILE_NAME} ${SIZE}`

### Example
`docker run -v ${DIR_WITH_TIF}:/data/test/PSSR/stats -v ${DIR_MODELS}:/data/test/PSSR/models jawon/pssr:0.0.0 . PSSR_for_EM_1024.pkl 1024`

`singularity run -B $WORKDIR:/data/test/PSSR/stats -B /work2/02114/wonaya/maverick2/:/data/test/PSSR/models docker://jawon/pssr:0.0.0 . PSSR_for_EM_1024.pkl 1024`
