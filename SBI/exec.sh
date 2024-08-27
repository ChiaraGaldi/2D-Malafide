#!/bin/sh

docker run -it --gpus all --shm-size 64G \
    -v //medias/db/ImagingSecurity_misc/galdi/Mastro/SelfBlendedImages:/app/ \
    sbi bash
