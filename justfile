# see https://stackoverflow.com/questions/58424974/anaconda-importerror-usr-lib64-libstdc-so-6-version-glibcxx-3-4-21-not-fo
set export
LD_LIBRARY_PATH := '/mmfs1/data/adhinart/mambaforge/envs/suprem/lib'
# https://bbs.archlinux.org/viewtopic.php?id=293565
LD_PRELOAD := '/usr/lib/libdrm_amdgpu.so.1'

default:
    echo $LD_LIBRARY_PATH
    just --list

local:
    python post_processing.py --reference_path "/run/host/var/home/jason/dumb/sample_vertebrae" --predict_path "/var/home/jason/Downloads/AbdomenAtlasDemoPredict" --output_path "/var/home/jason/Downloads/AbdomenAtlasDemoPredictPostProcessed"

andromeda:
    python post_processing.py --reference_path "/data/adhinart/zongwei/sample_vertebrae" --predict_path "/data/adhinart/zongwei/AbdomenAtlasDemoPredict" --output_path "/data/adhinart/zongwei/AbdomenAtlasDemoPredictPostProcessednonrigid"

test:
    python test.py
