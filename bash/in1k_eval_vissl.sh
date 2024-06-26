#!/bin/bash -v -x

output_dir=$1
original_params_file=$2
in_dir=${output_dir}/in_eval
dataset_root=$3
arch=$4

vissl_root=vissl

mkdir -p ${in_dir}

echo "Converting checkpoint to vissl format.."
python to_vissl.py --path ${original_params_file}
params_file="${original_params_file}_vissl"

echo "Running eval jobs"
cd ${vissl_root}
# Check the value of the string

if [ "$arch" = "vit_b" ]; then
  model_name="vit_b16_no_cls"
elif [ "$arch" = "vit_l" ]; then
  model_name="vit_l16_no_cls"
elif [ "$arch" = "vit_h" ]; then
  model_name="vit_h16_no_cls"
else
    echo "Model not found"
fi

python -m tools.run_distributed_engines config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear config.DATA.TRAIN.DATA_PATHS=["${dataset_root}/train"] config.DATA.TEST.DATA_PATHS=["${dataset_root}/val"] config.DATA.TRAIN.DATA_SOURCES='[disk_folder]' config.DATA.TRAIN.LABEL_SOURCES='[disk_folder]' config.DATA.TRAIN.DATASET_NAMES='[imagenet1k_folder]' config.DATA.TEST.DATA_SOURCES='[disk_folder]' config.DATA.TEST.LABEL_SOURCES='[disk_folder]' config.DATA.TEST.DATASET_NAMES='[imagenet1k_folder]' +config/benchmark/linear_image_classification/imagenet1k/models=${model_name} config.MODEL.TRUNK.VISION_TRANSFORMERS.DROP_PATH_RATE=0.0 config.MODEL.TRUNK.VISION_TRANSFORMERS.DROPOUT_RATE=0.0  config.LOG_FREQUENCY=1  config.CHECKPOINT.CHECKPOINT_FREQUENCY=10  config.SLURM.USE_SLURM=true config.SLURM.CONSTRAINT=volta32gb config.SLURM.LOG_FOLDER=${in_dir} config.SLURM.PARTITION=learnlab config.SLURM.NAME=bench_lin_eval_on_imagenet1k  config.MODEL.WEIGHTS_INIT.PARAMS_FILE=${params_file} config.CHECKPOINT.DIR=${in_dir}