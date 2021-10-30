# if trained on one GPU (Tesla V100)
#python train.py --clip  --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0
# if trained on two GPUs (Tesla V100)
# python train.py --clip  --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 8 --gpu_ids 0,1

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix w_ppa-cv_00

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix wo_ppa-cv_00 \
    -ppa

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix w_ppa-cv_02

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix wo_ppa-cv_02 \
    -ppa

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix w_ppa-cv_03

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix wo_ppa-cv_03 \
    -ppa

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix w_ppa-cv_04

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix wo_ppa-cv_04 \
    -ppa

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix w_ppa-cv_05

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix wo_ppa-cv_05 \
    -ppa

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix w_ppa-cv_06

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix wo_ppa-cv_06 \
    -ppa

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix w_ppa-cv_07

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq09.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix wo_ppa-cv_07 \
    -ppa

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix w_ppa-cv_09

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq10.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix wo_ppa-cv_09 \
    -ppa

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix w_ppa-cv_10

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 \
    -td datasets/kitti360_seq00.hdf5 datasets/kitti360_seq02.hdf5 datasets/kitti360_seq03.hdf5 datasets/kitti360_seq04.hdf5 datasets/kitti360_seq05.hdf5 datasets/kitti360_seq06.hdf5 datasets/kitti360_seq07.hdf5 datasets/kitti360_seq09.hdf5 \
    -tdc datasets/kitti360_acmnet_train.json \
    --suffix wo_ppa-cv_10 \
    -ppa
