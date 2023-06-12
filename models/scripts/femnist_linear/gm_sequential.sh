#! /usr/bin/bash

set -exu  # x: stack trace
export CUDA_VISIBLE_DEVICES=""


echo "Starting at $(date)"

aggregation="geom_median"
aggregation_out="gm"

outf="outputs/emnist/outputs"
logf="outputs/emnist/log"
num_rounds=2000
batch_size=10
num_epochs=1
clients_per_round=100
lr=1.0
reg=0.000000001
niter=2

data_dir="../"  # ensure data folder is at "../data/"


main_args=" -dataset femnist -model erm_log_reg -lr $lr  --data_dir ${data_dir}"
options=" --num-rounds ${num_rounds} --eval-every 50 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} -reg_param $reg "
options="${options} --weiszfeld-maxiter ${niter} --aggregation $aggregation "

for seed in 1 2 3 4 5
do
corruption="clean"

time python -u main.py ${main_args} $options  --seed $seed --output_summary_file ${outf}_${aggregation_out}_${corruption}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${seed} 2>&1

corruption="p_x"
for frac in 0.01 0.25
do
    time CUDA_VISIBLE_DEVICES="" python -u main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_${seed} 2>&1
done

corruption="omniscient"
for frac in 0.01 0.25
do
cmds="$cmds ; sleep 30s; sleep 3m ; sleep 3m ; sleep 3m"
    time CUDA_VISIBLE_DEVICES="" python -u main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_${seed} 2>&1
done


done  # seed

echo "Done at $(date)"
