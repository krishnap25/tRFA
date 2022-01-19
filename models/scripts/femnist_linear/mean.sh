#! /usr/bin/bash

source scripts/parallelize.sh


set -u  # x: stack trace
export CUDA_VISIBLE_DEVICES=""

njobs=6
cmds=""
echo "Starting at $(date)"

aggregation_out="avg"

outf="outputs/emnist/outputs"
logf="outputs/emnist/log"
num_rounds=6000
batch_size=10
num_epochs=1
clients_per_round=100
lr=1.0
reg=0.000000001

data_dir="../"  # ensure data is at "../data"

main_args=" -dataset femnist -model erm_log_reg -lr $lr  --data_dir ${data_dir}"
options=" --num-rounds ${num_rounds} --eval-every 50 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} -reg_param $reg "

for seed in  1 2 3 4 5
do
corruption="clean"
cmds="$cmds ; sleep 90s ; sleep 3m ; sleep 3m; sleep 3m; sleep 3m"

cmds="$cmds ; time python -u main.py ${main_args} $options  --seed $seed --output_summary_file ${outf}_${aggregation_out}_${corruption}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${seed} 2>&1 "

corruption="p_x"
for frac in 0.01 0.25
do
    cmds="$cmds ; sleep 90s; sleep 3m ; sleep 3m; sleep 3m"
    cmds="$cmds ; time CUDA_VISIBLE_DEVICES="" python -u main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_${seed} 2>&1 "
done

corruption="omniscient"
for frac in 0.01 0.25
do
    cmds="$cmds ; sleep 90s; sleep 3m ; sleep 3m; sleep 3m"
    cmds="$cmds ; time CUDA_VISIBLE_DEVICES="" python -u main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_${seed} 2>&1 "
done


done  # seed


echo "executing..."
set +u # for parallel exec to work (unbound variables)
f_ParallelExec $njobs "$cmds"

echo "Done at $(date)"
