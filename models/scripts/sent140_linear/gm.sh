#! /usr/bin/bash

export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source scripts/parallelize.sh
set -u  # x: stack trace
export CUDA_VISIBLE_DEVICES=""

cmds=""

echo "Starting at $(date)"

njobs=20
num_rounds=1000
batch_size=10
num_epochs=1
clients_per_round=50
lr=0.005
reg=0.0
model="erm_log_reg"
niter=2

dataset="sent140"
aggregation="geom_median"
aggregation_out="gm"

outf="outputs/sent140/outputs"
logf="outputs/sent140/log"

main_args=" -dataset sent140 -model erm_log_reg -lr $lr "
options=" --num-rounds ${num_rounds} --eval-every 50 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} -reg_param $reg "
options="${options} --weiszfeld-maxiter ${niter} --aggregation $aggregation "

for seed in 1 2 3 4 5
do
corruption="clean"

cmds="$cmds ; sleep 90s; sleep 90s ; sleep 90s; sleep 90s"
cmds="$cmds ; time python -u main.py ${main_args} $options  --seed $seed --output_summary_file ${outf}_${aggregation_out}_${corruption}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${seed} 2>&1 "

corruption="flip"
for frac in 0.01 0.05 0.1 0.25 0.45 0.75
do
    cmds="$cmds ; sleep 90s; sleep 90s ; sleep 90s; sleep 90s"
    cmds="$cmds ; time CUDA_VISIBLE_DEVICES="" python -u main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_${seed} 2>&1 "
done

corruption="gauss"
for frac in 0.01 0.05 0.1 0.25 0.45 0.75
do
    cmds="$cmds ; sleep 90s; sleep 90s ; sleep 90s; sleep 90s"
    cmds="$cmds ; time CUDA_VISIBLE_DEVICES="" python -u main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_${seed} 2>&1 "
done


done  # seed


echo "executing..."
set +u # for parallel exec to work (unbound variables)
f_ParallelExec $njobs "$cmds"

echo "Done at $(date)"
