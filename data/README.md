# Commands to download datasets

Go to the appropriate folder and run the commands given below. 
The test set is created by holding out a fraction of each user's data.

## FEMINST
Run
> time ./preprocess.sh -s niid --sf 1.0 -k 100 -t sample

Takes about 30-60 minutes to run and occupies 25GB on disk.


## SENT140
Run
> time ./preprocess.sh -s niid --sf 1.0 -k 50 -t sample --tf 0.8

Occupies 1.2GB on disk.


## SHAKESPEARE
Run
> time ./preprocess.sh -s niid --sf 1.0 -k 100 -t sample --tf 0.8 

Runs under a minute and occupies 287MB.
