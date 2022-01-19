#!/usr/bin/env bash

cd sent140

if [ ! -f 'glove.6B.50d.txt' ]; then
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    rm  glove.6B.100d.txt glove.6B.200d.txt glove.6B.zip
fi

if [ ! -f embs.json ]; then
    python3 get_embs.py -f glove.6B.50d.txt
fi

if [ ! -f embs300.json ]; then
    python3 get_embs.py -f glove.6B.300d.txt -o embs300.json
fi
