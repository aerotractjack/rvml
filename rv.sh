#!/bin/bash

train() {
    rastervision run local train.py -a config $1
}

predict() {
    python3 predict.py $1
}

usage() {
    echo -e "USAGE:\nTraining:\n\t$ rv.sh train /path/to/train.config\nPredict:\n\t$ rv.sh predict /path/to/predict.config"
}

if [[ "$1" = "-h" || "$1" = "--help" || "$1" = "--usage" || "$1" = "" ]]; then
	usage
    exit 0
fi

$@