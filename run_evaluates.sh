#!/bin/bash

poetry shell

for i in $(seq 0.005 0.005 0.045); do
python src/Evaluation/evaluation.py --inputfile best_test/asd5_$i.csv --outfile best_test/asd5_$i.json --all_categories Audio "Computer Vision" Graphs "Natural Language Processing" "Reinforcement Learning" Sequential
done

