#!/bin/bash

python3 src/collect_readmes.py --input_mode csvfile --input data/awesome_lists_links/awesome_lists.csv --githublinks_file data/awesome_lists_links/repos1.csv --link_mode awesomelist --readme_folder data/awesome_lists_links/readme/ --outfile data
python3 src/main.py --preprocess --preprocess_file data/readme_semantic_web.csv
python3 src/main.py --train_test_split --train_test_file data/readme_semantic_web_preprocessed.csv
python3 src/main.py --merge_csv --output_csv data/readme_base_semantic_web_preprocessed_test.csv --files data/readme_new_preprocessed_test.csv data/readme_semantic_web_preprocessed_test.csv
python3 src/main.py --merge_csv --output_csv data/readme_base_semantic_web_preprocessed_train.csv --files data/readme_new_preprocessed_train.csv data/readme_semantic_web_preprocessed_train.csv
python3 src/main.py --predict --input results/models/demo1/ --data_path data/readme_base_semantic_web_preprocessed_test.csv --output_csv results/predictions/demo1_predictions.csv
python3 src/main.py --evaluate --input results/predictions/demo1_predictions.csv
python3 src/collect_readmes.py --input_mode csvfile --input data/awesome_lists_links/awesome_lists.csv --githublinks_file data/awesome_lists_links/repos1.csv --link_mode awesomelist --readme_folder data/awesome_lists_links/readme/ --outfolder data --redownload
python3 src/collect_readmes.py --input_mode csvfile --input data/awesome_lists_links/awesome_lists.csv --githublinks_file data/awesome_lists_links/repos1.csv --link_mode awesomelist --readme_folder data/awesome_lists_links/readme/ --outfolder data --redownload
python3 src/main.py --preprocess --preprocess_file data/readme_c_c++.csv
python3 src/main.py --train_test_split --train_test_file data/readme_c_c++_preprocessed.csv
python3 src/main.py --merge_csv --output_csv data/readme_base_semantic_web_c_preprocessed_train.csv --files data/readme_new_preprocessed_train.csv data/readme_semantic_web_preprocessed_train.csv data/readme_c_c++_preprocessed_train.csv
python3 src/main.py --merge_csv --output_csv data/readme_base_semantic_web_c_preprocessed_test.csv --files data/readme_new_preprocessed_test.csv data/readme_semantic_web_preprocessed_test.csv data/readme_c_c++_preprocessed_test.csv
python3 src/main.py --train_models --train_set data/readme_base_semantic_web_c_preprocessed_train.csv --test_set data/readme_base_semantic_web_c_preprocessed_test.csv --additional_categories "Semantic web" "C C++"
python3 src/main.py --predict --input results/models/demo_semantic_web_c --data_path data/readme_base_semantic_web_preprocessed_c_test.csv --output_csv results/predictions/demo_semantic_web_c_predictions.csv
python3 src/main.py --predict --input results/models/demo_semantic_web_c --data_path data/readme_base_semantic_web_c_preprocessed_test.csv --output_csv results/predictions/demo_semantic_web_c_predictions.csv
python3 src/main.py --evaluate --input results/predictions/demo_semantic_web_c_predictions.csv