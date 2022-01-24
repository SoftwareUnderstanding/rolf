# Translation

Using current State of the Art models for translating english to french.

Currently hosted live at https://whispering-brook-50228.herokuapp.com/

`Note: Small models are hosted because of 500 MB memory limit.`

Dataset used is taken from manything.org(link -> http://www.manythings.org/anki/fra-eng.zip)

Files Info:

    main.py:Runs end to end project
    Serialize.py:Create standalone script from all modules(easy to run on colab,kaggle)
    Dockerfile:Docker configs
    app.py:For hosting on Heroku

How to run:

`Important:Serialize.py is designed to create standalone script that can run on cloud compute services(like kaggle, colab)
If you want to run locally configure main.py`

Command to compile single script

`python serialize.py --files_to_copy=data_loading.py,preprocessing.py,utils.py,modules.py,transformer.py --file_impo_lines=4,8 --main_parser_lines=11,34 --filename="fra.txt" --vocab_words=9000 --batch_size=256 --epochs=30 --log_dir=logs/ --d_model=128 --d_inner=64 --d_key=64 --heads=5 --layers=3 --dropout=0.2 --info="30 epochs model 128 inner 64 heads 5 layers 3 dropout 0.2" --kernel_name="transformer-auto"`

Models used:

    1.Attention model(link -> https://arxiv.org/abs/1409.0473)
    2.Transformer model(link -> https://arxiv.org/abs/1706.03762)
