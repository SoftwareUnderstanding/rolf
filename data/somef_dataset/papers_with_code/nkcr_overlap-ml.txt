# Overlapping with language modelling and emotion detection

Pytorch implementation to reproduce experiments from "[Alleviating Sequence Information Loss with Data Overlapping and Prime Batch Sizes](https://arxiv.org/abs/1909.08700)" - ([poster](poster-conll.pdf)).

If you use this code or our results in your research, please cite as appropriate:

```
@inproceedings{kocher-etal-2019-alleviating,
    title = "Alleviating Sequence Information Loss with Data Overlapping and Prime Batch Sizes",
    author = "Kocher, No{\'e}mien  and
      Scuito, Christian  and
      Tarantino, Lorenzo  and
      Lazaridis, Alexandros  and
      Fischer, Andreas  and
      Musat, Claudiu",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/K19-1083",
    doi = "10.18653/v1/K19-1083",
    pages = "890--899",
}
```

This repo holds experiments on 4 models using the "overlapping" method:

- **awd**, [AWD](https://arxiv.org/abs/1708.02182) ASGD Weight-Dropped LSTM, (`/awd`)
- **text simple**, a very basic lstm for language modelling, (`/simple`)
- **mos**, [MOS](https://arxiv.org/abs/1711.03953) Mixture of Softmaxes, (`/mos`)
- **voice simple**, a very basic LSTM for emotion detection on voice, (`/emotions`)

To specify which model to run, use `--main-model {simple-lstm | awd-lstm | mos-lstm | emotions-simple-lstm}` argument. There are additional common paramaters, as well as specific parameters for each model. Those can be found in `main_run.py`.

The taxonomy in the code may differe a bit from the paper, especially regarding the type of experiments. Here is the corresponding terms:

|In the code|In the paper|
|-----------|------------|
|No order|Extreme TOI|
|Local order|Inter-batch TOI|
|Standard order|Standard TOI|
|Total order (P)|Alleviated TOI (P)|

Experiments were run on a Tesla P100 GPU. Results are very likely to differ based on the GPU used.

## Set-up

Download the data (PTB, WT2, WT103):

```bash
chmod +x get_data.sh
./get_data.sh
```

For emotions, add in `data/IEMOCAP/` the `all_features_cv` files.

We use python `3.6` with Pytorch `0.4.1`. To create a new python environement and install dependencies, run:

```bash
python3 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

You can check your setup by launching a quick training over one epoch with the following command:

```bash
python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --epochs 1 --nhid 5 --emsize 5 --nlayers 1 --bptt 5
```

The program should exit without error and write the logs in the `logs/` folder. You can watch the logs with tensorboard by launching the following command:

```bash
tensorboard --logdir logs/
```

## About the files

`main_run.py` is the main entry point that parses arguments, does the global initialization and runs the corresponding model and task.

`awd/`, `emotions/`, `mos/` and `simple/` are the different models directories. `common/` holds the common initilization and utilities, such as the different data iterators, which are in the `DataSelector` class in `common/excavator.py`.

The `main_run.py` file, after performing the common initilizations, imports the `main.py` file corresponding to the choosen model.


# Commands to reproduce the experiments

**Note**: Those results do not use prime batch size, but the default parameters. To have better results, adapt the `--batch-size` param to the closest prime number.

**Quick anchors navigation**:

<table>
    <tr>
        <th>Model</th><th>Dataset</th><th>Experiments</th>
    </tr>
    <tr>
        <td rowspan="3">AWD</td>
        <td>PTB</td>
        <td><a href="#awd-ptb">Extreme / Inter-batch / Original / Alleviated TOI</a></td>
    </tr>
        <td>WT2</td>
        <td><a href="#awd-wt2">Extreme / Inter-batch / Original / Alleviated TOI</a></td>
    </tr>
    <tr>
        <td>WT103</td>
        <td><a href="#awd-wt103">Extreme / Inter-batch / Original / Alleviated TOI</a></td>
    </tr>
    <tr>
        <td rowspan="2">Text simple LSTM</td>
        <td>PTB</td>
        <td><a href="#simple-ptb">Extreme / Inter-batch / Original / Alleviated TOI</a></td>
    </tr>
        <td>WT2</td>
        <td><a href="#simple-wt2">Extreme / Inter-batch / Original / Alleviated TOI</a></td>
    </tr>
    <tr>
        <td rowspan="1">MOS</td>
        <td>PTB</td>
        <td><a href="#mos-ptb">Original / Alleviated TOI</a></td>
    </tr>
    <tr>
        <td rowspan="1">Voice simple LSTM</td>
        <td>IEMOCAP</td>
        <td><a href="#voice-simple-lstm">Extreme / Inter-batch / Original / Alleviated TOI</a></td>
    </tr>
</table>

## AWD PTB

**Extreme TOI**:

Expected results: `66.38` / `63.49` (validation / testing)

```bash
python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --seed-shuffle 141 --epochs 1000 --shuffle-full-seq
```

**Inter-batch TOI**:

Expected results: `66.96` / `64.20` (validation / testing)

```bash
python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --seed-shuffle 141 --epochs 1000 --shuffle-row-seq
```

**Standard TOI**:

Expected results: `61.28` / `58.94` (validation / testing)

```bash
python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epochs 1000
```

**Alleviated TOI {2,5,7,10}**:

Expected results (validation / testing): 

* 2: `61.73` / `59.37`
* 5: `63.37` / `60.50`
* 7: `59.22` / `56.7`
* 10: `68.09` / `65.88`

```bash
overlaps=(2 5 7 10)
epochs=1000
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epochs "$(($epochs/$k))" --init-seq "overlapCN_${k}"
    sleep 10
done
```

💥 With a prime batch size:

Expected results (validation / testing): 

* 2: `60.56` / `57.97`
* 5: `59.52` / `57.14`
* 7: `59.43` / `57.16`
* 10: `58.96` / **`56.46`**

```bash
overlaps=(2 5 7 10)
epochs=1000
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model awd-lstm --batch-size 19 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epochs "$(($epochs/$k))" --init-seq "overlapCN_${k}"
    sleep 10
done
```

## AWD WT2

**Extreme TOI**

Expected results: `77.14` / `73.52` (validation / testing)

```bash
python3 main_run.py --main-model awd-lstm --epochs 750 --data /data/noemien.kocher/datasets/wikitext-2 --dropouth 0.2 --seed 1882 --batch-size 80 --shuffle-full-seq
```

**Inter-batch TOI**

Expected results: `76.08` / `72.61` (validation / testing)

```bash
python main_run.py --main-model awd-lstm --epochs 750 --data /data/noemien.kocher/datasets/wikitext-2 --dropouth 0.2 --seed 1882 --batch-size 80 --shuffle-row-seq
```

**Standard TOI**

Expected results: `68.50` / `65.86` (validation / testing)

```bash
python3 main_run.py --main-model awd-lstm --epochs 750 --data /data/noemien.kocher/datasets/wikitext-2 --dropouth 0.2 --seed 1882 --batch-size 80
```

**Alleviated TOI {2,5,7,10}**

Expected results (validation / testing): 

* 2: `68.56` / `65.51`
* 5: `69.56` / `66.33`
* 7: `67.48` / `64.87`
* 10: `72.95` / `69.69`

```bash
overlaps=(2 5 7 10)
epochs=750
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model awd-lstm --data /data/noemien.kocher/datasets/wikitext-2 --dropouth 0.2 --seed 1882 --batch-size 80 --epochs "$(($epochs/$k))" --init-seq "overlapCN_${k}"
    sleep 10
done
```

💥 With a prime batch size:

Expected results (validation / testing): 

* 2: `68.11` / `65.14`
* 5: `67.74` / `65.11`
* 7: `67.79` / `64.79`
* 10: `67.47` / **`64.73`**

```bash
overlaps=(2 5 7 10)
epochs=750
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model awd-lstm --data /data/noemien.kocher/datasets/wikitext-2 --dropouth 0.2 --seed 1882 --batch-size 79 --epochs "$(($epochs/$k))" --init-seq "overlapCN_${k}"
    sleep 10
done
```

## AWD WT103

**Extreme TOI**

Expected results: `35.22` / `36.19` (validation / testing)

```bash
python3 -u main_run.py --main-model awd-lstm --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch-size 60 --optimizer adam --lr 1e-3 --data /data/noemien.kocher/datasets/wikitext-103 --when 12 --model QRNN --shuffle-full-seq
```

**Inter-batch TOI**

Expected results: `35.41` / `36.39` (validation / testing)

```bash
python3 -u main_run.py --main-model awd-lstm --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch-size 60 --optimizer adam --lr 1e-3 --data /data/noemien.kocher/datasets/wikitext-103 --when 12 --model QRNN --shuffle-row-seq
```

**Standard TOI**

Expected results: `32.18` / `32.94` (validation / testing)

```bash
python3 -u main_run.py --main-model awd-lstm --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch-size 60 --optimizer adam --lr 1e-3 --data /data/noemien.kocher/datasets/wikitext-103 --when 12 --model QRNN
```

**Alleviated TOI {2,5,7,10}**

Expected results (validation / testing): 

* 2: `36.94` / `34.31`
* 5: `38.50` / `40.04`
* 7: `31.78` / `32.72`
* 10: `48.28` / `49.49`

```bash
# base num epochs is 14
overlaps=(2 5 7 10)
when_steps=147456
max_steps=172032
for i in "${!overlaps[@]}"
do
        :
        python3 -u main_run.py --main-model awd-lstm --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch-size 60 --optimizer adam --lr 1e-3 --data /data/noemien.kocher/datasets/wikitext-103 --when-steps "$when_steps" --model QRNN --init-seq "overlapCN_${overlaps[$i]}" --log-dir /data/noemien.kocher/logs/ --max-steps "$max_steps"
        sleep 10
done
```

💥 With a prime batch size:

Expected results (validation / testing): 

* 2: `32.00` / `32.98`
* 5: `31.93` / `33.07`
* 7: `31.78` / `32.89`
* 10: `31.92` / **`32.85`**

```bash
# base num epochs is 14
overlaps=(2 5 7 10)
when_steps=147456
max_steps=172032
for i in "${!overlaps[@]}"
do
        :
        python3 -u main_run.py --main-model awd-lstm --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch-size 59 --optimizer adam --lr 1e-3 --data /data/noemien.kocher/datasets/wikitext-103 --when-steps "$when_steps" --model QRNN --init-seq "overlapCN_${overlaps[$i]}" --log-dir /data/noemien.kocher/logs/ --max-steps "$max_steps"
        sleep 10
done
```

## Simple PTB

**Extreme TOI**:

Expected results: `81.97` / `79.08` (validation / testing)

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 20 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1500 --lr-decay 1 --shuffle-full-seq
```

**Inter-batch TOI**:

Expected results: `81.67` / `78.59` (validation / testing)

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 20 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1500 --lr-decay 1 --shuffle-row-seq
```

**Standard TOI**:

Expected results: `77.54` / `75.36` (validation / testing)

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 20 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1500 --lr-decay 1
```

**Alleviated TOI {2,5,7,10}**:

Expected results (validation / testing): 

* 2: `78.48` / `76.55`
* 5: `91.95` / `89.64`
* 7: `77.47` / `74.98`
* 10: `92.92` / `92.07`

```bash
overlaps=(2 5 7 10)
epochs=100
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model simple-lstm --epochs "$(($epochs/$k))" --batch-size 20 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1500 --lr-decay 1 --init-seq "overlapCN_${k}"
    sleep 10
done
```

## Simple WT2

**Extreme TOI**

Expected results: `101.3` / `96.08` (validation / testing)

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 80 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1150 --lr-decay 1 --data /data/noemien.kocher/datasets/wikitext-2 --shuffle-full-seq
```

**Inter-batch TOI**

Expected results: `101.7` / `96.89` (validation / testing)

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 80 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1150 --lr-decay 1 --data /data/noemien.kocher/datasets/wikitext-2 --shuffle-row-seq
```

**Standard TOI**

Expected results: `98.85` / `93.15` (validation / testing)

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 80 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1150 --lr-decay 1 --data /data/noemien.kocher/datasets/wikitext-2
```

**Alleviated TOI {2,5,7,10}**

Expected results (validation / testing): 

* 2: `100.4` / `94.49`
* 5: `113.5` / `106.1`
* 7: `98.25` / `92.77`
* 10: `151.0` / `135.1`

```bash
overlaps=(2 5 7 10)
epochs=100
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model simple-lstm --epochs "$(($epochs/$k))" --batch-size 80 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1150 --lr-decay 1 --data /data/noemien.kocher/datasets/wikitext-2 --init-seq "overlapCN_${k}"
    sleep 10
done
```

## MOS PTB

**Standard TOI**:

Expected results: `58.49` / `56.19` (validation / testing)

```bash
python3 main_run.py --main-model mos-lstm --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch-size 12 --lr 20.0 --epochs 1000 --nhid 960 --nhidlast 620 --emsize 280 --n-experts 15
```

**Alleviated TOI {1..40}**:

💥 With a prime batch size:

```bash
epochs=2000
for k in {1..70}
do
        :
        python3 main_run.py --main-model mos-lstm --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch-size 13 --lr 20.0 --epochs "$(($epochs/$k))" --nhid 960 --nhidlast 620 --emsize 280 --n-experts 15 --init-seq "overlapCNF_${k}"
        sleep 10
done
```

Expected results (validation / testing): 

* 1: `58.36` /	`56.21`
* 2: `58.07` /	`55.76`
* 3: `58.03` /	`55.79`
* 4: `52.82` /	`55.63`
* 5: `57.81` /	`55.63`
* 6: `57.55` /	`55.32`
* 7: `57.47` /	`55.23`
* 8: `57.47` /	`55.34`
* 9: `57.16` /	`54.93`
* 10: `57.34` / `54.90`
* 11: `57.11` / `54.98`
* 12: `57.47` / `55.44`
* 13: `67.77` / `66.01`
* 14: `56.76` / **`54.58`** (paper's result)
* 15: `57.44` / `55.20`
* 16: `56.95` / `54.86`
* 17: `57.64` / `55.14`
* 18: `57.38` / `54.93`
* 19: `57.55` / `55.35`
* 20: `57.00` / `54.67`
* 21: `57.55` / `55.22`
* 22: `57.54` / `55.19`
* 23: `57.29` / `54.90`
* 24: `57.47` / `55.11`
* 25: `57.12` / `54.85`
* 26: `66.14` / `63.81`
* 27: `57.08` / `54.85`
* 28: `--.--` / `--.--`
* 29: `--.--` / `--.--`
* 30: `--.--` / `--.--`
* 31: `57.74` / `55.37`
* 32: `57.21` / `55.26`
* 33: `57.66` / `55.40`
* 34: `57.48` / `55.44`
* 35: `56.44` / **`54.33`** (post-result, not in the paper)
* 36: `57.10` / `55.09`
* 37: `57.55` / `55.29`
* 38: `57.04` / `54.87`
* 39: `64.37` / `62.54`
* 40: `57.52` / `54.99`

## Voice simple LSTM

**Extreme TOI**:

Expected result: `0.475` / `0.377` (WA / UA)

```bash
python3 main_run.py --main-model emotions-simple-lstm --cv 5 --data data/IEMOCAP/all_features_cv --test-batch-size 20 --lr 0.05 --log-interval 20 --lr-decay 1 --step-size 0.1 --epochs 60 --order complete_random
```

**Inter-batch TOI**:

Expected result: `0.478` / `0.386` (WA / UA)

```bash
python3 main_run.py --main-model emotions-simple-lstm --cv 5 --data data/IEMOCAP/all_features_cv --test-batch-size 20 --lr 0.05 --log-interval 20 --lr-decay 1 --step-size 0.1 --epochs 60 --window-size 300 --order local_order
```

**Standard TOI**:

Expected result: `0.486` / `0.404` (WA / UA)

```bash
python3 main_run.py --main-model emotions-simple-lstm --cv 5 --data data/IEMOCAP/all_features_cv --test-batch-size 20 --lr 0.05 --log-interval 20 --lr-decay 1 --step-size 0.1 --epochs 60 --order standard_order
```

**Alleviated TOI 10**:

Expected result: 

* 15k steps: `0.553` / `0.489` (WA / UA)
* 60 epochs: `0.591` / `0.523` (WA / UA)

```bash
python3 main_run.py --main-model emotions-simple-lstm --cv 5 --data data/IEMOCAP/all_features_cv --test-batch-size 20 --lr 0.05 --log-interval 20 --lr-decay 1 --step-size 0.1 --epochs 60 --order total_order
```

## Delayed-reset standard TOI {1,2,5,7,10} with PTB

Expected results (validation / testing): 

* 1: `61.28` / `58.94`
* 2: `60.76` / `58.55`
* 5: `60.10` / `57.83`
* 7: `60.08` / `57.76`
* 10: `60.05` / `57.78`

```bash
P=(1 2 5 7 10)
epochs=1000
for k in "${P[@]}"
do
    :
    python3 main_run.py --main-model awd-lstm-repetitions --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epochs 1000 --use-repetitions "${k}"
    sleep 10
done
```

# Acknowledgements

Code is heavily borrowed from the following sources:

- simple-lstm (`simple/`): https://github.com/deeplearningathome/pytorch-language-model
- awd-lstm (`awd/`): https://github.com/salesforce/awd-lstm-lm
- mos-lstm: (`mos/`) https://github.com/zihangdai/mos
