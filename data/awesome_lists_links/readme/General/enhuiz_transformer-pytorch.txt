# Transformer-pytorch

A simple implementation of the neural machine translation framework using pytorch.

## Models

- Transformer (https://arxiv.org/abs/1706.03762)

## Datasets

- **Xiaoshi**: A Chinese to Chinese traditional poetry dataset (https://github.com/enhuiz/XiaoShi).

- **Multi30K (En-De)**: WMT'17 Multimodal Translation task (En-De) (https://www.statmt.org/wmt17/multimodal-task.html).

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Train

```bash
python scripts/train.py config/xiaoshi/transformer.yml
```

### Validate

```bash
python scripts/validate.py config/xiaoshi/transformer.yml
```

### Test

```bash
python scripts/test.py config/xiaoshi/transformer.yml
```

## Results

### Multi30k (En-De)

#### Loss & Perplexity

![](fig/multi30k-loss.png)

![](fig/multi30k-ppl.png)

#### Scores

> Evaluation tool: https://github.com/jhclark/multeval

| Model       | BLEU (val/test) | METEOR (val/test) | TER (val/test) |
| ----------- | --------------- | ----------------- | -------------- |
| Transformer | 35.3/35.7       | 56.0/55.0         | 49.0/48.2      |

#### Examples

| En                                                                        | De (Model)                                                          | De (Ground Truth)                                                      |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| A man in an orange hat starring at something                              | ein mann mit orangefarbenem hut starrt etwas an                     | ein mann mit einem orangefarbenen hut der etwas <unk>                  |
| A Boston Terrier is running on lush green grass in front of a white fence | ein kampfrichter rennt auf einer grünen wiese vor einem weißen zaun | ein boston terrier läuft über saftig grünes gras vor einem weißen zaun |
| A girl in karate uniform breaking a stick with a front kick               | ein mädchen in karateanzügen folgt einem stock mit einem fußtritt   | ein mädchen in einem karateanzug bricht einen stock mit einem tritt    |

### Xiaoshi (Poetry-Chinese)

#### Loss & Perplexity

![](fig/xiaoshi-loss.png)

![](fig/xiaoshi-ppl.png)

#### Scores

| Model       | BLEU | TER   |
| ----------- | ---- | ----- |
| Transformer | 0.7  | 127.5 |

#### Examples

| Poetry                           | Chinese (Model)                                                      | Chinese (Ground Truth)                                                                                               |
| -------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 繖幄垂垂马踏沙，水长山远路多花。 | 黄 昏 时 分 ， 山 路 旁 边 的 树 木 ， 水 面 对 着 竹 林 。          | 马 后 垂 着 伞 盖 ， 马 蹄 踩 着 黄 沙 ， 沙 沙 ， 沙 沙 。 山 遥 遥 ， 水 茫 茫 ， 沿 路 见 到 那 么 多 的 野 花 。 |
| 眼中形势胸中策，缓步徐行静不哗。 | 当 年 的 文 章 中 之 后 ， 他 们 不 知 道 他 们 。                   | 眼 观 敌 我 形 势 ， 战 术 方 略 早 已 成 竹 在 胸 。 兵 马 缓 步 前 进 ， 三 军 肃 静 无 人 喧 哗 。                |
| 满腹诗书漫古今，频年流落易伤心。 | 当 年 我 们 在 这 时 节 的 时 候 ， 我 心 情 况 是 我 的 心 情 郎 。 | 装 满 一 肚 子 诗 书 ， 博 古 通 今 。 连 年 流 落 他 乡 ， 最 易 伤 情 。                                           |

## Write Your Own Configuration

Please check config/your-dataset-name/your-config.yml for details if you have your own dataset.

## Acknowledgement

- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://github.com/OpenNMT/OpenNMT-py


