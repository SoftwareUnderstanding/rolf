# Image Captioning with Bimodal Transformers

In this notebook we are going to be using COCO captioned image data to build a model that produces natural language descriptions of given images.

We will be using a InceptionV3 convolutional neural network pretrained on classifying imagenet images and an ALBERT transformer network pretrained on a general language modelling task.

We will construct the bimodal transformer to aggregate image and language information. The following is an outline of the model architecture:

<img src="bmt_architecture.png" width="600">

## Running this notebook

The conda environment is provided in the `env.yml`, which should be enough to recreate the set-up. 
Note: the code was written with the GPU version of tensorflow, which has some discrepencies with the CPU-only version. 

## Imports

```python
from datetime import datetime 
from functools import partial
from itertools import chain, cycle
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

import cv2
import tensorflow as tf
import transformers

print('Physical Devices:\n', tf.config.list_physical_devices(), '\n')
%load_ext tensorboard

DATASETS_PATH = os.environ['DATASETS_PATH']
MSCOCO_PATH = f'{DATASETS_PATH}/MSCOCO_2017'
COCO_VERSION = 'val2017'
LOAD_COCO_PREPROC_CACHE_IF_EXISTS = True

PRETRAINED_TRANSFORMER_VERSION = 'albert-base-v2'
Transformer = transformers.TFAlbertModel
Tokenizer = transformers.AlbertTokenizer
MLMHead = transformers.modeling_tf_albert.TFAlbertMLMHead

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUTS_DIR = f'./outputs/{stamp}'
print('\nOutput Directory:', OUTPUTS_DIR)
```

    Physical Devices:
     [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')] 
    
    The tensorboard extension is already loaded. To reload it, use:
      %reload_ext tensorboard
    
    Output Directory: ./outputs/20200424-000059
    


```python
log_dir = f'{OUTPUTS_DIR}/logs'
summary_writer = tf.summary.create_file_writer(log_dir)
tf.summary.trace_on(graph=True) 
print(log_dir)
```

    WARNING:tensorflow:Trace already enabled
    ./outputs/20200424-000059/logs
    

## Loading COCO

Firstly, we are going to load the COCO data. For the sakes of this demonstration I am running this on my limited machine, as such we will only load in the 5000 validation samples. 


```python
captions_path = f'{MSCOCO_PATH}/annotations/captions_{COCO_VERSION}.json'
coco_captions = json.loads(Path(captions_path).read_text())
coco_captions = {
    item['image_id']: item['caption']
    for item in coco_captions['annotations']
}
```


```python
images_path = f'{MSCOCO_PATH}/{COCO_VERSION}'
image_paths = Path(images_path).glob('*.jpg')
coco_imgs = {
    int(path.name.split('.')[0]): cv2.imread(str(path))
    for path in image_paths
}
```


```python
coco_data: List[Tuple[np.ndarray, str]] = [
    (img_id, coco_imgs[img_id], coco_captions[img_id])
    for img_id in coco_imgs
    if img_id in coco_captions
]
```


```python
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for i, ax in enumerate(chain(list(axs.flatten()))):
    img_id, img, cap = coco_data[i]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f'{img_id}: {cap}')
    ax.axis('off')
plt.show()
```


![png](output_7_0.png)


## Preprocessing the Data

In order to reduce the amount of learning that our model needs to do we are using the pretrained InceptionV3 model. To do this we will pass each of our images through InceptionV3 and extract the activations in the penultimate layers. These extracted tensors are our "image embeddings" and they capture the semantic content of the input in a ready-to-use state for downstream tasks. 

Additionally, our transformer network, ALBERT, only ingests sentences that have been tokenized. As such, we extract "caption encodings" using the tokenizer. This replaces (approximate) morphemes in the sentence with assigned identifiers. 

Once we have preprocessed the data we will store it in a cache for retrieval in the future. Additionally this helps relieve memory limitations as we can now discard the InceptionV3 model for the training phase. In order to keep track of the information in the cache we will also store the COCO image identifier (IID). 


```python
def inceptionv3_preprocess(img, img_size=(128, 129)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return tf.keras.applications.inception_v3.preprocess_input(img)
```


```python
def create_image_features_extract_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    return tf.keras.Model(new_input, hidden_layer)
```


```python
tokenizer = Tokenizer.from_pretrained(PRETRAINED_TRANSFORMER_VERSION)
```


```python
print('Vocab size:', tokenizer.vocab_size)
for (name, token), iden in zip(tokenizer.special_tokens_map.items(), 
                               tokenizer.all_special_ids):
    print(f'{name}: {token}, ID: {iden}')
print()
print('Input:', cap)
print('Encoded:', tokenizer.encode(cap))
```

    Vocab size: 30000
    bos_token: [CLS], ID: 2
    eos_token: [SEP], ID: 3
    unk_token: <unk>, ID: 0
    sep_token: [SEP], ID: 4
    pad_token: <pad>, ID: 1
    
    Input: a person on skis makes her way through the snow
    Encoded: [2, 21, 840, 27, 7185, 18, 1364, 36, 161, 120, 14, 2224, 3]
    


```python
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def coco_preprocess_batch(coco_batch: list, 
                          image_feature_extract_model: tf.keras.Model,
                          tokenizer: Tokenizer):
    iids = [iid for iid, _, _ in coco_batch]
    imgs = [img for _, img, _ in coco_batch]
    caps = [cap for _, _, cap in coco_batch]
    
    cap_encodings = [tokenizer.encode(cap) for cap in caps]
    x = np.array([inceptionv3_preprocess(img) for img in imgs])
    img_embeddings = image_feature_extract_model(x)
    return list(zip(iids, img_embeddings, cap_encodings))

def make_coco_preprocessed(tokenizer: Tokenizer,
                           batch_size=4):
    image_feature_extract_model = create_image_features_extract_model()
    return [
        record
        for coco_batch in batch(coco_data, batch_size)
        for record in coco_preprocess_batch(coco_batch,
                                            image_feature_extract_model,
                                            tokenizer)
    ]
```

Tensorflow does not have a built-in way to release a model from memory so we have to spin-up a concurrent process using `multiprocess` that does the preprocessing. When this process terminates it will clear the memory.


```python
preprocessed_coco_cache_dir = Path(f'{MSCOCO_PATH}/inception-{PRETRAINED_TRANSFORMER_VERSION}-preprocessed')
preprocessed_coco_cache_dir.mkdir(exist_ok=True)
preprocessed_coco_cache_path = preprocessed_coco_cache_dir / f'{COCO_VERSION}.npy'
if preprocessed_coco_cache_path.exists():
    print('Loading cached preprocessed data...')
    coco_preprocessed = list(np.load(preprocessed_coco_cache_path, 
                                     allow_pickle=True))
    coco_preprocessed = [tuple(x) for x in coco_preprocessed]
else:
    print('Preprocessing and creating cache...')
    def preprocess_and_cache():
        coco_preprocessed = make_coco_preprocessed(tokenizer)
        np.save(preprocessed_coco_cache_path, np.array(coco_preprocessed))
    
    # Using multiprocess to clear Inception GPU usage when process terminates
    p = multiprocessing.Process(target=preprocess_and_cache)
    p.start()
    p.join()
```

    Loading cached preprocessed data...
    


```python
coco_preprocessed[0]
```




    (139,
     <tf.Tensor: shape=(2, 2, 2048), dtype=float32, numpy=
     array([[[2.5995767 , 0.        , 0.4370903 , ..., 0.7036782 ,
              1.0261441 , 3.478441  ],
             [3.5346863 , 0.        , 0.48941162, ..., 0.7036782 ,
              1.0261441 , 3.478441  ]],
     
            [[1.6772687 , 0.        , 2.2373395 , ..., 0.7036782 ,
              1.0261441 , 3.478441  ],
             [1.7453028 , 0.        , 0.43314373, ..., 0.7036782 ,
              1.0261441 , 3.478441  ]]], dtype=float32)>,
     [2, 21, 634, 217, 29, 21, 633, 17, 21, 859, 3])



## Creating Train-Test Datasets

In order to train our model we will create train and test datasets from our preprocessed data. The training data will be organised in batches where sequences of varying lengths are collected into matrices. The matrices are filled by padding the shorter sequences with a special token defined by the tokenizer. Later the transformer will be instructed to not pay attention to these padding tokens.


```python
coco_train_data, coco_test_data = train_test_split(coco_preprocessed, test_size=0.2)
outputs = (tf.int32, tf.float32, tf.int32) 

BUFFER_SIZE = 10000
BATCH_SIZE = 8

coco_train = tf.data.Dataset.from_generator(lambda: cycle(coco_train_data), outputs)

# example sample to define the padding shapes
iid_ex, img_emb_ex, cap_enc_ex = next(iter(coco_train))

coco_train = coco_train.shuffle(BUFFER_SIZE)
coco_train = coco_train.padded_batch(
    BATCH_SIZE, 
    padded_shapes=(iid_ex.shape, img_emb_ex.shape, [None]), 
    padding_values=(0, 0.0, tokenizer.pad_token_id)
)

coco_test = tf.data.Dataset.from_generator(lambda: cycle(coco_test_data), outputs)
```


```python
coco_train
```




    <PaddedBatchDataset shapes: ((None,), (None, 2, 2, 2048), (None, None)), types: (tf.int32, tf.float32, tf.int32)>



## Masked Language Modelling

In order to train the caption generator we will learn through a masked language modelling (MLM) scheme. This involves arbitrarily removing tokens in a caption and having the model reconstruct the input. This process is inspired by the [Cloze deletion test](https://en.wikipedia.org/wiki/Cloze_test) from psychology. MLM was introducted by the authors of BERT.


```python
def create_mask_and_input(tar: tf.Tensor, 
                          tokenizer: Tokenizer,
                          prob_mask=0.15,
                          seed=None) -> tf.Tensor:
    """
    prob_mask hyperparams from: https://arxiv.org/pdf/1810.04805.pdf
    """
    if seed is not None:
        tf.random.set_seed(seed)
        
    where_masked = tf.random.uniform(tar.shape) < prob_mask
    for special_token in tokenizer.all_special_ids:
        where_masked &= tar != special_token
    
    mask_tokens = tf.multiply(tokenizer.mask_token_id, 
                              tf.cast(where_masked, tf.int32))
    not_masked = tf.multiply(tar, 1 - tf.cast(where_masked, tf.int32))
    inp = mask_tokens + not_masked
    
    return inp, where_masked
```


```python
iid, img_emb, cap_enc = next(iter(coco_train))
train_cap_enc, where_masked = create_mask_and_input(cap_enc, tokenizer)
for i in range(3):
    print(f'Batch Item {i}:')
    print(tokenizer.decode(train_cap_enc[i].numpy()))
    print(tokenizer.decode(cap_enc[i].numpy()))
    print()
```

    Batch Item 0:
    [CLS][MASK] girls share a bite of pizza clowning for the camera.[SEP]<pad><pad><pad><pad>
    [CLS] three girls share a bite of pizza clowning for the camera.[SEP]<pad><pad><pad><pad>
    
    Batch Item 1:
    [CLS] a bathroom done in almost total white[SEP]<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    [CLS] a bathroom done in almost total white[SEP]<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    
    Batch Item 2:
    [CLS] a[MASK] rider competing in an obstacle course event.[SEP]<pad><pad><pad><pad><pad><pad><pad>
    [CLS] a horseback rider competing in an obstacle course event.[SEP]<pad><pad><pad><pad><pad><pad><pad>
    
    

## Defining the Model Architecture

<img src="bmt_architecture.png" width="600">


```python
class BimodalMLMTransformer(tf.keras.Model):
    
    def __init__(self, transformer: Transformer, **kwargs):
        super().__init__(**kwargs)
        
        self.transformer = transformer
        self.predictions  = MLMHead(transformer.config, 
                                    transformer.albert.embeddings,
                                    name='predictions')
    
    def call(self, 
             inputs: tf.Tensor, 
             **transformer_kwargs):
        
        img_embs, cap_encs = inputs
        
        outputs = self.transformer(cap_encs, **transformer_kwargs)
        last_hidden_state, *_ = outputs
        batch_size, batch_seq_len, last_hidden_dim = last_hidden_state.shape

        # reshape and repeat image embeddings
        batch_size, *img_emb_shape = img_embs.shape
        img_emb_flattened = tf.reshape(img_embs, (batch_size, np.prod(img_emb_shape)))
        emb_flattened_reps = tf.repeat(tf.expand_dims(img_emb_flattened, 1), 
                                       batch_seq_len, axis=1)
        
        # concatenate the language and image embeddings
        embs_concat = tf.concat([last_hidden_state, emb_flattened_reps], 2)
        
        # generate mlm predictions over input sequence
        training = transformer_kwargs.get('training', False)
        prediction_scores = self.predictions(embs_concat, training=training)

        # Add hidden states and attention if they are here
        outputs = (prediction_scores,) + outputs[2:]

        return outputs
```


```python
transformer = Transformer.from_pretrained(PRETRAINED_TRANSFORMER_VERSION)
transformer.trainable = False
```


```python
bm_transformer = BimodalMLMTransformer(transformer)
```


```python
bm_transformer.summary()
```

    Model: "bimodal_mlm_transformer_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    tf_albert_model_1 (TFAlbertM multiple                  11683584  
    _________________________________________________________________
    predictions (TFAlbertMLMHead multiple                  5113312   
    =================================================================
    Total params: 12,890,848
    Trainable params: 1,207,264
    Non-trainable params: 11,683,584
    _________________________________________________________________
    


```python
iid, img_emb, cap_enc = next(iter(coco_train))
iid, img_emb, cap_enc
```




    (<tf.Tensor: shape=(8,), dtype=int32, numpy=array([230362, 472375, 389532, 160728,  71756, 359937, 394611, 500211])>,
     <tf.Tensor: shape=(8, 2, 2, 2048), dtype=float32, numpy=
     array([[[[8.2812655e-01, 0.0000000e+00, 2.3744605e+00, ...,
               2.6479197e+00, 6.6884130e-02, 0.0000000e+00],
              [1.9108174e+00, 7.6255983e-01, 1.2225776e+00, ...,
               2.6479197e+00, 6.6884130e-02, 0.0000000e+00]],
     
             [[3.3572485e+00, 0.0000000e+00, 5.0006227e+00, ...,
               2.6479197e+00, 6.6884130e-02, 0.0000000e+00],
              [2.5800207e+00, 6.2725401e-01, 2.6019411e+00, ...,
               2.6479197e+00, 6.6884130e-02, 0.0000000e+00]]],
     
     
            [[[2.5249224e+00, 1.4624733e+00, 2.5228009e+00, ...,
               0.0000000e+00, 0.0000000e+00, 1.4165843e-01],
              [9.5054090e-01, 6.7353225e-01, 1.2732261e+00, ...,
               0.0000000e+00, 0.0000000e+00, 1.4165843e-01]],
     
             [[1.6882330e-01, 2.8575425e+00, 2.6694603e+00, ...,
               0.0000000e+00, 0.0000000e+00, 1.4165843e-01],
              [0.0000000e+00, 0.0000000e+00, 2.5825188e+00, ...,
               0.0000000e+00, 0.0000000e+00, 1.4165843e-01]]],
     
     
            [[[1.0990660e+00, 0.0000000e+00, 3.5378442e-04, ...,
               3.6167808e+00, 8.9541662e-01, 2.8415552e-01],
              [1.3570967e-01, 4.8285586e-01, 0.0000000e+00, ...,
               3.6167808e+00, 8.9541662e-01, 2.8415552e-01]],
     
             [[4.2530566e-01, 0.0000000e+00, 0.0000000e+00, ...,
               3.6167808e+00, 8.9541662e-01, 2.8415552e-01],
              [6.9395107e-01, 0.0000000e+00, 0.0000000e+00, ...,
               3.6167808e+00, 8.9541662e-01, 2.8415552e-01]]],
     
     
            ...,
     
     
            [[[0.0000000e+00, 9.2166923e-02, 3.8562089e-01, ...,
               1.0446195e-01, 5.5479014e-01, 5.7398777e+00],
              [0.0000000e+00, 8.6664182e-01, 0.0000000e+00, ...,
               1.0446195e-01, 5.5479014e-01, 5.7398777e+00]],
     
             [[2.4859960e-01, 0.0000000e+00, 0.0000000e+00, ...,
               1.0446195e-01, 5.5479014e-01, 5.7398777e+00],
              [0.0000000e+00, 3.3620727e-01, 3.1173995e-02, ...,
               1.0446195e-01, 5.5479014e-01, 5.7398777e+00]]],
     
     
            [[[7.9029393e-01, 0.0000000e+00, 1.7498780e+00, ...,
               1.1096790e+00, 5.7162367e-02, 0.0000000e+00],
              [0.0000000e+00, 0.0000000e+00, 7.9035050e-01, ...,
               1.1096790e+00, 5.7162367e-02, 0.0000000e+00]],
     
             [[1.4421252e+00, 0.0000000e+00, 1.2098587e+00, ...,
               1.1096790e+00, 5.7162367e-02, 0.0000000e+00],
              [4.0289724e-01, 5.4949260e-01, 1.0056858e+00, ...,
               1.1096790e+00, 5.7162367e-02, 0.0000000e+00]]],
     
     
            [[[1.0952333e+00, 0.0000000e+00, 0.0000000e+00, ...,
               2.0560949e+00, 1.0755324e+00, 0.0000000e+00],
              [9.7229004e-01, 0.0000000e+00, 0.0000000e+00, ...,
               2.0560949e+00, 1.0755324e+00, 0.0000000e+00]],
     
             [[1.0152134e+00, 0.0000000e+00, 1.9725811e-01, ...,
               2.0560949e+00, 1.0755324e+00, 0.0000000e+00],
              [8.8078249e-01, 0.0000000e+00, 0.0000000e+00, ...,
               2.0560949e+00, 1.0755324e+00, 0.0000000e+00]]]], dtype=float32)>,
     <tf.Tensor: shape=(8, 17), dtype=int32, numpy=
     array([[    2,   238,  6913,  7443,    18,    50,  6120,    69,    35,
                21,   727,   131,  4005,    93, 18586,     9,     3],
            [    2,    21,  1952,    19,    21,  8835,  1805,   328,    20,
                21,  7021,     9,     3,     0,     0,     0,     0],
            [    2,    14,   169,    25,    27,    14,  1407,    20,   247,
                21,  2151,     9,     3,     0,     0,     0,     0],
            [    2,    21,  1335,    16,   148,  1017,  1451,    20,   162,
              2252,    68,     9,     3,     0,     0,     0,     0],
            [    2,    81,  5456,  2094,   429,    19,    14,   343,     9,
                 3,     0,     0,     0,     0,     0,     0,     0],
            [    2,    40,  2987,  1494,    13, 20629,  1683,  8951,    19,
                21,  4271,   865,     3,     0,     0,     0,     0],
            [    2, 10698, 16819,    18,    50, 18158,    19,    21,   575,
                16,  3961,     9,     3,     0,     0,     0,     0],
            [    2,    65, 13447,    18,    50,   827,    35,    24,  7484,
               719,    14,  1454,     3,     0,     0,     0,     0]])>)




```python
preds, *_ = bm_transformer((img_emb, cap_enc))
preds
```




    <tf.Tensor: shape=(8, 17, 30000), dtype=float32, numpy=
    array([[[-0.00255676, -0.28133634,  0.12696652, ...,  0.1115055 ,
              0.34350696,  0.3839145 ],
            [-0.09330822, -0.18555197,  0.01063907, ...,  0.03937093,
              0.41564718,  0.08644014],
            [-0.03865747, -0.10990265,  0.03537906, ...,  0.11738862,
              0.42082825,  0.17752409],
            ...,
            [-0.17437483, -0.23764138,  0.02301986, ...,  0.17670228,
              0.44555935,  0.24811713],
            [-0.06642682, -0.20172231,  0.01112237, ...,  0.07609118,
              0.36903346,  0.18448952],
            [-0.06554484, -0.18087997,  0.11517162, ...,  0.12602535,
              0.29741278,  0.15244102]],
    
           [[-0.48096776,  0.15347643,  0.08103826, ..., -0.18020293,
             -0.01668783,  0.085361  ],
            [-0.48062724,  0.15324363,  0.08098698, ..., -0.18029806,
             -0.01708161,  0.08533505],
            [-0.48065975,  0.1532947 ,  0.08097532, ..., -0.18026003,
             -0.01703009,  0.08532354],
            ...,
            [-0.4807474 ,  0.15330468,  0.08097569, ..., -0.1802503 ,
             -0.01706248,  0.08531008],
            [-0.480759  ,  0.15328895,  0.0809788 , ..., -0.18023495,
             -0.01707341,  0.08531431],
            [-0.48075563,  0.15328765,  0.08098627, ..., -0.18023524,
             -0.01707097,  0.08532181]],
    
           [[ 0.3369307 ,  0.1441591 ,  0.26657283, ...,  0.0022272 ,
              0.04089213,  0.35672793],
            [ 0.3369843 ,  0.14400263,  0.26646456, ...,  0.00215032,
              0.04073099,  0.35656595],
            [ 0.33697578,  0.14403068,  0.266475  , ...,  0.0021708 ,
              0.04077001,  0.35658827],
            ...,
            [ 0.33694553,  0.14403628,  0.26646152, ...,  0.00216704,
              0.04072389,  0.35657942],
            [ 0.33693755,  0.14403135,  0.2664589 , ...,  0.00217124,
              0.04070442,  0.35657936],
            [ 0.33694082,  0.1440314 ,  0.2664628 , ...,  0.00217356,
              0.04069646,  0.35658038]],
    
           ...,
    
           [[ 0.11474691,  0.43506846, -0.0488399 , ...,  0.08524048,
             -0.13549633, -0.03852874],
            [ 0.11484137,  0.43488133, -0.04889246, ...,  0.08528021,
             -0.13553528, -0.03858124],
            [ 0.1148397 ,  0.43490976, -0.04889422, ...,  0.08528391,
             -0.1355264 , -0.03857749],
            ...,
            [ 0.11481762,  0.43490323, -0.04890464, ...,  0.08527907,
             -0.13554427, -0.03859958],
            [ 0.11481231,  0.43490583, -0.04890385, ...,  0.08528252,
             -0.13554735, -0.03859513],
            [ 0.11481094,  0.4349066 , -0.04890129, ...,  0.08528319,
             -0.13554789, -0.03859063]],
    
           [[ 0.10874455, -0.6045106 ,  0.21369436, ..., -0.33030063,
             -0.2801535 ,  0.4735406 ],
            [ 0.10886773, -0.60490644,  0.21340694, ..., -0.3304925 ,
             -0.28036273,  0.4734494 ],
            [ 0.10889418, -0.60499537,  0.2133393 , ..., -0.3305158 ,
             -0.28033692,  0.4734438 ],
            ...,
            [ 0.10881165, -0.6049438 ,  0.2133807 , ..., -0.33045214,
             -0.2803491 ,  0.473436  ],
            [ 0.10878959, -0.6049538 ,  0.21338415, ..., -0.3304423 ,
             -0.28037205,  0.47344586],
            [ 0.10878292, -0.60495055,  0.21339329, ..., -0.33044457,
             -0.28040516,  0.4734507 ]],
    
           [[-0.12982284, -0.16419652, -0.07131735, ...,  0.43817106,
             -0.20346512,  0.38813096],
            [-0.12939592, -0.16425359, -0.07142249, ...,  0.43794665,
             -0.20368293,  0.38791475],
            [-0.1294634 , -0.16427207, -0.07144201, ...,  0.4379724 ,
             -0.20362571,  0.38792297],
            ...,
            [-0.12954767, -0.1643006 , -0.07145838, ...,  0.4379321 ,
             -0.20366599,  0.38793808],
            [-0.12957484, -0.16429578, -0.07144624, ...,  0.43794423,
             -0.20369111,  0.3879584 ],
            [-0.12957938, -0.16427921, -0.07143024, ...,  0.4379422 ,
             -0.2037229 ,  0.3879707 ]]], dtype=float32)>




```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1.0)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_acc = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
train_loss = tf.keras.metrics.Mean(name='train_loss')
```


```python
checkpoint_path = f'{OUTPUTS_DIR}/ckpts'

ckpt = tf.train.Checkpoint(transformer=bm_transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
```

## Training the Model

For lack of resources, we will not train the model to convergence. 

<img src="loss.PNG" width="600">


```python
def train_step(model: tf.keras.Model,
               optimizer: tf.keras.optimizers.Optimizer,
               tokenizer: Tokenizer,
               inp: Tuple):
    
    iids, img_embs, cap_encs = inp
    masked_cap_encs, where_masked = create_mask_and_input(cap_encs, tokenizer)
    
    with tf.GradientTape() as tape:
        attention_mask = tf.cast(masked_cap_encs != tokenizer.pad_token_id, 
                                 tf.int32)
        logits, *_ = model((img_embs, masked_cap_encs), 
                           attention_mask=attention_mask,
                           training=True)
        loss = loss_fn(cap_encs[where_masked], logits[where_masked])

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_acc(cap_encs[where_masked], logits[where_masked])
    train_loss(loss)
    
    return logits, loss


# train_step_signature = [
#     tf.TensorSpec(shape=(BATCH_SIZE, MAX_SEQ_SIZE), 
#                   dtype=tf.int32),
#     tf.TensorSpec(shape=(BATCH_SIZE, MAX_SEQ_SIZE), 
#                   dtype=tf.int32),
# ]


@tf.function
def train_step_tf(inp):
    train_step(bm_transformer, optimizer, tokenizer, inp)
```


```python
BATCHES_IN_EPOCH = 250
epoch = 0
```


```python
EPOCHS = 300
```


```python
try:
    while epoch < EPOCHS:
        start = time.time()

        train_loss.reset_states()
        train_acc.reset_states()

        for batch, inp in enumerate(coco_train):
            train_step_tf(inp)

            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_acc.result()))

                with summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), 
                                      step=epoch)
                    tf.summary.scalar('accuracy', train_acc.result(), 
                                      step=epoch)

            if batch >= BATCHES_IN_EPOCH:
                break

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                             train_loss.result(), 
                                                             train_acc.result()))

        print ('Time taken for epoch: {} secs\n'.format(time.time() - start))
        epoch += 1
        
except KeyboardInterrupt:
    print('Manual interrupt')
```

    WARNING:tensorflow:5 out of the last 5 calls to <function train_step_tf at 0x00000184665671E0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.
    WARNING:tensorflow:5 out of the last 14 calls to <function train_step_tf at 0x00000184665671E0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.
    Epoch 1 Loss 8.5700 Accuracy 0.1241
    Time taken for epoch: 212.10938668251038 secs
    
    Epoch 2 Loss 8.5289 Accuracy 0.1365
    Time taken for epoch: 32.14652180671692 secs
    
    Epoch 3 Loss 8.4865 Accuracy 0.1351
    Time taken for epoch: 31.461716651916504 secs
    
    Epoch 4 Loss 8.4061 Accuracy 0.1461
    Time taken for epoch: 31.625213623046875 secs
    
    Saving checkpoint for epoch 5 at ./outputs/20200424-000059/ckpts\ckpt-1
    Epoch 5 Loss 8.4077 Accuracy 0.1427
    Time taken for epoch: 45.481651306152344 secs
    
    Epoch 6 Loss 8.3722 Accuracy 0.1521
    Time taken for epoch: 35.16904425621033 secs
    
    Epoch 7 Loss 8.3403 Accuracy 0.1516
    Time taken for epoch: 32.767136335372925 secs
    
    Epoch 8 Loss 8.2875 Accuracy 0.1562
    Time taken for epoch: 32.058943033218384 secs
    
    Epoch 9 Loss 8.2958 Accuracy 0.1460
    Time taken for epoch: 32.35899996757507 secs
    
    Saving checkpoint for epoch 10 at ./outputs/20200424-000059/ckpts\ckpt-2
    Epoch 10 Loss 8.2949 Accuracy 0.1455
    Time taken for epoch: 33.14226007461548 secs
    
    Epoch 11 Loss 8.2278 Accuracy 0.1529
    Time taken for epoch: 36.743035078048706 secs
    
    Epoch 12 Loss 8.2331 Accuracy 0.1496
    Time taken for epoch: 33.16008901596069 secs
    
    Epoch 13 Loss 8.2192 Accuracy 0.1462
    Time taken for epoch: 31.892055988311768 secs
    
    Epoch 14 Loss 8.1839 Accuracy 0.1634
    Time taken for epoch: 33.14100193977356 secs
    
    Saving checkpoint for epoch 15 at ./outputs/20200424-000059/ckpts\ckpt-3
    Epoch 15 Loss 8.1425 Accuracy 0.1648
    Time taken for epoch: 33.36868143081665 secs
    
    Epoch 16 Loss 8.1432 Accuracy 0.1669
    Time taken for epoch: 32.37559223175049 secs
    
    Epoch 17 Loss 8.1604 Accuracy 0.1540
    Time taken for epoch: 32.93604588508606 secs
    
    Epoch 18 Loss 8.1091 Accuracy 0.1681
    Time taken for epoch: 32.19421696662903 secs
    
    Epoch 19 Loss 8.1024 Accuracy 0.1603
    Time taken for epoch: 32.191648721694946 secs
    
    Saving checkpoint for epoch 20 at ./outputs/20200424-000059/ckpts\ckpt-4
    Epoch 20 Loss 8.0905 Accuracy 0.1569
    Time taken for epoch: 32.74822497367859 secs
    
    Epoch 21 Loss 8.0538 Accuracy 0.1656
    Time taken for epoch: 33.30108594894409 secs
    
    Epoch 22 Loss 8.0506 Accuracy 0.1661
    Time taken for epoch: 32.749001026153564 secs
    
    Epoch 23 Loss 8.0177 Accuracy 0.1742
    Time taken for epoch: 32.90497159957886 secs
    
    Epoch 24 Loss 7.9912 Accuracy 0.1641
    Time taken for epoch: 32.182924032211304 secs
    
    Saving checkpoint for epoch 25 at ./outputs/20200424-000059/ckpts\ckpt-5
    Epoch 25 Loss 7.9911 Accuracy 0.1658
    Time taken for epoch: 33.31268310546875 secs
    
    Epoch 26 Loss 7.9825 Accuracy 0.1730
    Time taken for epoch: 32.88956332206726 secs
    
    Epoch 27 Loss 7.9386 Accuracy 0.1697
    Time taken for epoch: 34.573052406311035 secs
    
    Epoch 28 Loss 7.9192 Accuracy 0.1617
    Time taken for epoch: 32.02155542373657 secs
    
    Epoch 29 Loss 7.9343 Accuracy 0.1648
    Time taken for epoch: 31.97933578491211 secs
    
    Saving checkpoint for epoch 30 at ./outputs/20200424-000059/ckpts\ckpt-6
    Epoch 30 Loss 7.9217 Accuracy 0.1709
    Time taken for epoch: 33.65913105010986 secs
    
    Epoch 31 Loss 7.9072 Accuracy 0.1607
    Time taken for epoch: 34.309035778045654 secs
    
    Epoch 32 Loss 7.8848 Accuracy 0.1648
    Time taken for epoch: 34.44967031478882 secs
    
    Epoch 33 Loss 7.8494 Accuracy 0.1700
    Time taken for epoch: 32.957531452178955 secs
    
    Epoch 34 Loss 7.8096 Accuracy 0.1777
    Time taken for epoch: 34.66273093223572 secs
    
    Saving checkpoint for epoch 35 at ./outputs/20200424-000059/ckpts\ckpt-7
    Epoch 35 Loss 7.8573 Accuracy 0.1681
    Time taken for epoch: 34.2369430065155 secs
    
    Epoch 36 Loss 7.8250 Accuracy 0.1736
    Time taken for epoch: 32.97799587249756 secs
    
    Epoch 37 Loss 7.8189 Accuracy 0.1706
    Time taken for epoch: 32.79442739486694 secs
    
    Epoch 38 Loss 7.7623 Accuracy 0.1792
    Time taken for epoch: 31.407426595687866 secs
    
    Epoch 39 Loss 7.7417 Accuracy 0.1769
    Time taken for epoch: 32.07775044441223 secs
    
    Saving checkpoint for epoch 40 at ./outputs/20200424-000059/ckpts\ckpt-8
    Epoch 40 Loss 7.7358 Accuracy 0.1762
    Time taken for epoch: 32.241352796554565 secs
    
    Epoch 41 Loss 7.7385 Accuracy 0.1742
    Time taken for epoch: 32.191771507263184 secs
    
    Epoch 42 Loss 7.7238 Accuracy 0.1811
    Time taken for epoch: 32.3389778137207 secs
    
    Epoch 43 Loss 7.6836 Accuracy 0.1822
    Time taken for epoch: 31.849573850631714 secs
    
    Epoch 44 Loss 7.7032 Accuracy 0.1883
    Time taken for epoch: 31.35037922859192 secs
    
    Saving checkpoint for epoch 45 at ./outputs/20200424-000059/ckpts\ckpt-9
    Epoch 45 Loss 7.6463 Accuracy 0.1795
    Time taken for epoch: 32.21041989326477 secs
    
    Epoch 46 Loss 7.6770 Accuracy 0.1727
    Time taken for epoch: 32.37504720687866 secs
    
    Epoch 47 Loss 7.6422 Accuracy 0.1822
    Time taken for epoch: 32.046173334121704 secs
    
    Epoch 48 Loss 7.6739 Accuracy 0.1741
    Time taken for epoch: 31.732130527496338 secs
    
    Epoch 49 Loss 7.6078 Accuracy 0.1771
    Time taken for epoch: 31.73463797569275 secs
    
    Saving checkpoint for epoch 50 at ./outputs/20200424-000059/ckpts\ckpt-10
    Epoch 50 Loss 7.6045 Accuracy 0.1814
    Time taken for epoch: 32.54723358154297 secs
    
    Epoch 51 Loss 7.5395 Accuracy 0.1784
    Time taken for epoch: 33.120025396347046 secs
    
    Epoch 52 Loss 7.5560 Accuracy 0.1942
    Time taken for epoch: 33.869300842285156 secs
    
    Epoch 53 Loss 7.5621 Accuracy 0.1846
    Time taken for epoch: 32.78724646568298 secs
    
    Epoch 54 Loss 7.5164 Accuracy 0.1957
    Time taken for epoch: 32.754119634628296 secs
    
    Saving checkpoint for epoch 55 at ./outputs/20200424-000059/ckpts\ckpt-11
    Epoch 55 Loss 7.5321 Accuracy 0.1855
    Time taken for epoch: 34.364585399627686 secs
    
    Epoch 56 Loss 7.4824 Accuracy 0.1925
    Time taken for epoch: 50.343934535980225 secs
    
    Epoch 57 Loss 7.4791 Accuracy 0.1960
    Time taken for epoch: 38.734633922576904 secs
    
    Epoch 58 Loss 7.4629 Accuracy 0.1958
    Time taken for epoch: 31.541430950164795 secs
    
    Epoch 59 Loss 7.4564 Accuracy 0.1915
    Time taken for epoch: 46.09680962562561 secs
    
    Saving checkpoint for epoch 60 at ./outputs/20200424-000059/ckpts\ckpt-12
    Epoch 60 Loss 7.4171 Accuracy 0.1815
    Time taken for epoch: 32.78494930267334 secs
    
    Epoch 61 Loss 7.4545 Accuracy 0.1801
    Time taken for epoch: 33.62576079368591 secs
    
    Epoch 62 Loss 7.4147 Accuracy 0.1895
    Time taken for epoch: 32.35474634170532 secs
    
    Epoch 63 Loss 7.3497 Accuracy 0.1987
    Time taken for epoch: 31.210421323776245 secs
    
    Epoch 64 Loss 7.3550 Accuracy 0.1912
    Time taken for epoch: 31.335460662841797 secs
    
    Saving checkpoint for epoch 65 at ./outputs/20200424-000059/ckpts\ckpt-13
    Epoch 65 Loss 7.3569 Accuracy 0.1897
    Time taken for epoch: 32.94380211830139 secs
    
    Epoch 66 Loss 7.4173 Accuracy 0.1879
    Time taken for epoch: 31.802561044692993 secs
    
    Epoch 67 Loss 7.3570 Accuracy 0.1946
    Time taken for epoch: 31.86418342590332 secs
    
    Epoch 68 Loss 7.2703 Accuracy 0.1983
    Time taken for epoch: 31.40635895729065 secs
    
    Epoch 69 Loss 7.3286 Accuracy 0.1839
    Time taken for epoch: 31.013038635253906 secs
    
    Saving checkpoint for epoch 70 at ./outputs/20200424-000059/ckpts\ckpt-14
    Epoch 70 Loss 7.3184 Accuracy 0.1891
    Time taken for epoch: 32.54997110366821 secs
    
    Epoch 71 Loss 7.2440 Accuracy 0.2052
    Time taken for epoch: 32.168264389038086 secs
    
    Epoch 72 Loss 7.2776 Accuracy 0.1887
    Time taken for epoch: 31.91339421272278 secs
    
    Epoch 73 Loss 7.2838 Accuracy 0.1914
    Time taken for epoch: 30.959356784820557 secs
    
    Epoch 74 Loss 7.2986 Accuracy 0.1827
    Time taken for epoch: 31.433276891708374 secs
    
    Saving checkpoint for epoch 75 at ./outputs/20200424-000059/ckpts\ckpt-15
    Epoch 75 Loss 7.2501 Accuracy 0.1868
    Time taken for epoch: 32.311838150024414 secs
    
    Epoch 76 Loss 7.2328 Accuracy 0.1850
    Time taken for epoch: 32.36157846450806 secs
    
    Epoch 77 Loss 7.2148 Accuracy 0.2022
    Time taken for epoch: 32.239744663238525 secs
    
    Epoch 78 Loss 7.2250 Accuracy 0.1862
    Time taken for epoch: 31.68625807762146 secs
    
    Epoch 79 Loss 7.1974 Accuracy 0.1954
    Time taken for epoch: 31.45208168029785 secs
    
    Saving checkpoint for epoch 80 at ./outputs/20200424-000059/ckpts\ckpt-16
    Epoch 80 Loss 7.1588 Accuracy 0.1935
    Time taken for epoch: 32.47214603424072 secs
    
    Epoch 81 Loss 7.1429 Accuracy 0.1988
    Time taken for epoch: 32.57346820831299 secs
    
    Epoch 82 Loss 7.1840 Accuracy 0.1977
    Time taken for epoch: 32.19447302818298 secs
    
    Epoch 83 Loss 7.2175 Accuracy 0.1782
    Time taken for epoch: 31.672118186950684 secs
    
    Epoch 84 Loss 7.1474 Accuracy 0.1965
    Time taken for epoch: 31.61362600326538 secs
    
    Saving checkpoint for epoch 85 at ./outputs/20200424-000059/ckpts\ckpt-17
    Epoch 85 Loss 7.1722 Accuracy 0.1860
    Time taken for epoch: 32.345420598983765 secs
    
    Epoch 86 Loss 7.1148 Accuracy 0.1952
    Time taken for epoch: 32.344053983688354 secs
    
    Epoch 87 Loss 7.1947 Accuracy 0.1802
    Time taken for epoch: 32.88852143287659 secs
    
    Epoch 88 Loss 7.1099 Accuracy 0.2002
    Time taken for epoch: 31.86199688911438 secs
    
    Epoch 89 Loss 7.0477 Accuracy 0.2041
    Time taken for epoch: 30.808241605758667 secs
    
    Saving checkpoint for epoch 90 at ./outputs/20200424-000059/ckpts\ckpt-18
    Epoch 90 Loss 7.0374 Accuracy 0.1963
    Time taken for epoch: 32.30452060699463 secs
    
    Epoch 91 Loss 7.0211 Accuracy 0.2115
    Time taken for epoch: 31.954118490219116 secs
    
    Epoch 92 Loss 7.0618 Accuracy 0.1972
    Time taken for epoch: 31.96932578086853 secs
    
    Epoch 93 Loss 7.0137 Accuracy 0.2072
    Time taken for epoch: 30.37588620185852 secs
    
    Epoch 94 Loss 6.9425 Accuracy 0.2058
    Time taken for epoch: 31.73863172531128 secs
    
    Saving checkpoint for epoch 95 at ./outputs/20200424-000059/ckpts\ckpt-19
    Epoch 95 Loss 7.0167 Accuracy 0.1995
    Time taken for epoch: 32.66121816635132 secs
    
    Epoch 96 Loss 6.9587 Accuracy 0.2084
    Time taken for epoch: 31.995023488998413 secs
    
    Epoch 97 Loss 6.9821 Accuracy 0.1993
    Time taken for epoch: 31.98035979270935 secs
    
    Epoch 98 Loss 6.9490 Accuracy 0.2123
    Time taken for epoch: 31.142148733139038 secs
    
    Epoch 99 Loss 6.9409 Accuracy 0.2034
    Time taken for epoch: 30.83591103553772 secs
    
    Saving checkpoint for epoch 100 at ./outputs/20200424-000059/ckpts\ckpt-20
    Epoch 100 Loss 6.8738 Accuracy 0.2102
    Time taken for epoch: 31.925139665603638 secs
    
    Epoch 101 Loss 6.9783 Accuracy 0.1851
    Time taken for epoch: 32.123523235321045 secs
    
    Epoch 102 Loss 6.9226 Accuracy 0.2001
    Time taken for epoch: 32.61361622810364 secs
    
    Epoch 103 Loss 6.9494 Accuracy 0.1951
    Time taken for epoch: 31.51773500442505 secs
    
    Epoch 104 Loss 6.8642 Accuracy 0.2115
    Time taken for epoch: 31.930673599243164 secs
    
    Saving checkpoint for epoch 105 at ./outputs/20200424-000059/ckpts\ckpt-21
    Epoch 105 Loss 6.8924 Accuracy 0.1964
    Time taken for epoch: 32.19688034057617 secs
    
    Epoch 106 Loss 6.9015 Accuracy 0.1934
    Time taken for epoch: 32.31608605384827 secs
    
    Epoch 107 Loss 6.8894 Accuracy 0.1932
    Time taken for epoch: 33.12481236457825 secs
    
    Epoch 108 Loss 6.8762 Accuracy 0.1981
    Time taken for epoch: 31.48196268081665 secs
    
    Epoch 109 Loss 6.8606 Accuracy 0.2026
    Time taken for epoch: 30.79866600036621 secs
    
    Saving checkpoint for epoch 110 at ./outputs/20200424-000059/ckpts\ckpt-22
    Epoch 110 Loss 6.8463 Accuracy 0.1931
    Time taken for epoch: 32.12157487869263 secs
    
    Epoch 111 Loss 6.8152 Accuracy 0.2020
    Time taken for epoch: 32.97261571884155 secs
    
    Epoch 112 Loss 6.8183 Accuracy 0.2058
    Time taken for epoch: 32.45700001716614 secs
    
    Epoch 113 Loss 6.7767 Accuracy 0.2131
    Time taken for epoch: 31.51699709892273 secs
    
    Epoch 114 Loss 6.7902 Accuracy 0.2028
    Time taken for epoch: 31.315000772476196 secs
    
    Saving checkpoint for epoch 115 at ./outputs/20200424-000059/ckpts\ckpt-23
    Epoch 115 Loss 6.7667 Accuracy 0.2129
    Time taken for epoch: 32.45401167869568 secs
    
    Epoch 116 Loss 6.7645 Accuracy 0.1953
    Time taken for epoch: 32.16106057167053 secs
    
    Epoch 117 Loss 6.7699 Accuracy 0.2091
    Time taken for epoch: 32.616071462631226 secs
    
    Epoch 118 Loss 6.7214 Accuracy 0.2031
    Time taken for epoch: 31.721686601638794 secs
    
    Epoch 119 Loss 6.7219 Accuracy 0.2047
    Time taken for epoch: 34.52328824996948 secs
    
    Saving checkpoint for epoch 120 at ./outputs/20200424-000059/ckpts\ckpt-24
    Epoch 120 Loss 6.6857 Accuracy 0.2065
    Time taken for epoch: 32.91475439071655 secs
    
    Epoch 121 Loss 6.7332 Accuracy 0.2062
    Time taken for epoch: 32.52499842643738 secs
    
    Epoch 122 Loss 6.7020 Accuracy 0.1965
    Time taken for epoch: 31.756636381149292 secs
    
    Epoch 123 Loss 6.6565 Accuracy 0.2032
    Time taken for epoch: 31.728952884674072 secs
    
    Epoch 124 Loss 6.7094 Accuracy 0.2032
    Time taken for epoch: 31.94950032234192 secs
    
    Saving checkpoint for epoch 125 at ./outputs/20200424-000059/ckpts\ckpt-25
    Epoch 125 Loss 6.6704 Accuracy 0.2053
    Time taken for epoch: 32.71711778640747 secs
    
    Epoch 126 Loss 6.6699 Accuracy 0.2045
    Time taken for epoch: 31.81719422340393 secs
    
    Epoch 127 Loss 6.6774 Accuracy 0.2008
    Time taken for epoch: 32.533124923706055 secs
    
    Epoch 128 Loss 6.5764 Accuracy 0.2194
    Time taken for epoch: 31.083464860916138 secs
    
    Epoch 129 Loss 6.6060 Accuracy 0.2040
    Time taken for epoch: 32.271514892578125 secs
    
    Saving checkpoint for epoch 130 at ./outputs/20200424-000059/ckpts\ckpt-26
    Epoch 130 Loss 6.6295 Accuracy 0.2055
    Time taken for epoch: 32.72403931617737 secs
    
    Epoch 131 Loss 6.5526 Accuracy 0.2138
    Time taken for epoch: 32.26662826538086 secs
    
    Epoch 132 Loss 6.5830 Accuracy 0.2126
    Time taken for epoch: 32.874056816101074 secs
    
    Epoch 133 Loss 6.5776 Accuracy 0.2028
    Time taken for epoch: 33.24320316314697 secs
    
    Epoch 134 Loss 6.5611 Accuracy 0.2182
    Time taken for epoch: 32.718533515930176 secs
    
    Saving checkpoint for epoch 135 at ./outputs/20200424-000059/ckpts\ckpt-27
    Epoch 135 Loss 6.4800 Accuracy 0.2229
    Time taken for epoch: 33.004881381988525 secs
    
    Epoch 136 Loss 6.5098 Accuracy 0.2084
    Time taken for epoch: 33.50303912162781 secs
    
    Epoch 137 Loss 6.5525 Accuracy 0.2023
    Time taken for epoch: 34.43604230880737 secs
    
    Epoch 138 Loss 6.5753 Accuracy 0.1989
    Time taken for epoch: 34.08204007148743 secs
    
    Epoch 139 Loss 6.5419 Accuracy 0.2035
    Time taken for epoch: 32.50700283050537 secs
    
    Saving checkpoint for epoch 140 at ./outputs/20200424-000059/ckpts\ckpt-28
    Epoch 140 Loss 6.4724 Accuracy 0.2137
    Time taken for epoch: 33.4803192615509 secs
    
    Epoch 141 Loss 6.4688 Accuracy 0.2138
    Time taken for epoch: 32.21754193305969 secs
    
    Epoch 142 Loss 6.5129 Accuracy 0.2023
    Time taken for epoch: 33.239044427871704 secs
    
    Epoch 143 Loss 6.4453 Accuracy 0.2037
    Time taken for epoch: 31.961047649383545 secs
    
    Epoch 144 Loss 6.4791 Accuracy 0.2027
    Time taken for epoch: 32.27643179893494 secs
    
    Saving checkpoint for epoch 145 at ./outputs/20200424-000059/ckpts\ckpt-29
    Epoch 145 Loss 6.4800 Accuracy 0.2065
    Time taken for epoch: 33.85904669761658 secs
    
    Epoch 146 Loss 6.4300 Accuracy 0.1996
    Time taken for epoch: 33.26154112815857 secs
    
    Epoch 147 Loss 6.4503 Accuracy 0.2055
    Time taken for epoch: 33.00368022918701 secs
    
    Epoch 148 Loss 6.3686 Accuracy 0.2194
    Time taken for epoch: 31.74306845664978 secs
    
    Epoch 149 Loss 6.4002 Accuracy 0.2047
    Time taken for epoch: 31.76083517074585 secs
    
    Saving checkpoint for epoch 150 at ./outputs/20200424-000059/ckpts\ckpt-30
    Epoch 150 Loss 6.3495 Accuracy 0.2213
    Time taken for epoch: 33.106603384017944 secs
    
    Epoch 151 Loss 6.4186 Accuracy 0.2132
    Time taken for epoch: 33.09752058982849 secs
    
    Epoch 152 Loss 6.3561 Accuracy 0.2166
    Time taken for epoch: 33.46908903121948 secs
    
    Epoch 153 Loss 6.3759 Accuracy 0.2068
    Time taken for epoch: 31.840595245361328 secs
    
    Epoch 154 Loss 6.4010 Accuracy 0.2042
    Time taken for epoch: 32.46289777755737 secs
    
    Saving checkpoint for epoch 155 at ./outputs/20200424-000059/ckpts\ckpt-31
    Epoch 155 Loss 6.3562 Accuracy 0.2088
    Time taken for epoch: 33.20332145690918 secs
    
    Epoch 156 Loss 6.3271 Accuracy 0.2141
    Time taken for epoch: 32.81657028198242 secs
    
    Epoch 157 Loss 6.3416 Accuracy 0.2128
    Time taken for epoch: 33.37504196166992 secs
    
    Epoch 158 Loss 6.3514 Accuracy 0.2140
    Time taken for epoch: 31.780336618423462 secs
    
    Epoch 159 Loss 6.2401 Accuracy 0.2142
    Time taken for epoch: 32.418598890304565 secs
    
    Saving checkpoint for epoch 160 at ./outputs/20200424-000059/ckpts\ckpt-32
    Epoch 160 Loss 6.2680 Accuracy 0.2185
    Time taken for epoch: 32.82458972930908 secs
    
    Epoch 161 Loss 6.3015 Accuracy 0.2052
    Time taken for epoch: 33.63652992248535 secs
    
    Epoch 162 Loss 6.2244 Accuracy 0.2225
    Time taken for epoch: 32.90452194213867 secs
    
    Epoch 163 Loss 6.2291 Accuracy 0.2161
    Time taken for epoch: 33.23999762535095 secs
    
    Epoch 164 Loss 6.2191 Accuracy 0.2165
    Time taken for epoch: 31.4727885723114 secs
    
    Saving checkpoint for epoch 165 at ./outputs/20200424-000059/ckpts\ckpt-33
    Epoch 165 Loss 6.2288 Accuracy 0.2138
    Time taken for epoch: 33.100038290023804 secs
    
    Epoch 166 Loss 6.2358 Accuracy 0.2108
    Time taken for epoch: 32.964574337005615 secs
    
    Epoch 167 Loss 6.2230 Accuracy 0.2047
    Time taken for epoch: 33.70839977264404 secs
    
    Epoch 168 Loss 6.2491 Accuracy 0.2181
    Time taken for epoch: 32.17646598815918 secs
    
    Epoch 169 Loss 6.2090 Accuracy 0.2134
    Time taken for epoch: 31.735527515411377 secs
    
    Saving checkpoint for epoch 170 at ./outputs/20200424-000059/ckpts\ckpt-34
    Epoch 170 Loss 6.2238 Accuracy 0.2061
    Time taken for epoch: 33.63493323326111 secs
    
    Epoch 171 Loss 6.1645 Accuracy 0.2178
    Time taken for epoch: 33.208568811416626 secs
    
    Epoch 172 Loss 6.1769 Accuracy 0.2133
    Time taken for epoch: 33.21354818344116 secs
    
    Epoch 173 Loss 6.1050 Accuracy 0.2249
    Time taken for epoch: 31.823221683502197 secs
    
    Epoch 174 Loss 6.2179 Accuracy 0.2150
    Time taken for epoch: 32.192790269851685 secs
    
    Saving checkpoint for epoch 175 at ./outputs/20200424-000059/ckpts\ckpt-35
    Epoch 175 Loss 6.2101 Accuracy 0.2090
    Time taken for epoch: 33.59318518638611 secs
    
    Epoch 176 Loss 6.0808 Accuracy 0.2253
    Time taken for epoch: 32.49952030181885 secs
    
    Epoch 177 Loss 6.2087 Accuracy 0.2046
    Time taken for epoch: 33.47866773605347 secs
    
    Epoch 178 Loss 6.0613 Accuracy 0.2287
    Time taken for epoch: 32.70856595039368 secs
    
    Epoch 179 Loss 6.1333 Accuracy 0.2127
    Time taken for epoch: 35.68169045448303 secs
    
    Saving checkpoint for epoch 180 at ./outputs/20200424-000059/ckpts\ckpt-36
    Epoch 180 Loss 6.1376 Accuracy 0.2134
    Time taken for epoch: 32.8343825340271 secs
    
    Epoch 181 Loss 6.0871 Accuracy 0.2171
    Time taken for epoch: 32.829524517059326 secs
    
    Epoch 182 Loss 6.0883 Accuracy 0.2094
    Time taken for epoch: 32.82399344444275 secs
    
    Epoch 183 Loss 6.0359 Accuracy 0.2189
    Time taken for epoch: 31.574506282806396 secs
    
    Epoch 184 Loss 6.1455 Accuracy 0.2121
    Time taken for epoch: 32.426947832107544 secs
    
    Saving checkpoint for epoch 185 at ./outputs/20200424-000059/ckpts\ckpt-37
    Epoch 185 Loss 6.0757 Accuracy 0.2184
    Time taken for epoch: 33.064918994903564 secs
    
    Epoch 186 Loss 6.1190 Accuracy 0.2158
    Time taken for epoch: 32.906089067459106 secs
    
    Epoch 187 Loss 6.0488 Accuracy 0.2247
    Time taken for epoch: 33.43388772010803 secs
    
    Epoch 188 Loss 6.0730 Accuracy 0.2034
    Time taken for epoch: 31.79444169998169 secs
    
    Epoch 189 Loss 6.0124 Accuracy 0.2122
    Time taken for epoch: 31.17106866836548 secs
    
    Saving checkpoint for epoch 190 at ./outputs/20200424-000059/ckpts\ckpt-38
    Epoch 190 Loss 6.0409 Accuracy 0.2204
    Time taken for epoch: 33.21911144256592 secs
    
    Epoch 191 Loss 5.9412 Accuracy 0.2269
    Time taken for epoch: 33.042030811309814 secs
    
    Epoch 192 Loss 6.0198 Accuracy 0.2104
    Time taken for epoch: 33.275026082992554 secs
    
    Epoch 193 Loss 5.9817 Accuracy 0.2203
    Time taken for epoch: 32.16033577919006 secs
    
    Epoch 194 Loss 6.0571 Accuracy 0.2093
    Time taken for epoch: 32.024157762527466 secs
    
    Saving checkpoint for epoch 195 at ./outputs/20200424-000059/ckpts\ckpt-39
    Epoch 195 Loss 5.9508 Accuracy 0.2248
    Time taken for epoch: 34.382545471191406 secs
    
    Epoch 196 Loss 5.9149 Accuracy 0.2281
    Time taken for epoch: 32.91306233406067 secs
    
    Epoch 197 Loss 5.9724 Accuracy 0.2147
    Time taken for epoch: 33.344929933547974 secs
    
    Epoch 198 Loss 5.9856 Accuracy 0.2112
    Time taken for epoch: 31.509769439697266 secs
    
    Epoch 199 Loss 5.9136 Accuracy 0.2261
    Time taken for epoch: 31.990769386291504 secs
    
    Saving checkpoint for epoch 200 at ./outputs/20200424-000059/ckpts\ckpt-40
    Epoch 200 Loss 6.0041 Accuracy 0.2147
    Time taken for epoch: 33.52601170539856 secs
    
    Epoch 201 Loss 5.8815 Accuracy 0.2226
    Time taken for epoch: 33.59899973869324 secs
    
    Epoch 202 Loss 5.8567 Accuracy 0.2236
    Time taken for epoch: 33.056419134140015 secs
    
    Epoch 203 Loss 5.9175 Accuracy 0.2184
    Time taken for epoch: 32.526076316833496 secs
    
    Epoch 204 Loss 5.8755 Accuracy 0.2202
    Time taken for epoch: 31.501612186431885 secs
    
    Saving checkpoint for epoch 205 at ./outputs/20200424-000059/ckpts\ckpt-41
    Epoch 205 Loss 5.8554 Accuracy 0.2214
    Time taken for epoch: 32.93623495101929 secs
    
    Epoch 206 Loss 5.9552 Accuracy 0.2098
    Time taken for epoch: 32.64152383804321 secs
    
    Epoch 207 Loss 5.8511 Accuracy 0.2185
    Time taken for epoch: 33.4795184135437 secs
    
    Epoch 208 Loss 5.7961 Accuracy 0.2265
    Time taken for epoch: 31.434808254241943 secs
    
    Epoch 209 Loss 5.9082 Accuracy 0.2174
    Time taken for epoch: 32.05581045150757 secs
    
    Saving checkpoint for epoch 210 at ./outputs/20200424-000059/ckpts\ckpt-42
    Epoch 210 Loss 5.8937 Accuracy 0.2122
    Time taken for epoch: 33.18064570426941 secs
    
    Epoch 211 Loss 5.8388 Accuracy 0.2243
    Time taken for epoch: 32.1080482006073 secs
    
    Epoch 212 Loss 5.7673 Accuracy 0.2305
    Time taken for epoch: 33.43651556968689 secs
    
    Epoch 213 Loss 5.7709 Accuracy 0.2277
    Time taken for epoch: 31.85980749130249 secs
    
    Epoch 214 Loss 5.8870 Accuracy 0.2140
    Time taken for epoch: 31.53573250770569 secs
    
    Saving checkpoint for epoch 215 at ./outputs/20200424-000059/ckpts\ckpt-43
    Epoch 215 Loss 5.8577 Accuracy 0.2172
    Time taken for epoch: 32.807528257369995 secs
    
    Epoch 216 Loss 5.7872 Accuracy 0.2349
    Time taken for epoch: 32.41404747962952 secs
    
    Epoch 217 Loss 5.8133 Accuracy 0.2190
    Time taken for epoch: 32.827518463134766 secs
    
    Epoch 218 Loss 5.8385 Accuracy 0.2152
    Time taken for epoch: 32.21304678916931 secs
    
    Epoch 219 Loss 5.8088 Accuracy 0.2161
    Time taken for epoch: 32.11529612541199 secs
    
    Saving checkpoint for epoch 220 at ./outputs/20200424-000059/ckpts\ckpt-44
    Epoch 220 Loss 5.7859 Accuracy 0.2094
    Time taken for epoch: 33.281551361083984 secs
    
    Epoch 221 Loss 5.7531 Accuracy 0.2239
    Time taken for epoch: 32.879026889801025 secs
    
    Epoch 222 Loss 5.7651 Accuracy 0.2244
    Time taken for epoch: 32.80668544769287 secs
    
    Epoch 223 Loss 5.6969 Accuracy 0.2360
    Time taken for epoch: 31.80838179588318 secs
    
    Epoch 224 Loss 5.7757 Accuracy 0.2169
    Time taken for epoch: 31.821268558502197 secs
    
    Saving checkpoint for epoch 225 at ./outputs/20200424-000059/ckpts\ckpt-45
    Epoch 225 Loss 5.7614 Accuracy 0.2136
    Time taken for epoch: 32.70617485046387 secs
    
    Epoch 226 Loss 5.7222 Accuracy 0.2214
    Time taken for epoch: 32.36755657196045 secs
    
    Epoch 227 Loss 5.7005 Accuracy 0.2170
    Time taken for epoch: 32.28456354141235 secs
    
    Epoch 228 Loss 5.7114 Accuracy 0.2157
    Time taken for epoch: 32.70860958099365 secs
    
    Epoch 229 Loss 5.7370 Accuracy 0.2176
    Time taken for epoch: 31.852805137634277 secs
    
    Saving checkpoint for epoch 230 at ./outputs/20200424-000059/ckpts\ckpt-46
    Epoch 230 Loss 5.7092 Accuracy 0.2304
    Time taken for epoch: 32.81659483909607 secs
    
    Epoch 231 Loss 5.6904 Accuracy 0.2265
    Time taken for epoch: 32.69200944900513 secs
    
    Epoch 232 Loss 5.5740 Accuracy 0.2415
    Time taken for epoch: 32.8305184841156 secs
    
    Epoch 233 Loss 5.6458 Accuracy 0.2252
    Time taken for epoch: 31.709481716156006 secs
    
    Epoch 234 Loss 5.6313 Accuracy 0.2213
    Time taken for epoch: 31.85732388496399 secs
    
    Saving checkpoint for epoch 235 at ./outputs/20200424-000059/ckpts\ckpt-47
    Epoch 235 Loss 5.6318 Accuracy 0.2332
    Time taken for epoch: 32.77073693275452 secs
    
    Epoch 236 Loss 5.6590 Accuracy 0.2204
    Time taken for epoch: 32.39152240753174 secs
    
    Epoch 237 Loss 5.6216 Accuracy 0.2233
    Time taken for epoch: 32.74852418899536 secs
    
    Epoch 238 Loss 5.6372 Accuracy 0.2279
    Time taken for epoch: 35.53684735298157 secs
    
    Epoch 239 Loss 5.6745 Accuracy 0.2214
    Time taken for epoch: 31.673789739608765 secs
    
    Saving checkpoint for epoch 240 at ./outputs/20200424-000059/ckpts\ckpt-48
    Epoch 240 Loss 5.6444 Accuracy 0.2216
    Time taken for epoch: 32.78073525428772 secs
    
    Epoch 241 Loss 5.6589 Accuracy 0.2091
    Time taken for epoch: 32.986528158187866 secs
    
    Epoch 242 Loss 5.5904 Accuracy 0.2280
    Time taken for epoch: 32.80307745933533 secs
    
    Epoch 243 Loss 5.5778 Accuracy 0.2271
    Time taken for epoch: 31.623878717422485 secs
    
    Epoch 244 Loss 5.6325 Accuracy 0.2235
    Time taken for epoch: 32.24830627441406 secs
    
    Saving checkpoint for epoch 245 at ./outputs/20200424-000059/ckpts\ckpt-49
    Epoch 245 Loss 5.6024 Accuracy 0.2258
    Time taken for epoch: 32.996535539627075 secs
    
    Epoch 246 Loss 5.6243 Accuracy 0.2144
    Time taken for epoch: 33.58451437950134 secs
    
    Epoch 247 Loss 5.5630 Accuracy 0.2293
    Time taken for epoch: 33.302998781204224 secs
    
    Epoch 248 Loss 5.5749 Accuracy 0.2152
    Time taken for epoch: 31.95845365524292 secs
    
    Epoch 249 Loss 5.6050 Accuracy 0.2141
    Time taken for epoch: 31.592703342437744 secs
    
    Saving checkpoint for epoch 250 at ./outputs/20200424-000059/ckpts\ckpt-50
    Epoch 250 Loss 5.5537 Accuracy 0.2267
    Time taken for epoch: 32.66682744026184 secs
    
    Epoch 251 Loss 5.5411 Accuracy 0.2257
    Time taken for epoch: 32.69304084777832 secs
    
    Epoch 252 Loss 5.5749 Accuracy 0.2278
    Time taken for epoch: 33.10207915306091 secs
    
    Epoch 253 Loss 5.5603 Accuracy 0.2286
    Time taken for epoch: 32.626872539520264 secs
    
    Epoch 254 Loss 5.5513 Accuracy 0.2173
    Time taken for epoch: 31.68764352798462 secs
    
    Saving checkpoint for epoch 255 at ./outputs/20200424-000059/ckpts\ckpt-51
    Epoch 255 Loss 5.6054 Accuracy 0.2182
    Time taken for epoch: 33.205010414123535 secs
    
    Epoch 256 Loss 5.5428 Accuracy 0.2329
    Time taken for epoch: 32.32252025604248 secs
    
    Epoch 257 Loss 5.5888 Accuracy 0.2101
    Time taken for epoch: 32.512999296188354 secs
    
    Epoch 258 Loss 5.5257 Accuracy 0.2186
    Time taken for epoch: 31.463117599487305 secs
    
    Epoch 259 Loss 5.5051 Accuracy 0.2241
    Time taken for epoch: 31.926247119903564 secs
    
    Saving checkpoint for epoch 260 at ./outputs/20200424-000059/ckpts\ckpt-52
    Epoch 260 Loss 5.4954 Accuracy 0.2269
    Time taken for epoch: 33.06800627708435 secs
    
    Epoch 261 Loss 5.5362 Accuracy 0.2218
    Time taken for epoch: 32.46455383300781 secs
    
    Epoch 262 Loss 5.5016 Accuracy 0.2190
    Time taken for epoch: 33.53307867050171 secs
    
    Epoch 263 Loss 5.4462 Accuracy 0.2332
    Time taken for epoch: 31.592995405197144 secs
    
    Epoch 264 Loss 5.5039 Accuracy 0.2279
    Time taken for epoch: 31.52744460105896 secs
    
    Saving checkpoint for epoch 265 at ./outputs/20200424-000059/ckpts\ckpt-53
    Epoch 265 Loss 5.4864 Accuracy 0.2232
    Time taken for epoch: 32.6013708114624 secs
    
    Epoch 266 Loss 5.5226 Accuracy 0.2296
    Time taken for epoch: 33.290045738220215 secs
    
    Epoch 267 Loss 5.4265 Accuracy 0.2322
    Time taken for epoch: 33.31650400161743 secs
    
    Epoch 268 Loss 5.4408 Accuracy 0.2262
    Time taken for epoch: 31.076684713363647 secs
    
    Epoch 269 Loss 5.5111 Accuracy 0.2167
    Time taken for epoch: 31.94932532310486 secs
    
    Saving checkpoint for epoch 270 at ./outputs/20200424-000059/ckpts\ckpt-54
    Epoch 270 Loss 5.4420 Accuracy 0.2326
    Time taken for epoch: 33.21883201599121 secs
    
    Epoch 271 Loss 5.3804 Accuracy 0.2362
    Time taken for epoch: 32.79451012611389 secs
    
    Epoch 272 Loss 5.3734 Accuracy 0.2377
    Time taken for epoch: 33.4981324672699 secs
    
    Epoch 273 Loss 5.3677 Accuracy 0.2298
    Time taken for epoch: 31.395419359207153 secs
    
    Epoch 274 Loss 5.3766 Accuracy 0.2239
    Time taken for epoch: 32.19995427131653 secs
    
    Saving checkpoint for epoch 275 at ./outputs/20200424-000059/ckpts\ckpt-55
    Epoch 275 Loss 5.4710 Accuracy 0.2176
    Time taken for epoch: 32.8692741394043 secs
    
    Epoch 276 Loss 5.3421 Accuracy 0.2377
    Time taken for epoch: 32.85851287841797 secs
    
    Epoch 277 Loss 5.3488 Accuracy 0.2330
    Time taken for epoch: 32.90205693244934 secs
    
    Epoch 278 Loss 5.3711 Accuracy 0.2351
    Time taken for epoch: 32.3742151260376 secs
    
    Epoch 279 Loss 5.3810 Accuracy 0.2298
    Time taken for epoch: 31.67891550064087 secs
    
    Saving checkpoint for epoch 280 at ./outputs/20200424-000059/ckpts\ckpt-56
    Epoch 280 Loss 5.2990 Accuracy 0.2321
    Time taken for epoch: 33.284619092941284 secs
    
    Epoch 281 Loss 5.4318 Accuracy 0.2220
    Time taken for epoch: 32.81750965118408 secs
    
    Epoch 282 Loss 5.3557 Accuracy 0.2340
    Time taken for epoch: 33.53207206726074 secs
    
    Epoch 283 Loss 5.3137 Accuracy 0.2368
    Time taken for epoch: 31.81239891052246 secs
    
    Epoch 284 Loss 5.3290 Accuracy 0.2326
    Time taken for epoch: 31.37002968788147 secs
    
    Saving checkpoint for epoch 285 at ./outputs/20200424-000059/ckpts\ckpt-57
    Epoch 285 Loss 5.3806 Accuracy 0.2287
    Time taken for epoch: 32.96224617958069 secs
    
    Epoch 286 Loss 5.3303 Accuracy 0.2324
    Time taken for epoch: 33.1435112953186 secs
    
    Epoch 287 Loss 5.2905 Accuracy 0.2299
    Time taken for epoch: 32.89712452888489 secs
    
    Epoch 288 Loss 5.2307 Accuracy 0.2344
    Time taken for epoch: 31.410616397857666 secs
    
    Epoch 289 Loss 5.3134 Accuracy 0.2269
    Time taken for epoch: 31.973753213882446 secs
    
    Saving checkpoint for epoch 290 at ./outputs/20200424-000059/ckpts\ckpt-58
    Epoch 290 Loss 5.3592 Accuracy 0.2268
    Time taken for epoch: 32.78321099281311 secs
    
    Epoch 291 Loss 5.3007 Accuracy 0.2317
    Time taken for epoch: 32.59153342247009 secs
    
    Epoch 292 Loss 5.3051 Accuracy 0.2405
    Time taken for epoch: 32.69443893432617 secs
    
    Epoch 293 Loss 5.2988 Accuracy 0.2299
    Time taken for epoch: 31.684515953063965 secs
    
    Epoch 294 Loss 5.2783 Accuracy 0.2330
    Time taken for epoch: 32.07484006881714 secs
    
    Saving checkpoint for epoch 295 at ./outputs/20200424-000059/ckpts\ckpt-59
    Epoch 295 Loss 5.3596 Accuracy 0.2261
    Time taken for epoch: 32.61194944381714 secs
    
    Epoch 296 Loss 5.3519 Accuracy 0.2248
    Time taken for epoch: 32.64351487159729 secs
    
    Epoch 297 Loss 5.2681 Accuracy 0.2363
    Time taken for epoch: 32.683409214019775 secs
    
    Epoch 298 Loss 5.3137 Accuracy 0.2255
    Time taken for epoch: 35.489052534103394 secs
    
    Epoch 299 Loss 5.3141 Accuracy 0.2324
    Time taken for epoch: 32.225311517715454 secs
    
    Saving checkpoint for epoch 300 at ./outputs/20200424-000059/ckpts\ckpt-60
    Epoch 300 Loss 5.2749 Accuracy 0.2331
    Time taken for epoch: 32.732651472091675 secs
    
    


```python

```
