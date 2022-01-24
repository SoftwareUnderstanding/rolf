# TabNet
(Yet another) PyTorch implementation of [TabNet](http://https://arxiv.org/abs/1908.07442 "TabNet").

Implements base model code, including categorical embeddings, but does not implement interpretability measures as outlined in the paper
## Summary

TabNet should **not act as a replacement to boosted methods** for typical data science use-cases. However, it may provide improved performance in use-cases where the labelled dataset is large (e.g., millions of examples), or in situations where only an unsupervised model is needed (e.g., fraud detection).

- **Performance:** While the paper demonstrates promising results, my TabNet implemention underperformed XGBoost in `adult_census` and only slightly outperformed XGBoost in `forest_census` (likely driven by the larger size of the dataset). These results are produced without hyperparamater tuning.
- **Training time:** The training time for TabNet models is considerably higher than the XGBoost counterpart on CPU, though this difference is lower on GPU. As such, TabNet should only be considered when plenty of samples are available (e.g., as with `Forest Cover`)
- **Interpretability:** Aside from (i) visualising the embedding space of categorical features, and (ii) providing some intuition on which features the model was attending to while predicting on a text example, the vanilla TabNet model does not provide much additional interpretability over the importance plots already available in XGBoost. 


## Results

| Dataset  | XGBoost  | TabNet |
| :------------ |:------------:| :-----:|
| [Adult Census](https://www.kaggle.com/uciml/adult-census-income "Adult Census")     | 0.867 | 0.847 |
| [Forest Cover](https://www.kaggle.com/uciml/forest-cover-type-dataset "Forest Cover")      |  0.919        |   0.953 |

Note: Tests can be replicated by running the appropriate files in `/tests/examples`. Datasets will be downloaded with the repository.

## How to use

The TabNet implementation is highly configurable, facilitated by the large number of input parameters. A full list of parameters can be found at the top of `/src/train.py`.

Supports Pandas or numpy arrays as inputs. Handles categorical inputs out-of-box, provided they are correctly configured (see `forest_cover.csv` for an example).

A TabNet model can be trained as follows: 
```python
import sys
sys.path.append(os.path.abspath("../../src/"))
from train import TabNet

fc_tabnet_model = TabNet(model_params=model_params)
fc_tabnet_model.fit(
	X_train,
	y_train,
	X_val,
	y_val,
	train_params=train_params,
	save_params={
		"model_name": data_params["model_save_name"],
		"tensorboard_folder": "../../runs/",
		"save_folder": data_params["model_save_dir"],
	},
)
fc_tabnet_model = TabNet(save_file=save_file)
y_tabnet_val_pred = fc_tabnet_model.predict(X_val)
```

Training / validation losses are logged to Tensorboard. Run `tensorboard --logdir=./runs/` in your terminal to access this data.

## Backlog
- [ ] Add ability to handle `NaN` and/or missing inputs
- [ ] Add feature mask interpretability measure
- [ ] Add embedding extraction and visualisation
- [ ] Support for using TabNet for anomaly detection
- [ ] Support for using TabNet as an imputer

