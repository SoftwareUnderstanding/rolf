# Classify programming language
It uses ULMFIT model [https://arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146)
Model with default parameters achieves ~96% accuracy on validation set.

It requires:
- python 3.7
- gpu for training (recommended)


Basing training classifier:
```bash
python train.py --input data.csv --model-path ulmfit.pkl 
```
Running service that predicts programming language:
```bash
python app.py application.conf
```

Example of curl:
```bash
curl -X POST -H "Content-Type:application/json" 'http://0.0.0.0:8010/predict' 
--data '{"text": "#include <iostream>"}'
```

Running service from docker:
```bash
docker build -t detectorService -f DockerfileService .
```
```bash
docker run --rm -it detectorService python app.py application.conf
```

Further improvements:
- Compare with baseline models such as CNN, N-grams features
- Try different tokenizations instead of just on letters
- Try different hyperparameters, bigger model.
