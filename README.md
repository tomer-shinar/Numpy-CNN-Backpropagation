# Train
To train a model run the command
```
python3 train.py [device]
```
This will train a model on `train.csv` and validate it on `validation.csv`.
The model will be saved to `model.pkl`.

`device` is the device that model will train on, `gpu` / `cpu`. default is `cpu`.

# Predict
To predict the model run the command:
```
python3 predict.py [test file] [preds file] [model file]
```
`test file` is the file of the test set to predict on. <br>
`preds file` is the file that the predictions will be written to.<br>
`model file` is the file where the model is saved.

Example (the command you should run):
```
python3 predict.py test.csv output model.pkl
```