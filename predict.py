import pickle
import sys
from train import DataLoader
import numpy as np


def main():
    test_file, preds_file, model_file = sys.argv[1:4]

    # load model
    f = open(model_file, 'rb')
    model = pickle.load(f)
    f.close()
    model.eval()

    # load data
    test_loader = DataLoader.create(csv_file=test_file, batch_size=1, train=False,normilize=True, device='cpu', labeled=False)
    # predict
    y_pred = model.forward(test_loader.X)
    preds = np.argmax(y_pred, axis=1)
    preds += 1

    # save predictions
    f = open(preds_file, "w")
    preds = [str(p) for p in preds]
    f.write("\n".join(preds))
    f.close()


if __name__ == '__main__':
    main()
