import sys
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.ndimage import rotate
import numpy
from layers import *
import pandas as pd


BATCH_SIZE = 64
LEARNING_RATE = 0.0004
EPOCHS = 100
LR_DECAY = 0.3
DROP_RATE = 0.05
WEIGHT_DECAY = 0.2
MIXUP_ALPHA = 0.4
NOISE_RATE = 0.1

np.random.seed(15)


def translate(imgs, shift, direction, roll=True):
    #assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    new_imgs = []
    for i,img in enumerate(imgs):
        img = img.copy()
        if direction[i] == 'right':
            right_slice = img[:, -shift[i]:].copy()
            img[:, shift[i]:] = img[:, :-shift[i]]
            if roll:
                img[:,:shift[i]] = np.fliplr(right_slice)
        if direction[i] == 'left':
            left_slice = img[:, :shift[i]].copy()
            img[:, :-shift[i]] = img[:, shift[i]:]
            if roll:
                img[:, -shift[i]:] = left_slice
        if direction[i] == 'down':
            down_slice = img[-shift[i]:, :].copy()
            img[shift[i]:, :] = img[:-shift[i],:]
            if roll:
                img[:shift[i], :] = down_slice
        if direction[i] == 'up':
            upper_slice = img[:shift[i], :].copy()
            img[:-shift[i], :] = img[shift[i]:, :]
            if roll:
                img[-shift[i]:,:] = upper_slice
        new_imgs.append(img)
    return np.array(new_imgs)


def rotate_img(imgs, angle, bg_patch=(5,5)):
    #assert len(img.shape) <= 3, "Incorrect image shape"
    new_imgs = []
    for i,img in enumerate(imgs):
        img = numpy.array(img.get())
        rgb = len(img.shape) == 3
        if rgb:
            bg_color = numpy.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
        else:
            bg_color = numpy.mean(img[:bg_patch[0], :bg_patch[1]])
        img = rotate(img, angle[i], reshape=False)
        mask = [img <= 0, numpy.any(img <= 0, axis=-1)][rgb]
        img[mask] = bg_color
        new_imgs.append(img)

    return np.array(new_imgs)
    
    
def create_aug_data1(X,y):
    dir=['up','down','left','right']
    trans = translate(X, direction=[dir[numpy.random.randint(low=0,high=4)] for i in range(X.shape[0])], shift=np.random.randint(1, 16,size=X.shape[0]))
    return trans,y


def create_aug_data2(X,y):
    rots = rotate_img(X, angle=np.random.randint(0, 359,size=X.shape[0]))
    return rots,y


def load_img(rows):
  return rows.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


  
class DataLoader:

    @staticmethod
    def train_test_split(csv_file, batch_size=64, is_conv=True,normilize=True, device='cpu', labeled=True):
        if device=='cpu':
            np_device=np
        elif device=='gpu':
            np_device=cp
        X,y = DataLoader.read_data(csv_file,np_device, labeled=labeled)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=6)
        return DataLoader(X_train, y_train, batch_size=batch_size, is_conv=is_conv,train=True,normilize=normilize,device=device),DataLoader(X_val, y_val, batch_size=1, is_conv=is_conv,train=False,normilize=normilize,device=device)

    @staticmethod
    def create(csv_file, batch_size=64, is_conv=True, train=False,normilize=True, device='cpu', labeled=True):
        if device=='cpu':
            np_device=np
        elif device=='gpu':
            np_device=cp
        X,y = DataLoader.read_data(csv_file,np_device, labeled=labeled)
        return DataLoader(X, y, batch_size=batch_size, is_conv=is_conv, train=train,normilize=normilize, device=device, labeled=labeled)

    @staticmethod
    def read_data(csv_file,np_device, labeled=True):
        df = pd.read_csv(csv_file, header=None).to_numpy()
        if labeled:
            return np_device.asarray(df[:, 1:]), np_device.asarray(df[:, 0].astype(int) - 1)
        else:
            return np_device.asarray(df[:, 1:]), None

    def __init__(self, X, Y, batch_size=64, is_conv=True,train=False, normilize=True, device='cpu', labeled=True):
        self.device=device
        if device=='cpu':
            self.np_device=np
        elif device=='gpu':
            self.np_device=cp
        self.np_device.random.seed(15)

        self.is_conv = is_conv
        self.batch_size = batch_size
        self.train = train
        if is_conv:
            X = load_img(X)
        self.X = X
        self.labeled = labeled
        if self.labeled:
            self.y = self.np_device.eye(10)[Y]

        if normilize:
            self.X = normilizeData(self.X)

    def add(self,X,y):
        self.X=self.np_device.append(self.X, X,axis=0)
        self.y=self.np_device.append(self.y, y,axis=0)

    def __iter__(self):
        if self.train:
            initX,initY=self.X,self.y
            if self.device=='gpu':
                initX,initY=self.X.get(),self.y.get()
            aug1X, aug1Y = create_aug_data1(initX,initY)
            aug2X, aug2Y = mixup(initX,initY,MIXUP_ALPHA)#create_aug_data2(initX,initY)
            mixX, mixY = mixup(initX,initY,MIXUP_ALPHA)
            X,y = initX, initY
            X,y = np.append(X, aug1X,axis=0), np.append(y, aug1Y,axis=0)
            X,y = np.append(X,aug2X,axis=0), np.append(y,aug2Y,axis=0)
            X,y = np.append(X,mixX,axis=0), np.append(y,mixY,axis=0)
            X,y = shuffle(X,y)
            X = noise(X)
            X,y = self.np_device.asarray(X), self.np_device.asarray(y)
            for i in range(len(X)//self.batch_size):
                yield (X[i*self.batch_size:(i+1)*self.batch_size], y[i*self.batch_size:(i+1)*self.batch_size])
        else:
            for i in range(len(self.X)//self.batch_size):
                if self.labeled:
                    yield (self.X[i*self.batch_size:(i+1)*self.batch_size], self.y[i*self.batch_size:(i+1)*self.batch_size])
                else:
                    yield self.X[i*self.batch_size:(i+1)*self.batch_size]

    def __len__(self):
        return len(self.X)

    def all(self):
        all_batch = 200
        for i in range(len(self.X)//all_batch):
            yield (self.X[i*all_batch:(i+1)*all_batch], self.y[i*all_batch:(i+1)*all_batch])


def mul(X,alpha):
    mix = []
    for x,a in zip(X,alpha):
        mix.append(x*a)
    return np.array(mix)


def mixup(X,y, alpha):
    lambd = np.random.beta(alpha, alpha, X.shape[0])
    Xs,ys = shuffle(X,y)
    return mul(X,lambd)+mul(Xs,1-lambd), mul(y,lambd)+mul(ys,1-lambd)


def shuffle(X,y):
    p = np.random.permutation(len(X))
    return X[p], y[p]


def noise(X):
    mask = np.random.random_sample(X.shape)
    mask = mask < (1-NOISE_RATE)
    X = np.multiply(X, mask)
    return X


def normilizeData(X):
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / std


class Optimizer:
    def __init__(self, lr, lr_decay, weight_decay):
        self.lr = lr
        self.initial_lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay

    def step(self, model, y, y_pred, epoch):
        error = y_pred - y
        model.backward(error)
        model.fit(self.lr, self.weight_decay)

        # lr decay
        self.lr = self.initial_lr * (1 / (1 + self.lr_decay * (epoch+1)))


def train(model, train_loader, val_loader, optimizer, epochs):
    acc = []
    for epoch in range(epochs):
        print(f'training on epoch {epoch}')
        model.train()
        i=0
        for X,y in train_loader:
            y_pred = model.forward(X)
            optimizer.step(model, y, y_pred, epoch)
            i+=1
        val_acc,val_loss = test(model, val_loader)
        train_acc,train_loss = test(model, train_loader)
        print(
            f'epoch {epoch + 1} train accuracy: {train_acc:.2f} val accuracy: {val_acc:.2f}')
        print(
            f'epoch {epoch + 1} train loss: {train_loss:.2f} val loss: {val_loss:.2f}')
    return acc


def test(model, data_loader):
    model.eval()
    n = 0
    acc=0
    loss = 0
    for X, y in data_loader.all():
        y_pred = model.forward(X)
        loss += cross_entropy(y, y_pred,device=model.device) / len(y)
        y_pred = y_pred.argmax(1)
        y=model.device.argwhere(y==1)[:,1]
        acc += (y_pred == y).sum() * 100 / len(y)
        n += 1
    return (acc / n), (loss / n)
    

def cross_entropy(y, y_pred, device):
    return -(y * device.log(y_pred)).sum()


def create_model():
    return Model(
        layers=[
                Conv2d(in_channels=3,out_channels=32,kernel_size=3, padding=True),
                BatchNorm((32, 32, 32)),
                Relu(),
                MaxPooling2D(kernel_size=2),
                Dropout(drop_rate=DROP_RATE),
                
                Conv2d(in_channels=32,out_channels=64,kernel_size=3, padding=True),
                BatchNorm((16, 16, 64)),
                Relu(),
                MaxPooling2D(kernel_size=2),
                Dropout(drop_rate=DROP_RATE),

                Flatten(),
                Linear(in_dim=8*8*64,out_dim=128),
                Relu(),
                Dropout(drop_rate=DROP_RATE),
                Linear(in_dim=128,out_dim=10),
                Softmax()
        ]
  )


def main():
    device = 'cpu'
    if len(sys.argv)>1:
        device = sys.argv[1]
        if device == 'gpu':
            import cupy as cp
     
    print(f'loading data on {device}')
    train_loader = DataLoader.create('train.csv', batch_size=BATCH_SIZE, train=True, device=device)
    val_loader = DataLoader.create('validate.csv', batch_size=1, device=device)
    
    print(f'creating model on {device}')
    model = create_model()
    model.to(device=device)
    optimizer = Optimizer(lr=LEARNING_RATE, lr_decay=LR_DECAY, weight_decay=WEIGHT_DECAY)
 
    train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, epochs=EPOCHS)
    
    
if __name__== '__main__':
    main()