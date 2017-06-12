import numpy as np
import json
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from traincommon import *
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
import h5py


def multi2splithot(Yf):
    """Takes a hot with multiple occcurrences and splits it into multiple binary hots one for each entry of the input

    e.g. multiple fingers at time => array of 2-hot for every finger
    """
    if len(Yf.shape) == 1:
        Yhot = []                
        for j in range(np.min(Yf),np.max(Yf)+1):
            f = np.where(Yf == j)[0]
            Y = np.zeros(Yf.shape)
            Y[f] = 1
            # Yhot = np.zeros((X.shape[0], clazzes)) # a.max()+1
            # Yhot[np.where(Yf != finger)[0],0] = 1
            # Yhot[np.where(Yf == finger)[0],1] = 1
            Yhot.append(index2onehot(Y))
    else:
        print "multi2splithot","hot mode"
        Yhot = []
        for j in range(0,Yf.shape[1]):
            Y = np.zeros((Yf.shape[0],2))
            Y[Yf[:,j] == 1,1] = 1
            Y[:,0] = 1-Y[:,1] # others
            
            #Y = np.zeros((Yf.shape[0],1)) # label 1 means correct
            #Y[f] = 1 # 1 when it is
            # Yhot = np.zeros((X.shape[0], clazzes)) # a.max()+1
            # Yhot[np.where(Yf != finger)[0],0] = 1
            # Yhot[np.where(Yf == finger)[0],1] = 1
            Yhot.append(Y)
    return Yhot

class Wrapper:
    def __init__(self,clazz,hotout,**kwargs):
        self.kwargs = kwargs
        self.clazz = clazz
        self.hotout = hotout
        self.metrics_names = ['','accuracy']
    def fit(self,xx,y,validation_data=None,validation_split=None,**kwargs):
        # x is [x1,x2]
        self.clf = self.clazz(**self.kwargs)
        z = [np.reshape(q,(q.shape[0],-1)) for q in xx]
        x = np.concatenate(z,axis=1)
        print x.shape
        self.clf.fit(x,y if self.hotout else onehot2index(y))
        return 1
    def predict(self,xx,verbose=False):
        z = [np.reshape(q,(q.shape[0],-1)) for q in xx]
        x = np.concatenate(z,axis=1)
        return index2onehot(self.clf.predict(x))
    def get_weights(self):
        return None
    def set_weights(self,x):
        pass
    def evaluate(self,xx,Y,verbose=False):
        z = [np.reshape(q,(q.shape[0],-1)) for q in xx]
        x = np.concatenate(z,axis=1)
        Yp = self.clf.predict(x)
        if self.hotout:
            Yp = onehot2index(Yp)
        M = confusion_matrix(Yp, onehot2index(Y).astype(np.int32))
        return (0,cm_info(M)["accuracy"])
    def save_weights(self,filename):
        #f = h5py.File(filename, "w")
        #dset = f.create_dataset("model", (1,), dtype='i')
        #binary_blob = pickle.dumps(self.clf)
        #dset.attrs["pickle"] = np.void(binary_blob)
        pickle.dump(self.clf,open(filename,"wb"))
    def load_weights(self,filename):
        #f = h5py.File(filename, "r")
        #dset = f["model"]
        #out = dset.attrs["pickle"]
        self.clf = pickle.load(open(filename,"rb")) #out.tostring())
    def to_json(self):
        return json.dumps(dict(model="lda",kwargs=self.kwargs))
    def from_json(self,q):
        self.clf = self.clazz(**q["kwargs"])

class LDAWrapper(Wrapper):
    def __init__(self,**kwargs):
        Wrapper.__init__(self,LinearDiscriminantAnalysis,False,**kwargs)

class SVMWrapper(Wrapper):
    def __init__(self,**kwargs):
        Wrapper.__init__(self,SVC,False,**kwargs)

def cm_info(cm):
    def getAccuracy(matrix):
        # sum(diag(mat))/(sum(mat))
        sumd = np.sum(np.diagonal(matrix))
        sumall = np.sum(matrix)
        sumall = np.add(sumall, 0.00000001)
        return sumd/sumall

    def getPrecision(matrix):
        # diag(mat) / rowSum(mat)
        sumrow = np.sum(matrix, axis=1)
        sumrow = np.add(sumrow, 0.00000001)
        precision = np.divide(np.diagonal(matrix), sumrow)
        return np.sum(precision)/precision.shape[0]

    def getRecall(matrix):
        # diag(mat) / colsum(mat)
        sumcol = np.sum(matrix, axis=0)
        sumcol = np.add(sumcol, 0.00000001)
        recall = np.divide(np.diagonal(matrix), sumcol)
        return np.sum(recall)/recall.shape[0]

    def getf1(matrix):
        # 2*precision*recall/(precision+recall)
        precision = getPrecision(matrix)
        recall = getRecall(matrix)
        return (2*precision*recall)/(precision+recall)

    return dict(precision=getPrecision(cm), recall=getRecall(cm), f1=getf1(cm),accuracy=getAccuracy(cm))

# supports any based value
def index2onehot(x, nminmax=None):
    if nminmax is None:
        nminmax = (np.min(x), np.max(x))


    Yhot = np.zeros((x.size, int(nminmax[1]-nminmax[0]+1)))
    Yhot[np.arange(x.size), (x-nminmax[0]).astype(np.int32)] = 1
    return Yhot

# returns index 0 based
def onehot2index(x):
    return np.argmax(x, axis=1)

def vstack1(a):
    return np.hstack([np.reshape(x[0], (x[0].size,)) for x in a])

def leavekoutsplit(usersfield, test_subjects, val_count):
    uf = np.reshape(usersfield, (usersfield.size,))
    uu = set(uf.tolist())
        
    tr_idx = np.array(list(uu - set(test_subjects)))
    te_idx = list(test_subjects)
    np.random.shuffle(tr_idx)
    tr_idx = tr_idx.tolist()

    if val_count != 0:
        val_idx = tr_idx[:val_count]
        tr_idx = tr_idx[len(val_idx):]
    else:
        val_idx = []


    # then map back to the original vectors
    oo = []
    ot = [tr_idx, te_idx, val_idx]
    if len(val_idx) == 0:
        del ot[-1]
    for a in ot:
        oo.append(vstack1([np.where(uf == x) for x in a]))
    if len(val_idx) == 0:
        oo.append(np.array(()))
    return dict(indices=tuple(oo), users=tuple(ot))

def scaler2dict(ss):
    return dict(params=ss.get_params(deep=True),attribs=dict(scale_=ss.scale_,mean_=ss.mean_,n_samples_seen_=ss.n_samples_seen_))

def dict2scaler(scalerinfo):
    ss = StandardScaler()
    ss.set_params(**scalerinfo["params"])
    for k,v in scalerinfo["attribs"].iteritems():
        setattr(ss,k,v)
    return ss

def labeledshufflesplit(usersfield, test_factor, val_factor,shuffle = True):
    uf = np.reshape(usersfield, (usersfield.size,))
    uu = list(set(uf.tolist()))
    if shuffle:
        np.random.shuffle(uu)

    # work on users lvels
    te_idx = uu[:int(test_factor * len(uu))]
    tr_idx = uu[len(te_idx):]
    val_idx = tr_idx[:int(val_factor * len(tr_idx))]
    if val_factor != 0 and len(val_idx) == 0:
        val_idx = [tr_idx[0]]

    tr_idx = tr_idx[len(val_idx):]

    # then map back to the original vectors
    oo = []
    ot = [tr_idx, te_idx, val_idx]
    if len(val_idx) == 0:
        del ot[-1]
    for a in ot:
        oo.append(vstack1([np.where(uf == x) for x in a]))
    if len(val_idx) == 0:
        oo.append(np.array(()))
    return dict(indices=tuple(oo), users=tuple(ot))

def asvector(x):
    return np.reshape(x, (x.size,))

class SplittableDataset:
    def __init__(self,X):
        self.X = X
        self.X_tr = None
        self.X_te = None
        self.X_va = None
    def split(self,tr_idx,te_idx,val_idx=None):
        X = self.X
        # fix me here
        self.X_tr = X[tr_idx]
        self.X_te = X[te_idx]
        if val_idx is not None and len(val_idx) > 0:
            self.X_va = X[val_idx]
        else:
            self.X_va = None
    @property
    def shape(self):
        return self.X.shape
    def apply(self,fx):
        if self.X_tr is not None:
            self.X_tr = fx("train",self.X_tr)
            self.X_te = fx("test",self.X_te)
            if self.X_va is not None:
                self.X_va = fx("validation",self.X_va)
        self.X = fx("all",self.X)
    def reordertrain(self,indices):
        if self.X_tr is not None:
            self.X_tr = self.X_tr[indices]
    @property
    def all(self):
        return self.X
    @property
    def train(self):
        return self.X_tr
    @property
    def test(self):
        return self.X_te
    @property
    def validation(self):
        return self.X_va

class MultiSplittableDataset:
    def __init__(self,X):
        self.Xa = X       
        self.X = [SplittableDataset(x) for x in X]
        self.X_tr = None
        self.X_te = None
        self.X_va = None
    def split(self,tr_idx,te_idx,val_idx=None):
        # split each contained block
        for a in self.X:
            a.split(tr_idx,te_idx,val_idx)        
        self._sync()
    def _sync(self):
        #  esxpose met
        self.X_tr = [a.train for a in self.X]
        self.X_te = [a.test for a in self.X]
        if self.X[0].validation is not None:
            self.X_va = [a.validation for a in self.X]
        else:
            self.X_va = None        
    @property
    def shape(self):
        return self.X[0].shape
    def apply(self,fx):
        self.Xa = [fx("all",q) for q in self.Xa]
        for a in self.X:
            a.apply(fx)
        self._sync()
    def reordertrain(self,indices):
        for a in self.X:
            a.reordertrain(indices)
        self._sync()
    @property
    def all(self):
        return self.Xa
    @property
    def train(self):
        return self.X_tr
    @property
    def test(self):
        return self.X_te
    @property
    def validation(self):
        return self.X_va