import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter

n_sne=7000

tsne=TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results=tsne.fit_transform(df.loc[rndperm[:n_k]])

if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('-l', dest='length', type=int, default=32, help='time window length')
    args = p.parse_args()

    if not os.path.exists('./tsne-{}'.format(args.length)):
        os.mkdir('./tsne-{}'.format(args.length))

    dataset_list=['cairo', 'kyoto11', 'kyoto7', 'kyoto8', 'milan']
    for item in dataset_list:
        ## load numpy file
        X=np.load('../npy/{}-x-noidle.npy'.format(item), allow_pickle=True)
        if -1*(X.shape[0]%args.length)!=0:
            X=X[:-1*(X.shape[0]%args.length)]
        X=X.reshape(-1, args.length, 1)
        if X.shape[0]>n_sne:
            X=X[:n_sne]
        print(X.shape)

        y=np.load('../npy/{}-y-noidle.npy'.format(item), allow_pickle=True)
        if -1*(y.shape[0]%args.length)!=0:
            y=y[:-1*(y.shape[0]%args.length)]
        y=y.reshape(-1, args.length, 1)
        if y.shape[0]>n_sne:
            y=y[:n_sne]
        print(y.shape); y=np.array(y, dtype=int)

        y_=[]
        for i in range(y.shape[0]):
            y_.append(np.argmax(np.bincount(y[i].flatten())))
        print(Counter(y_))
        # X=np.array(X_, dtype=object); X = sequence.pad_sequences(X, maxlen=32, dtype='int32')
        y_=np.array(y_, dtype=object); y_=y_.reshape(-1,1); print(y_.shape)

        tsne_result=tsne.fit_transform(X.reshape(-1, args.length), y_)

        labels=np.unique(y_)
        scatter=plt.scatter(tsne_result[:,0], tsne_result[:,1], c=y_)
        handles,_=scatter.legend_elements(prop='colors')
        plt.legend(handles,labels)
        plt.savefig('./tsne-{}/{}'.format(args.length, item))
