from src.Power_Dataframe import Power_Dataframe
import torch
import itertools
from src.models.neural_networks import *


if __name__=='__main__':
    # Run some example tests here

    PATH='./TEMP.db' # Insert path of the DB here (e.g. './TEMP.db')
    DF = Power_Dataframe(results_filename=PATH, sequential=True)


    net = MNIST_FCNN
    opt_params=[]
    # for comb in itertools.product(['MNIST'], ['SGD','ADAM'], [{'lr':0.01,'momentum':0.9},{'lr':0.01,'momentum':0.8},{'lr':0.1,'momentum':0.9}], [10000]):
    for comb in itertools.product(['MNIST'], ['SGD','ADAM'], [{'lr':0.01},{'lr':0.02},{'lr':0.03}], [10000]): 
        if comb[1]=='ADAM': 
            opt_fun = torch.optim.Adam
        elif comb[1]=='SGD':
            opt_fun = torch.optim.SGD

        opt_params.append(comb[2])
        network_params = {'hidden_sizes':[64,32]}

        df=DF.get(dataset=comb[0], mb_size=comb[3], opt_fun=opt_fun, optimizer_params=comb[2], ignore_optimizer_params=['params'], network_fun=net, network_params=network_params, 
                 loss_function=nn.CrossEntropyLoss(), loss_params={}, trials=6, epochs=7, pretraining_status=0, overlap_ratio=0, IGNORED_FIELDS=None)
    print('done')

