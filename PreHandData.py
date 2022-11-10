
from utils_HU import *
from data_process import create_dataset_for_train


#  Prehandle data
datasets = ['kiba']
print('datasets:',datasets)
cuda_name = 'cuda:6'
print('cuda_name:', cuda_name)
fold = 1
cross_validation_flag = True
for dataset in datasets:
    if dataset == 'davis':
        home_path = 'PreData'
    else:
        home_path = 'PreKiba'
    print('starting...')
    train_data, valid_data, valid_drugs, train_drugs, smile_graph, train_prot_keys, valid_prot_keys = create_dataset_for_train(
        dataset, fold)
   
    pretrain_path = home_path + '/train_' + str(fold) + '.pt'
    prevalid_path = home_path + '/valid_' + str(fold) + '.pt'
    torch.save(train_data, pretrain_path)
    torch.save(valid_data, prevalid_path)
    print(dataset+str(fold)+'succeed!!!!!!')

