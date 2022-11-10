from Kiba_gnn import GNNNet
from utils import *
from emetrics import *
# import torch.nn as nn

# datasets = [['davis', 'kiba'][int(sys.argv[1])]]
datasets = ['kiba']
print('datasets:',datasets)
#cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][int(sys.argv[2])]
cuda_name = 'cuda:6'
print('cuda_name:', cuda_name)
#fold = [0, 1, 2, 3, 4][int(sys.argv[3])]
fold = 1
cross_validation_flag = True
# print(int(sys.argv[3]))

TRAIN_BATCH_SIZE = 768
TEST_BATCH_SIZE = 768
print('TRAIN_BATCH_SIZE',TRAIN_BATCH_SIZE)
LR = 0.0006
NUM_EPOCHS = 2000
model_name = 'Kiba_train'
print("model_name",model_name)
print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'Results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Main program: iterate over different datasets
result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')#cuda:0
#device = torch.device('cpu')
model = GNNNet()
model_st = GNNNet.__name__
dataset = datasets[0]
model_file_name = 'models/'+model_name+ '.model'
model.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for dataset in datasets:
    if dataset == 'kiba':
        path_data = 'PreKiba'
    else :
        path_data = 'PreData'
    print("data:",path_data)
    train_path = path_data+'/train_'+str(fold)+'.pt'
    valid_path = path_data+'/valid_'+str(fold)+'.pt'
    train_data = torch.load(train_path)
    valid_data = torch.load(valid_path)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate, num_workers=8, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                               collate_fn=collate, num_workers=8, pin_memory=True)

    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1

    count = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)  
        val = get_mse(G, P)
        print('valid result:', val, best_mse)
        if val < best_mse:
            best_mse = val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
        
            # cindex = get_cindex(G, P)  # DeepDTA
            # cindex2 = get_ci(G, P)  # GraphDTA
            if best_mse < 0.1327:
                rm2 = get_rm2(G, P)  # DeepDTA
                pearson = get_pearson(G, P)
                spearman = get_spearman(G, P)
                rmse = get_rmse(G, P)
                result_file_name = 'Results/' + model_name+ '.txt'
                result_str = "in epoch" + str(best_epoch) + \
                             " best_mse:" + str(best_mse) + \
                             " best_pearson:" + str(pearson) + \
                             " best_rm2:" + str(rm2) + \
                             " best_spearman:" + str(spearman) + \
                             " best_rmse" + str(rmse)
                open(result_file_name, 'w').writelines(result_str)
                print('rmse improved at epoch ', best_epoch, '; best_test_mse', best_mse, model_st, dataset, fold)
            else:
                print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, model_st, dataset, fold)





