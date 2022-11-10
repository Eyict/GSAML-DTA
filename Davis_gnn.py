import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
import torch.nn.functional as f

class GNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GNNNet, self).__init__()

        print('GNNNet Loaded')

        self.n_output = n_output
       
        self.enc_mean = nn.Linear(output_dim*8, output_dim*8)
        self.enc_std = nn.Linear(output_dim*8, output_dim*8)
       
        self.mol_W1 = nn.Parameter(torch.zeros(size=(num_features_mol*3, num_features_mol*3)))
        self.mol_W2 = nn.Parameter(torch.zeros(size=(num_features_mol*9, num_features_mol*9)))
        self.mol_W3 = nn.Parameter(torch.zeros(size=(output_dim, output_dim)))
        nn.init.xavier_uniform_(self.mol_W1.data, gain=1.414)  
        nn.init.xavier_uniform_(self.mol_W2.data, gain=1.414)  
        nn.init.xavier_uniform_(self.mol_W3.data, gain=1.414)  
        self.pro_W1 = nn.Parameter(torch.zeros(size=(num_features_pro * 3, num_features_pro * 3)))
        self.pro_W2 = nn.Parameter(torch.zeros(size=(num_features_pro * 12, num_features_pro * 12)))
        self.pro_W3 = nn.Parameter(torch.zeros(size=(output_dim, output_dim)))
        nn.init.xavier_uniform_(self.pro_W1.data, gain=1.414)  
        nn.init.xavier_uniform_(self.pro_W2.data, gain=1.414) 
        nn.init.xavier_uniform_(self.pro_W3.data, gain=1.414)  
        
        self.n_output = n_output
        self.conv1 = GATConv(num_features_mol, num_features_mol, heads=3)

        self.conv3 = GCNConv(num_features_mol * 3, output_dim)
        self.fc_g1 = torch.nn.Linear(num_features_mol * 3+output_dim, 2048)
        self.fc_g2 = torch.nn.Linear(2048, output_dim*4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
       
        self.n_output = n_output
        self.pro_conv1 = GATConv(num_features_pro, num_features_pro, heads=3)
        # self.pro_conv2 = GATConv(num_features_pro * 3, num_features_pro * 3, heads=4)
        self.pro_conv3 = GCNConv(num_features_pro * 3, output_dim)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 3 + output_dim, 2048)
        self.pro_fc_g2 = torch.nn.Linear(2048, output_dim*4)


        # combined layers
        self.normal = nn.BatchNorm1d(output_dim*8)
        self.fc1 = nn.Linear(output_dim*8, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.out = nn.Linear(512, self.n_output)

    # return attention_weight_matrix
    def forward(self, data_mol, data_pro):
       
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch
       
        x1 = self.conv1(mol_x, mol_edge_index)
        x1 = self.relu(x1)
        # x2 = self.conv2(x1, mol_edge_index)
        # x2 = self.relu(x2)
        x3 = self.conv3(x1, mol_edge_index)
        x3 = self.relu(x3)
      
        x1 = torch.mm(x1, self.mol_W1)
        x3 = torch.mm(x3, self.mol_W3)
        # TODO end
        x = torch.concat([x1,x3],dim=1)
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = gep(x, mol_batch)  # global pooling
       
        xt1 = self.pro_conv1(target_x, target_edge_index)
        xt1 = self.relu(xt1)
        # xt2 = self.pro_conv2(xt1, target_edge_index)
        # xt2 = self.relu(xt2)
        xt3 = self.pro_conv3(xt1, target_edge_index)
        xt3 = self.relu(xt3)
      
        xt1 = torch.mm(xt1, self.pro_W1)
        xt3 = torch.mm(xt3, self.pro_W3)
        #TODO end
        xt = torch.concat([xt1,xt3],dim=1)
        xt = self.pro_fc_g1(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = gep(xt, target_batch)  # global pooling
       
        xc = torch.cat((x,xt), 1)
        
        enc_mean, enc_std = self.enc_mean(xc), f.softplus(self.enc_std(xc) - 5)
        eps = torch.randn_like(enc_std)
        latent = enc_mean + enc_std * eps
  
        xc = self.fc1(latent)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out,enc_mean,enc_std,latent

