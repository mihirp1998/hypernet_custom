import torch
import torch.nn.functional as F
import pickle
import torch.nn as nn
import hyperparams as hyp
from torch.nn.parameter import Parameter
import ipdb 
st = ipdb.set_trace


class HyperNet(nn.Module):
    def __init__(self,layer_size):
        super(HyperNet, self).__init__()
        self.layer_size = layer_size
        self.emb_dimension = hyp.hypernet_input_size
        total_instances = hyp.total_instances
        self.embedding = nn.Embedding(total_instances, self.emb_dimension)
        range_val = 1
        variance = ((2*range_val))/(12**0.5)
        nn.init.uniform_(self.embedding.weight,a=-range_val,b=range_val)
        self.weight_names = ['net.conv3d.0.0', 'net.conv3d.1.0', 'net.conv3d.2.0', 'net.conv3d.3.0', 'net.conv3d.4.0', 'net.conv3d_transpose.0.0', 'net.conv3d_transpose.1.0', 'net.conv3d_transpose.2.0', 'net.conv3d_transpose.3.0', 'net.final_feature']
        
        if hyp.hypernet_nonlinear:
            self.hidden1 = nn.Linear(16, 32)
            self.hidden2 = nn.Linear(32, 16)


        bias_variances = [5e-2, 2e-2, 1.4e-2, 9.8e-3, 7e-3, 4.8e-3, 5.2e-3, 7e-3, 9.5e-3, 6.5e-2]
        weight_variances = [5.5e-2, 2e-2, 1.4e-2, 9.8e-3, 7e-3, 4.8e-3, 5.5e-3, 7e-3, 9.5e-3, 6.5e-2]

        if hyp.hardcode_variance or True:
            self.kernel_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=(variance*weight_variances[index])/((variance**2-weight_variances[index]**2)**0.5)),requires_grad=True) for index,i in enumerate(self.layer_size)])
            self.bias_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, i[0]), mean=0, std=(variance*bias_variances[index])/((variance**2-bias_variances[index]**2)**0.5)),requires_grad=True) for index,i in enumerate(self.layer_size)])        
        else:
            self.kernel_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=1000/(i[1]*i[2]*i[3]*i[4]*initrange*self.emb_dimension)),requires_grad=True) for i in self.layer_size])
            self.bias_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, i[0]), mean=0, std=1000/(i[1]*i[2]*i[3]*i[4]*initrange*self.emb_dimension)),requires_grad=True) for i in self.layer_size])
        print("initalized")
        # self.bias_encoderBias = nn.ParameterList([Parameter(torch.nn.init.zeros_(torch.empty(self.total(i))),requires_grad=True) for i in self.layer_size])


    def total(self,tensor_shape):
        return tensor_shape[0]*tensor_shape[1]*tensor_shape[2]*tensor_shape[3]*tensor_shape[3]

    def forward(self,indices,summ_writer):
        # st()
        # contextEmbed = self.context_embeddings_100(id_num)
        embed = self.embedding(indices)
        if hyp.hypernet_nonlinear:
            embed = self.hidden1(embed)
            embed = F.leaky_relu(embed)
            embed = self.hidden2(embed)
            embed = F.leaky_relu(embed)
            
        if hyp.hypernet_bias:
            pass
        else:
            feat_kernels = [(torch.matmul(embed,self.kernel_encoderWeights[i])).view([hyp.B]+self.layer_size[i]) for i in range(len(self.kernel_encoderWeights))]
            feat_Bias = [(torch.matmul(embed,self.bias_encoderWeights[i])).view([hyp.B]+self.layer_size[i][0:1]) for i in range(len(self.bias_encoderWeights))]
        if hyp.vis_feat_weights:
            names = pickle.load(open('names.p',"rb"))
            for i in range(10):
                weight_name = self.weight_names[i]+".weight"
                bias_name = self.weight_names[i]+".bias"
                summ_writer.summ_histogram(weight_name, feat_kernels[i].clone().cpu().data.numpy())
                summ_writer.summ_histogram(bias_name, feat_Bias[i].clone().cpu().data.numpy())
        return [feat_kernels,feat_Bias]
 
 	def summ_histogram(self, name, data):
		if self.save_this:
			data = data.flatten() 
			self.writer.add_histogram(name, data, global_step=self.global_step)


