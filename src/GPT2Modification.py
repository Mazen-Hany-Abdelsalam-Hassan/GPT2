from GPT2 import GPTModel , GPT_CONFIG_124M
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self ,model:GPTModel, out_units:int =1 , num_block2train:int = 2):
        super().__init__()
        #assert num_block2train > GPT_CONFIG_124M["n_layers"]
        self.model = model
        self.out_units=out_units
        self.num_block2train = num_block2train
        self._replace_heads()
        self._freez_except()
    def forward(self, x):
        return self.model(x)

    def _replace_heads(self):
        self.model.out_head = nn.Linear(GPT_CONFIG_124M['emb_dim'] , self.out_units)

    def _freez_except(self):
        ## Freeze all the model weights
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        ## unfreeze the classification head and the (weights and biases)  layer
        for parameter in self.model.out_head.parameters():
            parameter.requires_grad =True

        ## unfreeze the last layer normalization
        for parameter in self.model.final_norm.parameters():
            parameter.requires_grad = True

        ## unfreeze the self.num_block2train
        for i in range(1,self.num_block2train+1):

            for parameter in self.model.trf_blocks[-i].parameters():
                parameter.requires_grad = True
    def return_model(self):
        return  self.model
