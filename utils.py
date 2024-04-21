import torch

def norm_model_weights(model, model_type='llama'):
    last_q = None
    lqb = None
    lqkm = None
    last_v = None
    lvb = None
    lvom = None
    last_up = None
    bias = False
    for name, param in model.named_parameters():
        if "q_proj" in name:
            if "bias" in name:
                bias = True
                lqb = param
            else:
                last_q = param
        if "k_proj" in name:
            if "bias" in name:
                if lqkm is not None:
                    param.data = param.data * lqkm.flatten()
                    pass
            else:
                if model_type == 'llama':
                    # print(last_q.data.shape, param.data.shape)
                    mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=1, keepdim=True) / 
                                    torch.mean(torch.abs(param.data), dim=1, keepdim=True).repeat(
                                                                        int(last_q.data.shape[0] / param.data.shape[0]), 1))
                    # print(mult.shape, mult)
                    mult = torch.mean(mult)
                    # mult = np.sqrt(2.0)
                    last_q.data = last_q.data / mult#.transpose(0, 1)
                    # if bias: lqb.data = lqb.data / mult#.transpose(0, 1).flatten()
                    param.data = param.data * mult#[:param.data.shape[0]] # 
                    lqkm = mult
                    
                else:
                    # print(last_q.data.shape, param.data.shape)
                    # print(name)
                    # mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=1, keepdim=True) / 
                    #                 torch.mean(torch.abs(param.data), dim=1, keepdim=True)) # Loss: 0.14939117
                    mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=1, keepdim=True) / 
                                    torch.mean(torch.abs(param.data), dim=0, keepdim=True).transpose(0, 1)).transpose(0, 1) # Loss: 0.04619789
                    # mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=1, keepdim=True).transpose(0, 1) / 
                    #                 torch.mean(torch.abs(param.data), dim=0, keepdim=True)) # Loss: 0.04619789
                    # mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=0, keepdim=True).transpose(0, 1) / 
                    #                 torch.mean(torch.abs(param.data), dim=1, keepdim=True)) # Loss: 0.07823181
                    # mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=0, keepdim=True) / 
                    #                 torch.mean(torch.abs(param.data), dim=0, keepdim=True)).transpose(0, 1) # Loss: 0.01167393
                    # mult = mult / 2
                    # print(mult.shape, mult)
                    mult = torch.mean(mult)
                    last_q.data = last_q.data / mult#.transpose(0, 1)
                    if bias: lqb.data = lqb.data / mult.flatten() # .transpose(0, 1)
                    param.data = param.data * mult # 
                    lqkm = mult
                    # print(last_q.data.norm(), param.data.norm())
        
        if "v_proj" in name:
            if "bias" in name:
                lvb = param
            else:
                last_v = param
        if "o_proj" in name:
            if "bias" in name:
                param.data = param.data * lvom
            else:
                # print(last_v.data.shape, param.data.shape)
                # print(mult.shape, mult)
                if model_type == 'llama':
                    mult = torch.sqrt(torch.mean(torch.abs(last_v.data), dim=0, keepdim=True) / 
                            torch.mean(torch.abs(param.data), dim=0, keepdim=True))
                    mult = torch.mean(mult)
                    # mult = np.sqrt(1.0 / 2.0)
                    last_v.data = last_v.data / mult
                    if bias: lvb.data = lvb.data / mult
                    param.data = param.data * mult
                    lvom = mult
                else:
                    mult = torch.sqrt(torch.mean(torch.abs(last_v.data), dim=1, keepdim=True).transpose(0, 1).repeat(1, 
                                                                                int(param.data.shape[0] / last_v.data.shape[0])) / 
                                    torch.mean(torch.abs(param.data), dim=0, keepdim=True))
                    last_v.data = last_v.data / mult.transpose(0, 1)[:last_v.data.shape[0]]
                    if bias: lvb.data = lvb.data / mult.transpose(0, 1).flatten()
                    param.data = param.data * mult
                    lvom = mult

        if "up_proj" in name:
            last_up = param
        if "down_proj" in name:
            # print(last_up.data.shape, param.data.shape)
            mult = torch.sqrt(torch.mean(torch.abs(last_up.data), dim=1, keepdim=True).transpose(0, 1) / 
                            torch.mean(torch.abs(param.data), dim=0, keepdim=True))
            last_up.data = last_up.data / mult.transpose(0, 1)
            param.data = param.data * mult
            # print(mult, mult.shape)

        # if "model.norm.weight" == name:
        #     param.data /= 200
        # if "model.norm.bias" == name:
        #     param.data /= 200
        # if "lm_head.weight" == name:
        #     param.data *= 200
        
    return model