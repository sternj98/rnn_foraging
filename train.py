import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def opt_s_rmse(output_dict,example_output,opt_s):
    mse_loss = nn.MSELoss()
    return torch.sqrt(mse_loss(output_dict['readout_output'][:,:,-1],torch.from_numpy(opt_s)[:,:,-1]))

def zero_loss(output_dict,example_output,opt_s):
    return torch.tensor([0.])

def gradClamp(parameters, clip=100):
    """
        Element-wise gradient clipping
    """
    for p in parameters:
        if (p.requires_grad == True and p.grad != None):
            p.grad.data.clamp_(-clip,clip)

def train(tr_cond,n_batches,task,model,param_l2_penalty,lr,grad_clip,optimizer,print_every,performance_loss_fn,regularization_loss_fn):
    # build optimizer and loss function (this can be functionalized)
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),lr = lr,weight_decay = param_l2_penalty)

    if performance_loss_fn == "opt_s_rmse":
        performance_loss_fn = opt_s_rmse

    if regularization_loss_fn == "zero_loss":
        regularization_loss_fn = zero_loss
    total_losses = np.zeros(n_batches)
    performance_losses = np.zeros(n_batches)

    # TRAINING
    s_vec, opt_s_vec, ex_pred_vec, frac_rmse_vec = [], [], [], []
    for i, (example_input, example_output, opt_s , opt_s_sigma2) in task: # loop over batches in training set
        optimizer.zero_grad() # zero the gradients
        output_dict = model.forward(example_input) # perform forward pass

        # calculate loss
        performance_loss = performance_loss_fn(output_dict,example_output,opt_s)
        regularization_loss = regularization_loss_fn(output_dict,example_output,opt_s)

        loss = (performance_loss + regularization_loss).double()
        loss.backward() # backpropagate gradients
        if (model.model_kwargs['reservoir_training'] == True and i == model.model_kwargs['reservoir_training_params']['weight_lock_step']):
            print("Turning off recurrent weight training")
            for i_param in model.core.parameters():
                i_param.requires_grad = False
        gradClamp(model.parameters(), clip = grad_clip) # clip gradients
        optimizer.step() # perform optimization step

        total_losses[i] = loss.item()
        performance_losses[i] = performance_loss.item()

        s_vec.append(example_output[:,:,-1].detach().numpy())
        opt_s_vec.append(opt_s[:,:,-1])
        ex_pred_vec.append(output_dict['readout_output'].detach().numpy()[:,:,-1])

        if i % print_every == 0:
            rmse_opt  = np.sqrt(np.nanmean((np.asarray(s_vec) - np.asarray(opt_s_vec))**2))
            rmse_net  = np.sqrt(np.nanmean((np.asarray(s_vec) - np.squeeze(np.asarray(ex_pred_vec)))**2))
            frac_rmse = (rmse_net - rmse_opt) / rmse_opt
            frac_rmse_vec.append(frac_rmse)
            if i > 0:
                print('Batch #%d; Performance Loss: %.3f; Frac. RMSE: %.6f; Opt. RMSE: %.6f; Net. RMSE: %.6f' % (i, np.mean(performance_losses[i-print_every:i]),frac_rmse, rmse_opt, rmse_net))
            else:
                print('Batch #%d; Performance Loss: %.3f; Frac. RMSE: %.6f; Opt. RMSE: %.6f; Net. RMSE: %.6f' % (i, np.mean(performance_losses[i]),frac_rmse, rmse_opt, rmse_net))
            s_vec       = []
            opt_s_vec   = []
            ex_pred_vec = []

    return model,total_losses,performance_losses
