import datetime
import pytz
import os

import torch

# from our files
from models import RecurrentModel

def save_model(model,train_params):
    """
        Save model and training params to .pt file
    """
    # make outer path with date
    date = (datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%m_%d"))
    outerPath = '..' + '/runs/' + str(date) + '/'
    if not os.path.exists(outerPath):
        os.makedirs(outerPath)

    core_kwargs = model.model_kwargs['core_kwargs']
    param_init_kwargs = model.model_kwargs['param_init_kwargs']

    # ready train_params for saving
    train_params_save = train_params.copy()
    point_process_type = "stochastic_foraging_session"
    task = train_params['task']
    y_generator_fn_name = task.y_generator.__name__

    if y_generator_fn_name == 'generate_session_y':
        generation_fn = "stochastic_foraging_session"
    elif y_generator_fn_name == 'generate_from_rewtimes_y':
        generation_fn = 'rewtimes_foraging_session'
    elif y_generator_fn_name == 'srw_point_process':
        generation_fn = 'srw_point_process'

    train_params_save['generation_fn'] = generation_fn
    train_params_save['model'] = []
    train_params_save['task'] = []

    included_params = [
        'offdiag=' + str(np.sqrt(core_kwargs['hidden_size']) * param_init_kwargs['offdiag_val']),
        'diag=' + str(param_init_kwargs['diag_val']),
        'sigma2eps=' + str(task.sigma2_eps)]

    currentTime = (datetime.datetime.now(pytz.timezone('America/Toronto')).strftime("%H_%M"))

    separator = '_'
    run_id = separator.join(str(ip) for ip in included_params)
    savepath = outerPath + currentTime + separator + run_id + '.pt'

    # package up
    save_dict = dict(
        model_kwargs=model.model_kwargs,
        model_state_dict=model.state_dict(),
        train_params=train_params_save)
    # and save
    torch.save(
        obj=save_dict,
        f=savepath)

    print('\tSaved model!')

def load_model(run_path):
    """
        Load model and training parameters from a directory
    """
    save_dict = torch.load(run_path)
    saved_model_kwargs = save_dict['model_kwargs']
    saved_model = RecurrentModel(saved_model_kwargs)
    saved_model.load_state_dict(save_dict['model_state_dict'])
    train_params = save_dict['train_params']
    return saved_model,train_params
