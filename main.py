from run.train import train
from run.predict import predict
import os, torch, numpy as np

def set_seed(seed):
    ##### Set random seed for reproducibility #####
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    ##### hyperparameter #####
    dataset = 'temperature'  # 'temperature', 'airquality'

    if dataset == 'temperature':
        datafile ='gsod_NMG_20180101_0731_rm60%_s10.csv'
        datapath = './Data/Temp/dataset/' + datafile
        lr = 0.0005  # learning rate
        hidden_neuron = [4, 128, 16]  # [input dimension, model dimension, trend dimension]
        weight = 0.5  # weight of position embedding and time embedding 

    elif dataset == 'airquality':
        datafile ='AQ_BJ_20211101_1107_rm60%_s10.csv'
        datapath = './Data/AirQuality/dataset/' + datafile
        lr = 0.0001
        hidden_neuron = [6, 256, 32]
        weight = 0.6

    batch_size = 128  # batch size
    loss_type = 'rmse'  # loss function type: 'rmse', 'mae', 'mse', 'mape'
    optim_type='adam'  # optimizer type: 'adam', 'sgd'
    if_summary = True  # if save the training summary or not
    if_save_model = True  # if save the best model or not
    info = {'net_info': 'default'}  # extra information for saving folder naming only

    ##### train #####
    modelname = 'ST-KrigingNet'  
    set_seed(42)  # set random seed for reproducibility
    train_info, min_loss, best_epoch, best_inverse = train(modelname, datapath, batch_size, lr, hidden_neuron, 
                                                            weight, loss_type=loss_type, optim_type=optim_type, info=info,
                                                            if_summary=if_summary, if_save_model=if_save_model) 
    with open('./results/train_log.txt', 'a', encoding='utf-8') as f:
        f.write('\rtrain_info: {} /**/ datafile: {}\rmin_loss({}): {:.5f}; best_epoch: {:.5f}; best_rmse_inverse: {:.5f}; best_inverse MAE/RMSE/R2: {:.3f}/{:.3f}/{:.3f} '\
        '/**/ hidden_neurons: {}; weight: {}; model: {}; batch_size: {}; lr: {}; optim_type: {}\r'
                .format(train_info, datafile, loss_type, min_loss, best_epoch, best_inverse[1], best_inverse[0], best_inverse[1], best_inverse[2],
                        hidden_neuron, weight, modelname, batch_size, lr, optim_type))
    
    ##### predict #####
    if if_save_model is True:
        #  load the best model and predict
        result_diag, result_path = predict(modelname, datapath, train_info, hidden_neuron, weight=weight, if_save_result=True)
