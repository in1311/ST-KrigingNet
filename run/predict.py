import datetime, os 
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import DIAGNOSIS
from utils.loaddataset import DataSet
from model.net import STKrigingNet

def predict(modelname, datapath, train_info, hidden_neurons, weight=0.7, if_save_result=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ##### read the sampled dataset #####
    # read the sampled dataset
    sampledata = pd.read_csv(datapath)

    # get the name of sampled dataset
    datafile = datapath[datapath.rfind('/')+1:]
    datafilename = datafile[0:datafile.rfind('.')]

    ###### preprocess the sampled dataset #####
    # Scaling the data
    dataset = DataSet(sampledata)
    data_scaler = dataset.scaler_data()

    # Take the train set as known points (observed locations) and transform to tensor
    data_train_scaler = data_scaler['train']
    coods_know_norm = torch.from_numpy((data_train_scaler.values[:, 1:4].astype(float))).to(torch.float32).to(device)
    all_kown_features = torch.from_numpy(data_train_scaler.values[:, 4:].astype(float)).to(torch.float32).to(device)

    # Read the test point data
    data_train, data_test = dataset.get_data()
    if data_test.empty is not True:
        print('predict test')
        data_pre = data_test
    data_pre_scaler = data_scaler['test']
    datapre_dataloader = DataLoader(data_pre_scaler.values, shuffle=False, batch_size=256, drop_last=False)
    coods_unknow = data_pre.values[:, 1:4]
    coods_know = data_train.values[:, 1:4]

    ##### Model initialization #####
    # Define the STKrigingNet model 
    d_input, d_model, d_trend = hidden_neurons
    model = STKrigingNet(d_input=d_input, d_model=d_model, d_trend=d_trend, known_num=len(coods_know_norm), weight=weight, device=device)
    
    # load the best model parameters
    result_dir = './results/'+ modelname + '/' + datafilename + '/'  + train_info + '/'
    model.load_state_dict(torch.load(result_dir + 'checkpoint.pth',map_location=torch.device(device)))
    
    # Calculate position/time embedding before training to increase the speed of training
    all_know_te, all_know_pe = model.get_pe(coods_know_norm)
    model.to(device)

    ##### predict #####
    result = []
    trend_unknow_pre= []
    with torch.no_grad():
        model.eval()  # set the model to evaluation mode
        for i in tqdm(datapre_dataloader):
            # model input
            i = i.to(torch.float32)
            input_coods = i[:, 1:4].to(device)
            input_features = i[:, 4:4+d_input].to(device)
            input_features[:,-1] = 0
            input_te, input_pe = model.get_pe(input_coods)

            # model execution
            output, [cov_know, trend_know_pre, trend_unknow_pre_] = model(input_coods, input_features, input_te, input_pe, coods_know_norm, all_kown_features, all_know_te, all_know_pe)
            result.extend(output.cpu().detach().numpy())
            trend_unknow_pre.extend(trend_unknow_pre_.cpu().detach().numpy())
    cov_know = cov_know.cpu().detach().numpy()
    trend_know_pre = trend_know_pre.cpu().detach().numpy()
    trend_unknow_pre = np.array(trend_unknow_pre)
    dict = {}
    dict['cov_know'] = cov_know
    dict['coods_know'] = coods_know
    dict['coods_know_norm'] = coods_know_norm
    dict['coods_unknow'] = coods_unknow
    dict['trend_know_pre'] = trend_know_pre
    dict['trend_unknow_pre'] = trend_unknow_pre
    np.savez(result_dir + '/dict.npz', save_dic=dict)
    
    # reverse the output
    result_inverse = dataset.scaler_label.inverse_transform(np.array(result).reshape(-1,1))
    
    # diagnose the reversed output
    diag_inverse = DIAGNOSIS(result_inverse, np.array(data_pre['z']).reshape(-1,1))
    test_rmse_inverse, test_mse_inverse, test_mae_inverse, test_mape_inverse = diag_inverse.get()

    # print diagnostic results
    print('mae:{:.3f}, rmse:{:.3f}, R2:{:.3f}, mse:{:.3f}, mape:{:.3f}%'.format(test_mae_inverse, test_rmse_inverse, diag_inverse.v_r2, test_mse_inverse, test_mape_inverse*100))
    
    ##### save the predict result and diagnostic results #####
    if if_save_result is True:
        # save the predict result
        data_pre['predict'] = np.array(result_inverse)
        data_pre.to_csv(result_dir + 'predict_result.csv', index=False)

        with open(result_dir + 'result_diag.txt', 'w') as f:
            f.write('-----------Diagnosis-------------')
            f.write('\r MAE/RMSE/R2: {:.3f}/{:.3f}/{:.3f}'.format(test_mae_inverse, test_rmse_inverse, diag_inverse.v_r2))
    
    # os.remove(result_dir + 'checkpoint.pth')
    return [test_rmse_inverse, test_mae_inverse, diag_inverse.v_r2], result_dir + '/predict_result.csv'
