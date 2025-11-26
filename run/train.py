import datetime, os, random
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import DIAGNOSIS, EarlyStopping, RMSE, MSE, MAE, MAPE
from utils.loaddataset import DataSet
from model.net import STKrigingNet


def train(modelname, datapath, 
          batch_size=128, lr=0.0001, hidden_neurons=[4, 256, 16], weight=0.8, 
          loss_type='rmse', optim_type='adam', info=None,
          if_summary=False, if_save_model=False):
    
    # Test if GPU working or not
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(device)

    ##### read data #####
    # read the data
    data = pd.read_csv(datapath)
    print(data.describe())
    # print(data.head())

    ##### process data #####
    # Load data via class DataSet
    dataset = DataSet(data)
    data_train, data_test = dataset.get_data()

    # Scaling the data
    data_scaler = dataset.scaler_data()

    # get the scaled train data and  test data
    data_train_scaler = data_scaler['train']
    data_test_scaler = data_scaler['test']
    print('data_train_scaler:')
    print(data_train_scaler.describe())

    # Take the train set as known points (observed points)
    all_know_coods = data_train_scaler.values[:, 1:4]
    all_kown_features = data_train_scaler.values[:, 4:]
    
    # tramsform the data to tensor
    all_know_coods = torch.from_numpy((all_know_coods.astype(float))).to(torch.float32).to(device)
    all_kown_features = torch.from_numpy(all_kown_features.astype(float)).to(torch.float32).to(device)

    # get the dataloader
    if_split = True
    if if_split:
        # split the observed data into train set and validation set
        random.seed(42)
        val_index = random.sample(range(len(data_train_scaler)), int(len(data_train_scaler) * 0.2))  # 20% for validation
        val_data = data_train_scaler.values[val_index]
        train_data = np.delete(data_train_scaler.values, val_index, axis=0)
        train_dataloader = DataLoader(train_data.astype(float), shuffle=True, batch_size=batch_size, drop_last=False)
        val_dataloader = DataLoader(val_data.astype(float), shuffle=False, batch_size=batch_size, drop_last=False)
    else:
        # take the whole observed data as the train set
        train_dataloader = DataLoader(data_train_scaler.values.astype(float), shuffle=True, batch_size=batch_size, drop_last=False)
        val_dataloader = DataLoader(data_test_scaler.values.astype(float), shuffle=False, batch_size=batch_size, drop_last=False) 
    
    ##### STKrigingNet model initialization #####
    # Define the STKrigingNet model 
    d_input, d_model, d_trend = hidden_neurons
    model = STKrigingNet(d_input=d_input, d_model=d_model, d_trend=d_trend, known_num=len(all_know_coods), weight=weight, device=device)
        
    # Calculate position/time embeddings before training to increase the speed of training
    all_know_te, all_know_pe = model.get_pe(all_know_coods)
    model.to(device)

    ##### loss function ##### 
    if loss_type=='mae':
        criterion = MAE()
    elif loss_type=='mse':
        criterion = MSE()
    elif loss_type=='rmse':
        criterion = RMSE()
    elif loss_type=='mape':
        criterion = MAPE()

    ##### optimizer #####
    if optim_type=='adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif optim_type=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0)
        
    # learning rate decay 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8,  verbose=True, patience=5, min_lr=1e-6)  # lr_sche 1

    ##### Build a SummaryWriter base on TensorBoard #####
    # get the datafile name
    datafile = datapath[datapath.rfind('/')+1:]
    datafilename = datafile[0:datafile.rfind('.')]

    # get the save path
    start_time = datetime.datetime.now()
    train_info = start_time.strftime("%m%d_%H%M%S") + info['net_info']
    save_dir = './results/' +  modelname + '/' + datafilename + '/' + train_info

    # get the summary path and build the summarywriter
    if if_summary:
        summarypath = save_dir + '/summary/'
        writer = SummaryWriter(summarypath)
        print('summary path: \n' + summarypath)


    ##### early stopping #####
    patience = 20
    early_stopping = EarlyStopping(patience, verbose=True, delta=0.0005)

    ##### start train #####
    total_train_step = 0
    min_loss, best_epoch = float('inf'), float('inf')
    best_rmse_inverse,best_mae_inverse,best_r2_inverse = float('inf'),float('inf'),float('inf')

    print(train_info + '  start')
    epoch = 300
    for e in tqdm(range(epoch)):

        ##### training #####
        total_train_loss = 0
        train_num = 0
        model.train()  
        for step, i in enumerate(train_dataloader):
            
            # get the model input
            i = i.to(torch.float32)
            # input_pe = model.pe_know[i[:,0].type(torch.long)]
            input_coods = i[:, 1:4].to(device)
            input_features = i[:, 4:4+d_input].to(device)
            input_features[:,-1] = 0
            
            # model execution
            optimizer.zero_grad()
            input_te, input_pe = model.get_pe(input_coods)
            output, _ = model(input_coods, input_features, input_te, input_pe, all_know_coods, all_kown_features, all_know_te, all_know_pe)

            # calculate loss and optimize the model
            target = i[:, -1].to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # diagnose the output
            diag = DIAGNOSIS(output, target)
            l_rmse, l_mse, l_mae, l_mape = diag.get()
            
            # tensorboard summary
            if if_summary:
                total_train_step = total_train_step + 1
                writer.add_scalar("train_loss", loss, total_train_step)
                writer.add_scalar("train_rmse", l_rmse, total_train_step)
                writer.add_scalar("train_mae", l_mae, total_train_step)
                writer.add_scalar("train_mape", l_mape, total_train_step)
            
            # print the training log
            total_train_loss = total_train_loss + loss.item()
            train_num +=1
            if step % 10 == 0:
                print('\n--------train epoch {},step {},loss: {:.5f}, RMSE:{:.5f},MSE:{:.5f}, MAE:{:.5f}, MAPE:{:.5f}--'
                        .format(e, step, loss.item(), l_rmse, l_mse, l_mae, l_mape))
        
        # tensorboard summary
        total_train_loss = total_train_loss/train_num
        if if_summary:
            writer.add_scalar("total_train_loss", total_train_loss, e)

        ##### validation #####
        total_val_loss, l2p, l3p, l4p = 0, 0, 0, 0
        target_inverse_list, output_inverse_list = [], []
        val_num = 0
        with torch.no_grad():
            model.eval()  # set the model to evaluation mode
            for step, i in enumerate(val_dataloader):

                # get the model input
                i = i.to(torch.float32)
                input_coods = i[:, 1:4].to(device)
                input_features = i[:, 4:4+d_input].to(device)
                input_features[:,-1] = 0
                
                # model execution
                input_te, input_pe = model.get_pe(input_coods)
                output, _ = model(input_coods, input_features, input_te, input_pe, all_know_coods, all_kown_features, all_know_te, all_know_pe)                             
                
                # diagnose the output
                target = i[:, -1].to(device)
                val_loss = criterion(output, target)
                val_diag = DIAGNOSIS(output, target)
                t_rmse, t_mse, _, t_mape = val_diag.get()
                total_val_loss += val_loss.item()
                l2p += t_rmse
                l3p += t_mse
                l4p += t_mape
                
                # inverse the scaled output and calculate the diagnosis
                output_inverse = dataset.scaler_label.inverse_transform(output.cpu().clone().detach().numpy().reshape(-1, 1))
                target_inverse = dataset.scaler_label.inverse_transform(target.cpu().clone().detach().numpy().reshape(-1, 1))
                output_inverse_list.extend(output_inverse.reshape(-1))
                target_inverse_list.extend(target_inverse.reshape(-1))

                val_num = val_num + 1

            # calculate the final result
            final_loss = total_val_loss / val_num
            final_rmse = l2p / val_num
            final_mse = l3p / val_num
            final_mape = l4p / val_num
            val_diag_inverse = DIAGNOSIS(np.array(output_inverse_list).reshape(-1, 1), np.array(target_inverse_list).reshape(-1, 1))
            final_rmse_inverse = val_diag_inverse.v_rmse
            final_mae_inverse = val_diag_inverse.v_mae
            final_r2_inverse = val_diag_inverse.v_r2

            # adjust learning rate 
            scheduler.step(final_loss)

            # tensorboard summary
            if if_summary:
                writer.add_scalar("total_val_loss", final_loss, e)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], e)

            # print the log
            end_time = datetime.datetime.now()
            print('--------val loss: {:.5f}, RMSE:{:.5f},MSE:{:.5f}, MAPE:{:.5f}, RMSE_inv:{:.5f}, epoch time:{:.2f}m--'
                .format(final_loss, final_rmse, final_mse, final_mape, final_rmse_inverse, (end_time-start_time).seconds/60))

            # save the best model parameters and info
            if final_loss < min_loss:
                min_loss = final_loss
                best_epoch = e
                best_rmse_inverse = final_rmse_inverse
                best_mae_inverse = final_mae_inverse
                best_r2_inverse = final_r2_inverse

                if if_summary:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # save the model parameters
                    if if_save_model:
                        torch.save(model.state_dict(), save_dir + '/checkpoint.pth')
                    
                    # save the model info
                    with open(save_dir + '/checkpoint_info.txt', 'w') as f:
                        f.write('-----------model_info-------------')
                        f.write('\rmodel name:  STKrigingNet\r')
                        f.write('train info:  {}\r'.format(train_info))
                        f.write('min loss:  {}\r'.format(min_loss))
                        f.write('batch_size: {}\rlr: {}\rhidden_neuron: {}\rweight: {}\r'
                                .format(batch_size, lr, hidden_neurons, weight))
                        f.write('best_epoch:  {}\r'.format(best_epoch))
                        f.write('best_rmse_inverse:  {}\r'.format(best_rmse_inverse))
                        f.write('best inverse MAE/RMSE/R2: {:.3f}/{:.3f}/{:.3f}\r'
                            .format(best_mae_inverse,best_rmse_inverse, best_r2_inverse))
                        f.write('spend time: {} minutes\r'.format((datetime.datetime.now()-start_time).seconds/60))
                        f.write('epoch: {}\r'.format(e))

            ##### early stopping #####
            early_stopping(final_loss, model, save_model_dir=None)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # close the tensorboard summary
    if if_summary:
        writer.close()

    print(train_info + '  end')

    # print the results
    print('******** {} *********'.format(datafilename))
    print('final min loss:', min_loss)
    print('best_epoch:', best_epoch)
    print('best inverse MAE/RMSE/MAPE/R2: {:.3f}/{:.3f}/{:.3f}'
        .format(best_mae_inverse,best_rmse_inverse, best_r2_inverse))
    end_time = datetime.datetime.now()
    print('spend time: {:3f} minutes'.format((end_time-start_time).seconds/60))

    best_inverse = [best_mae_inverse, best_rmse_inverse, best_r2_inverse]
    
    return train_info, min_loss, best_epoch, best_inverse