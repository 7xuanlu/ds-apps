clc;
close all;
clear all;
%%%%%%%% Loading dataset
load Part_1.mat;

%%%%% Defining the size of whole based and parameter based features
whole_based_crs = single(zeros(2000,1000));
param_based_crs = single(zeros(12,1000));
%%
samplingrate = single(125);
del_val = single(-125);

%%%% Start and End index to decide training, validation and testing
%%%% dataset. Training:- 1:1000; Validation:- 1001:2000; Testing:-
%%%% 2001:3000
istart = 1;
iend = 1000;
%%% Some records are noisy and such noise is not removed using
%%% preprocessing. So, the algorithm might show some error while running.
%%% In that case, the for loop has to be run from the error index again or
%%% after skipping the record which shows the error.
for i = istart:iend
    i
    rng('shuffle');
    %%%%% Selecting the ecg, ppg and abp parts of the record
    ecg_orig = Part_1{1,i}(3,:);
    ppg_orig = Part_1{1,i}(1,:);
    abp_orig = Part_1{1,i}(2,:);
    
    %%%%% Detrending and normalizing the signals
    ecg_temp = ecg_orig-mean(ecg_orig);
    ppg_temp = ppg_orig-mean(ppg_orig);
    abp_mean = mean(abp_orig);
    abp_temp = abp_orig-abp_mean;
    abp_min = min(abp_temp);
    abp_max = max(abp_temp);
    ecg = (ecg_temp-min(ecg_temp))/(max(ecg_temp)-min(ecg_temp));
    ppg = (ppg_temp-min(ppg_temp))/(max(ppg_temp)-min(ppg_temp));
    abp = (abp_temp-abp_min)/(abp_max-abp_min);
    
    %%%%%%%% Uncomment block if noisy parts in the signals have been
    %%%%%%%% identified and their indexes known.
    %{
    cd ../../DATA/
    filnam = sprintf('s%d.txt',i);
    filnam2 = sprintf('S%d.txt',i);
    if isfile(filnam)
        fid = fopen(filnam,'r');
        Q = textscan(fid,'%d %d');
        fclose(fid);
        for j=1:length(Q{1})
            ecg(Q{1,1}(j):Q{1,2}(j)) = del_val;
            ppg(Q{1,1}(j):Q{1,2}(j)) = del_val;
            abp(Q{1,1}(j):Q{1,2}(j)) = del_val;
        end
    elseif isfile(filnam2)
        fid = fopen(filnam2,'r');
        Q = textscan(fid,'%d %d');
        fclose(fid);
        for j=1:length(Q{1})
            ecg(Q{1,1}(j):Q{1,2}(j)) = del_val;
            ppg(Q{1,1}(j):Q{1,2}(j)) = del_val;
            abp(Q{1,1}(j):Q{1,2}(j)) = del_val;
        end
    else
        
    end
    ecg(ecg==del_val) = [];
    ppg(ppg==del_val) = [];
    abp(abp==del_val) = [];
    cd ../Second_Report/'MLab Code'
    %}
    
    %%%%%%%% Indexing to be used if features from 10 random peaks are to be
    %%%%%%%% collected separately
    %{
    idxi = 1+10*(i-1-(istart-1));
    idxf = 10*(i-(istart-1));
    i_c = i-(istart-1);
    %}
    
    %%%%%%%%%% Preprocessing signals
    ecg_f = preprocessing(ecg);
    abp_f = preprocessing(abp);
    ppg_f = preprocessing(ppg);
    
    %%%%%%%%%%% Extracting median features from each record
    [p_b_cr,w_b_cr] = feature_extraction(ecg_f,ppg_f,abp_f);
    param_based_crs(:,i_c) = p_b_cr;
    whole_based_crs(1:length(w_b_cr),i_c) = w_b_cr;
    param_based_crs(11,i_c) = param_based_crs(11,i_c)*(abp_max-abp_min) + abp_min + abp_mean;
    param_based_crs(12,i_c) = param_based_crs(12,i_c)*(abp_max-abp_min) + abp_min + abp_mean;
    %%%%%%%% features from 10 random peaks if collected separately 
    %param_based_dt(:,idxi:idxf) = p_b_dt;
    %whole_based_dt(1:length(w_b_dt),idxi:idxf) = w_b_dt;
    %param_based_dt(11,idxi:idxf) = param_based_dt(11,idxi:idxf)*(abp_max-abp_min) + abp_min + abp_mean;
    %param_based_dt(12,idxi:idxf) = param_based_dt(12,idxi:idxf)*(abp_max-abp_min) + abp_min + abp_mean;
end

%%%%%%% finding records with error, so they did not collect any features.
%%%%%%% Removing those feature columns
n_ex = find(param_based_crs(11,:)==0);
param_based_crs(:,n_ex)=[];
whole_based_crs(:,n_ex)=[];
%hist(param_based_crs(11,:),length(param_based_crs));
n_ex = find(param_based_crs(11,:)< 80 | param_based_crs(11,:) > 180 | param_based_crs(12,:) > 110 | param_based_crs(12,:) < 50);
param_based_crs(:,n_ex)=[];
whole_based_crs(:,n_ex)=[];
%hist(param_based_crs(11,:),length(param_based_crs));
whole_based_pc = pca(whole_based_crs,'NumComponents',15);
%%
%%%%%%%%% Code to be used for training features
%%{
param_based_train = param_based_crs';
whole_based_train = whole_based_pc;
save('features_train.mat','param_based_train','whole_based_train');
%}
%%
%%%%%%%%% Code to be used for validation features
%%{
param_based_valid = param_based_crs';
whole_based_valid = whole_based_pc;
save('features_valid.mat','param_based_valid','whole_based_valid');
%}
%%
%%%%%%%%% Code to be used for test features
%%{
param_based_test = param_based_crs';
whole_based_test = whole_based_pc;
save('features_test.mat','param_based_test','whole_based_test');
%}