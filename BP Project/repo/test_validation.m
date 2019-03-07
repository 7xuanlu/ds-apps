clc;
close all;
clearvars;

%%%%%%%%% Loading training, validation and test features
load('features_train.mat');
load('features_valid.mat');
load('features_test.mat');

param_based_train = param_based_train';
param_based_valid = param_based_valid';
param_based_test = param_based_test';
whole_based_train = whole_based_train';
whole_based_valid = whole_based_valid';
whole_based_test = whole_based_test';

%%%%%%%%%%%%%%% Linear Regression


%%%%%% Learning/validation/testing with parameter based features
X = [ones(length(param_based_train),1) param_based_train(1:10,:)'];

X_valid = [ones(length(param_based_valid),1) param_based_valid(1:10,:)'];

X_test = [ones(length(param_based_test),1) param_based_test(1:10,:)'];

y = param_based_train(11,:)';

[b,bint] = regress(y,X);

y_model = X_valid * b;
y_valid = param_based_valid(11,:)';
mae_lr_vld_pb_SBP = sum(abs(y_model-y_valid))/length(y_valid);

std_lr_vld_pb_SBP = std(abs(y_model-y_valid));

y_model = X_test * b;
y_test = param_based_test(11,:)';

mae_lr_tst_pb_SBP = sum(abs(y_model-y_test))/length(y_test);

std_lr__tst_pb_SBP = std(abs(y_model-y_test));

y = param_based_train(12,:)';

[b,bint] = regress(y,X);

y_model = X_valid * b;
y_valid = param_based_valid(12,:)';
mae_lr_vld_pb_DBP = sum(abs(y_model-y_valid))/length(y_valid);

std_lr_vld_pb_DBP = std(abs(y_model-y_valid));

y_model = X_test * b;
y_test = param_based_test(12,:)';

mae_lr_tst_pb_DBP = sum(abs(y_model-y_test))/length(y_test);

std_lr_tst_pb_DBP = std(abs(y_model-y_test));


%%%%%%%%%%%%% Learning/Validation Testing with whole based features
X = [ones(length(whole_based_train),1) whole_based_train'];

X_valid = [ones(length(whole_based_valid),1) whole_based_valid'];

X_test = [ones(length(whole_based_test),1) whole_based_test'];

y = param_based_train(11,:)';

[b,bint] = regress(y,X);

y_model = X_valid * b;
y_valid = param_based_valid(11,:)';
mae_lr_vld_wb_SBP = sum(abs(y_model-y_valid))/length(y_valid);

std_lr_vld_wb_SBP = std(abs(y_model-y_valid));

y_model = X_test * b;
y_test = param_based_test(11,:)';

mae_lr_tst_wb_SBP = sum(abs(y_model-y_test))/length(y_test);

std_lr_tst_wb_SBP = std(abs(y_model-y_test));

y = param_based_train(12,:)';

[b,bint] = regress(y,X);

y_model = X_valid * b;
y_valid = param_based_valid(12,:)';
mae_lr_vld_wb_DBP = sum(abs(y_model-y_valid))/length(y_valid);

std_lr_vld_wb_DBP = std(abs(y_model-y_valid));

y_model = X_test * b;
y_test = param_based_test(12,:)';

mae_lr_tst_wb_DBP = sum(abs(y_model-y_test))/length(y_test);

std_lr_tst_wb_DBP = std(abs(y_model-y_test));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ensemble based learning

%%%%%%%%%%% Learning/Validation/Testing with parameter based features
X = param_based_train(1:10,:)';

X_valid = param_based_valid(1:10,:)';

X_test = param_based_test(1:10,:)';

y = param_based_train(11,:)';

%%%%%%%%%%%%%%%%%%% Random Forrest
Mdl = fitrensemble(X,y,'Method','Bag','NumLearningCycles',500);

y_model = predict(Mdl,X_valid);
y_valid = param_based_valid(11,:)';
mae_rf_vld_pb_SBP = sum(abs(y_model-y_valid))/length(y_valid);

std_rf_vld_pb_SBP = std(abs(y_model-y_valid));

y_model = predict(Mdl,X_test);
y_test = param_based_test(11,:)';

mae_rf_tst_pb_SBP = sum(abs(y_model-y_test))/length(y_test);

std_rf_tst_pb_SBP = std(abs(y_model-y_test));

y = param_based_train(12,:)';

Mdl = fitrensemble(X,y,'Method','Bag','NumLearningCycles',500);

y_model = predict(Mdl,X_valid);
y_valid = param_based_valid(12,:)';
mae_rf_vld_pb_DBP = sum(abs(y_model-y_valid))/length(y_valid);

std_rf_vld_pb_DBP = std(abs(y_model-y_valid));

y_model = predict(Mdl,X_test);
y_test = param_based_test(12,:)';

mae_rf_tst_pb_DBP = sum(abs(y_model-y_test))/length(y_test);

std_rf_tst_pb_DBP = std(abs(y_model-y_test));


%%%%%%%%%%% Learning/Validation/Testing with whole based features
X = whole_based_train';

X_valid = whole_based_valid';

X_test = whole_based_test';

y = param_based_train(11,:)';

Mdl = fitrensemble(X,y,'Method','Bag','NumLearningCycles',500);

y_model = predict(Mdl,X_valid);
y_valid = param_based_valid(11,:)';
mae_rf_vld_wb_SBP = sum(abs(y_model-y_valid))/length(y_valid);

std_rf_vld_wb_SBP = std(abs(y_model-y_valid));

y_model = predict(Mdl,X_test);
y_test = param_based_test(11,:)';

mae_rf_tst_wb_SBP = sum(abs(y_model-y_test))/length(y_test);

std_rf_tst_wb_SBP = std(abs(y_model-y_test));

y = param_based_train(12,:)';

Mdl = fitrensemble(X,y,'Method','Bag','NumLearningCycles',500);

y_model = predict(Mdl,X_valid);
y_valid = param_based_valid(12,:)';
mae_rf_vld_wb_DBP = sum(abs(y_model-y_valid))/length(y_valid);

std_rf_vld_wb_DBP = std(abs(y_model-y_valid));

y_model = predict(Mdl,X_test);
y_test = param_based_test(12,:)';

mae_rf_tst_wb_DBP = sum(abs(y_model-y_test))/length(y_test);

std_rf_tst_wb_DBP = std(abs(y_model-y_test));


%%%%%%%%%%%%%%%%%%%%LS Boost

%%%%%%%%%%% Learning/Validation/Testing with parameter based features
X = param_based_train(1:10,:)';

X_valid = param_based_valid(1:10,:)';

X_test = param_based_test(1:10,:)';

y = param_based_train(11,:)';

Mdl = fitrensemble(X,y,'Method','LSBoost','NumLearningCycles',500,'OptimizeHyperparameters','auto');

y_model = predict(Mdl,X_valid);
y_valid = param_based_valid(11,:)';
mae_lsb_vld_pb_SBP = sum(abs(y_model-y_valid))/length(y_valid);

std_lsb_vld_pb_SBP = std(abs(y_model-y_valid));

y_model = predict(Mdl,X_test);
y_test = param_based_test(11,:)';

mae_lsb_tst_pb_SBP = sum(abs(y_model-y_test))/length(y_test);

std_lsb_tst_pb_SBP = std(abs(y_model-y_test));

y = param_based_train(12,:)';

Mdl = fitrensemble(X,y,'Method','LSBoost','NumLearningCycles',500,'OptimizeHyperparameters','auto');

y_model = predict(Mdl,X_valid);
y_valid = param_based_valid(12,:)';
mae_lsb_vld_pb_DBP = sum(abs(y_model-y_valid))/length(y_valid);

std_lsb_vld_pb_DBP = std(abs(y_model-y_valid));

y_model = predict(Mdl,X_test);
y_test = param_based_test(12,:)';

mae_lsb_tst_pb_DBP = sum(abs(y_model-y_test))/length(y_test);

std_lsb_tst_pb_DBP = std(abs(y_model-y_test));

%%%%%%%%%% Learning/Validation/Testing with whole based features
X = whole_based_train';

X_valid = whole_based_valid';

X_test = whole_based_test';

y = param_based_train(11,:)';

Mdl = fitrensemble(X,y,'Method','LSBoost','NumLearningCycles',500,'OptimizeHyperparameters','auto');

y_model = predict(Mdl,X_valid);
y_valid = param_based_valid(11,:)';
mae_lsb_vld_wb_SBP = sum(abs(y_model-y_valid))/length(y_valid);

std_lsb_vld_wb_SBP = std(abs(y_model-y_valid));

y_model = predict(Mdl,X_test);
y_test = param_based_test(11,:)';

mae_lsb_tst_wb_SBP = sum(abs(y_model-y_test))/length(y_test);

std_lsb_tst_wb_SBP = std(abs(y_model-y_test));

y = param_based_train(12,:)';

Mdl = fitrensemble(X,y,'Method','LSBoost','NumLearningCycles',500,'OptimizeHyperparameters','auto');

y_model = predict(Mdl,X_valid);
y_valid = param_based_valid(12,:)';
mae_lsb_vld_wb_DBP = sum(abs(y_model-y_valid))/length(y_valid);

std_lsb_vld_wb_DBP = std(abs(y_model-y_valid));

y_model = predict(Mdl,X_test);
y_test = param_based_test(12,:)';

mae_lsb_tst_wb_DBP = sum(abs(y_model-y_test))/length(y_test);

std_lsb_tst_wb_DBP = std(abs(y_model-y_test));