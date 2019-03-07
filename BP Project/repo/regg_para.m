%% Parameter-based Regression
clc;
close all;
clear all;
%% Training
load('features_train.mat');
df_train = param_based_train;
sbp = df_train(:, end-1);
dbp = df_train(:, end);
df_train = df_train(:, 1: end-2);

% Parameter based training
tree_para_sbp = fitrtree(df_train, sbp, "OptimizeHyperparameters", "auto");
tree_para_dbp = fitrtree(df_train, dbp, "OptimizeHyperparameters", "auto");
svm_para_sbp = fitrsvm(df_train, sbp, "OptimizeHyperparameters", "auto");
svm_para_dbp = fitrsvm(df_train, dbp, "OptimizeHyperparameters", "auto");

%% Validation
load('features_valid.mat');
df_valid = param_based_valid;
sbp = df_valid(:, end-1);
dbp = df_valid(:, end);
df_valid = df_valid(:, 1: end-2);

% Evaluation on decision tree
mae_valid_sbp_tree = sum(abs(predict(tree_para_sbp, df_valid)-sbp)) ...
    /size(df_valid, 1);
std_valid_sbp_tree = std(abs(predict(tree_para_sbp, df_valid)-sbp));
mae_valid_dbp_tree = sum(abs(predict(tree_para_dbp, df_valid)-dbp)) ...
    /size(df_valid, 1);
std_valid_dbp_tree = std(abs(predict(tree_para_dbp, df_valid)-dbp));

% Evaluation on svm
mae_valid_sbp_svm = sum(abs(predict(svm_para_sbp, df_valid)-sbp)) ...
    /size(df_valid, 1);
std_valid_sbp_svm = std(abs(predict(svm_para_sbp, df_valid)-sbp));
mae_valid_dbp_svm = sum(abs(predict(svm_para_dbp, df_valid)-dbp)) ...
    /size(df_valid, 1);
std_valid_dbp_svm = std(abs(predict(svm_para_dbp, df_valid)-dbp));
%% Test
load('features_test.mat');
df_test = param_based_test;
sbp = df_test(:, end-1);
dbp = df_test(:, end);
df_test = df_test(:, 1: end-2);

% Evaluation on decision tree
mae_test_sbp_tree = sum(abs(predict(tree_para_sbp, df_test)-sbp)) ...
    /size(df_test, 1);
std_test_sbp_tree = std(abs(predict(tree_para_sbp, df_test)-sbp));
mae_test_dbp_tree = sum(abs(predict(tree_para_dbp, df_test)-dbp)) ...
    /size(df_test, 1);
std_test_dbp_tree = std(abs(predict(tree_para_dbp, df_test)-dbp));

% Evaluation on svm
mae_test_sbp_svm = sum(abs(predict(svm_para_sbp, df_test)-sbp)) ...
    /size(df_test, 1);
std_test_sbp_svm = std(abs(predict(svm_para_sbp, df_test)-sbp));
mae_test_dbp_svm = sum(abs(predict(svm_para_dbp, df_test)-dbp)) ...
    /size(df_test, 1);
std_test_dbp_svm = std(abs(predict(svm_para_dbp, df_test)-dbp));