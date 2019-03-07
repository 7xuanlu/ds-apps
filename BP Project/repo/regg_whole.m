%% Whole Based Regression
clc;
close all;
clear all;
%% Training
load('features_train.mat');
df_train = whole_based_train;
sbp = param_based_train(:, end-1);
dbp = param_based_train(:, end);

% Whole based training
tree_whole_sbp = fitrtree(df_train, sbp, "OptimizeHyperparameters", "auto");
tree_whole_dbp = fitrtree(df_train, dbp, "OptimizeHyperparameters", "auto");
svm_whole_sbp = fitrsvm(df_train, sbp, "OptimizeHyperparameters", "auto");
svm_whole_dbp = fitrsvm(df_train, dbp, "OptimizeHyperparameters", "auto");

%% Validation
load('features_valid.mat');
df_valid = whole_based_valid;
sbp = param_based_valid(:, end-1);
dbp = param_based_valid(:, end);

% Evaluation on decision tree
mae_valid_sbp_tree = sum(abs(predict(tree_whole_sbp, df_valid)-sbp)) ...
    /size(df_valid, 1);
std_valid_sbp_tree = std(abs(predict(tree_whole_sbp, df_valid)-sbp));
mae_valid_dbp_tree = sum(abs(predict(tree_whole_dbp, df_valid)-dbp)) ...
    /size(df_valid, 1);
std_valid_dbp_tree = std(abs(predict(tree_whole_dbp, df_valid)-dbp));

% Evaluation on svm
mae_valid_sbp_svm = sum(abs(predict(svm_whole_sbp, df_valid)-sbp)) ...
    /size(df_valid, 1);
std_valid_sbp_svm = std(abs(predict(svm_whole_sbp, df_valid)-sbp));
mae_valid_dbp_svm = sum(abs(predict(svm_whole_dbp, df_valid)-dbp)) ...
    /size(df_valid, 1);
std_valid_dbp_svm = std(abs(predict(svm_whole_dbp, df_valid)-dbp));

%% Test
load('features_test.mat');
df_test = whole_based_test;
sbp = param_based_test(:, end-1);
dbp = param_based_test(:, end);

% Evaluation on decision tree
mae_test_sbp_tree = sum(abs(predict(tree_whole_sbp, df_test)-sbp)) ...
    /size(df_test, 1);
std_test_sbp_tree = std(abs(predict(tree_whole_sbp, df_test)-sbp));
mae_test_dbp_tree = sum(abs(predict(tree_whole_dbp, df_test)-dbp)) ...
    /size(df_test, 1);
std_test_dbp_tree = std(abs(predict(tree_whole_dbp, df_test)-dbp));

% Evaluation on svm
mae_test_sbp_svm = sum(abs(predict(svm_whole_sbp, df_test)-sbp)) ...
    /size(df_test, 1);
std_test_sbp_svm = std(abs(predict(svm_whole_sbp, df_test)-sbp));
mae_test_dbp_svm = sum(abs(predict(svm_whole_dbp, df_test)-dbp)) ...
    /size(df_test, 1);
std_test_dbp_svm = std(abs(predict(svm_whole_dbp, df_test)-dbp));