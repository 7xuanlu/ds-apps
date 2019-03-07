function X_f2 = preprocessing(sgnl)

%%%%%% resampling to 1000 Hz
sgnl_rsmp = resample(sgnl,8,1);
%%%%%% decomposing into Db8 wavelets to 10 levels
[decomp,bookp] = wavedec(sgnl_rsmp,10,'db8');
[cd1,cd2,cd3,cd4,cd5,cd6,cd7,cd8,cd9,cd10] = detcoef(decomp,bookp,[1 2 3 4 5 6 7 8 9 10]);
a10 = appcoef(decomp,bookp,'db8');
%%%%%%%%%%%% Uncomment this block to plot the decomposed wavelets
%{
figure(1);
subplot(10,1,1)
plot(cd10);
subplot(10,1,2)
plot(cd9)
subplot(10,1,3)
plot(cd8);
subplot(10,1,4)
plot(cd7);
subplot(10,1,5);
plot(cd6);
subplot(10,1,6);
plot(cd5);
subplot(10,1,7);
plot(cd4);
subplot(10,1,8);
plot(cd3);
subplot(10,1,9);
plot(cd2);
subplot(10,1,10);
plot(cd1);
%}

%%%%%%%% Zeroing the 0-0.25 Hz components and 250-500 Hz components
a10 = 0 * a10;
cd10 = 0 * cd10;
cd2 = 0 * cd2;

X = [a10,cd10,cd9,cd8,cd7,cd6,cd5,cd4,cd3,cd2,cd1];

%%%%%%%% Recomposing the wavelets into  the signal
X_f = waverec(X,bookp,'db8');

%%%%% Detrending and normalizing the recomposed signal
X_f_mean = mean(X_f);
X_f_temp = X_f - X_f_mean;
X_f2 = (X_f_temp - min(X_f))/(max(X_f) - min(X_f));

%%%%%%% Uncomment this block for plot original signal and preprocessed signal
%{
tm = (1:1:length(X_f2))./1000;
t_sgnl = (1:1:length(sgnl))./125;

figure(2);
plot(tm,X_f2);
xlabel('Time (in s)');
ylabel('Normalized Amplitude');
figure(3);
plot(t_sgnl,sgnl);
xlabel('Time (in s)');
ylabel('Normalized Amplitude');
%}
end