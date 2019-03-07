function [p_b,w_b_c] = feature_extraction(ecg,ppg,abp)
%%

%%%%%%%% Computing discrete wavelet decomposition of ABP signals and
%%%%%%%% selecting 4 to 8 levels to get a peaked signal
wt = modwt(abp,8,'db8');
wtrec = zeros(size(wt));
wtrec(4:8,:) = wt(4:8,:);
y_abp = imodwt(wtrec,'db8');
y_abp = single(y_abp);


%%%%%%% Using findpeaks function to find the SBP peaks
tm = single(1:1:length(y_abp));
[abpeaks,alocs,w,p] = findpeaks(y_abp,tm,'MinPeakDistance',400);
abpeaks = single(abpeaks);
alocs = single(alocs);


%%%%%%%%%%% Uncomment  block if peak selection is to be made on the basis of
%%%%%%%%%%% prominence of peaks
%{
%p = single(p);
%n_p = find(p<0.1);
%abpeaks(n_p) = [];
%alocs(n_p) = [];
%figure(7);
%hist(p,length(p));
[idx,C] = kmeans(p',3,'Replicates',5);
[B I] = sort(C);
n_p1 = length(idx(idx==1));
n_p2 = length(idx(idx==2));
n_p3 = length(idx(idx==3));
n_temp = find(C==C(I(1)) | C==C(I(2)));
n_p = find(idx==n_temp(1) | idx==n_temp(2));
if (n_p1 < length(idx)/6.5) || (n_p2 < length(idx)/6.5) || (n_p3 < length(idx)/6.5)
    [idx,C] = kmeans(p',4,'Replicates',5);
    [B I] = sort(C);
    n_p1 = length(idx(idx==1));
    n_p2 = length(idx(idx==2));
    n_p3 = length(idx(idx==3));
    n_p4 = length(idx(idx==4));
    n_temp = find(C==C(I(1)) | C==C(I(2)));
    n_p = find(idx==n_temp(1) | idx==n_temp(2));
    if (n_p1 < length(idx)/8) || (n_p2 < length(idx)/8) || (n_p3 < length(idx)/8) || (n_p4 < length(idx)/8)
        [idx,C] = kmeans(p',2,'Replicates',5);
        [B I] = sort(C);
        n_temp = find(C==C(I(1)));
        n_p = find(idx==n_temp(1));
    end
end
abpeaks(n_p) = [];
alocs(n_p) = [];
%}

%%%%%%%% Computing discrete wavelet decomposition of PPG signals and
%%%%%%%% selecting 4 to 8 levels to get a peaked signal
wt = modwt(ppg,8,'db8');
wtrec = zeros(size(wt));
wtrec(4:8,:) = wt(4:8,:);
y_ppg = imodwt(wtrec,'db8');
y_ppg = single(y_ppg);


%%%%%%%%%% identifying all PPG peaks using findpeaks function
tm = single(1:1:length(y_ppg));
[pppeaks,plocs,w,p] = findpeaks(y_ppg,tm,'MinPeakDistance',400);
%figure(6);
%hist(p,length(p));
%{


%%%%%%%% Purging peaks other than systolic peaks by using kmeans clustering
%%%%%%%% based on prominence of peaks and length of ppg signal
if (length(y_ppg) > 70000) && (length(y_ppg) < 100000)
    [idx,C] = kmeans(p',3,'Replicates',5);
    [B I] = sort(C);
    n_temp = find(C==C(I(1)) | C==C(I(2)));
    n_p = find(idx==n_temp(1) | idx==n_temp(2));
elseif length(y_ppg) > 100000
    [idx,C] = kmeans(p',4,'Replicates',5);
    [B I] = sort(C);
    n_temp = find(C==C(I(1)) | C==C(I(2)));
    n_p = find(idx==n_temp(1) | idx==n_temp(2));
else
    [idx,C] = kmeans(p',2,'Replicates',5);
    [B I] = sort(C);
    n_temp = find(C==C(I(1)));
    n_p = find(idx==n_temp(1));
end
pppeaks = single(pppeaks);
plocs = single(plocs);
p = single(p);
pppeaks(n_p) = [];
plocs(n_p) = [];
%}

%%%%%%%%% Uncomment block to generate all DBP peaks, all maximum gradient point of
%%%%%%%%% ppg signal and all inflection points of ppg signal.
%{

for i=1:length(alocs)-1
    [loc,idx] = min(abp(alocs(i):alocs(i+1)));
    dblocs(i) = alocs(i) + idx - 1;
end

for i=1:length(plocs)-1
    [loc,idx] = min(ppg(plocs(i):plocs(i+1)));
    pmlocs(i) = plocs(i) + idx - 1;
    temp = ppg(pmlocs(i):plocs(i+1));
    grad = gradient(temp);
    qw = find(grad==max(grad));
    grad_indx(i) = qw(1) + pmlocs(i); 
end

for i=1:length(plocs)-1
    temp = ppg(plocs(i):pmlocs(i));
    [res_x, idx] = knee_pt(temp,tm(plocs(i):pmlocs(i)));
    inflc_indx(i) = idx + plocs(i) - 1;
end
%}

%}

%%%%%%%% Computing discrete wavelet decomposition of ECG signals and
%%%%%%%% selecting 4 to 6 levels to get a peaked signal
wt = modwt(ecg,8,'db8');
wtrec = zeros(size(wt));
wtrec(4:6,:) = wt(4:6,:);
y_ecg = imodwt(wtrec,'db8');
y_ecg = single(y_ecg);
y_ecg = single(y_ecg);


%%%%%%% Using findpeaks function to find the all ECG peaks
tm = single(1:1:length(y_ecg));
[qrspeaks,rlocs,w,p] = findpeaks(y_ecg,tm);
qrspeaks = single(qrspeaks);
rlocs = single(rlocs);
p = single(p);
%figure(5);
%hist(p,length(p));

%%%%%%%%%%% Using kmeans clustering to select R peaks based on prominence
[idx,C] = kmeans(p',3,'Replicates',5);
[B I] = sort(C);
n_temp = find(C==C(I(1)) | C==C(I(2)));
n_p = find(idx==n_temp(1) | idx==n_temp(2));
qrspeaks(n_p) = [];
rlocs(n_p) = [];


%%%%%%%%%%% Uncomment block and sub-blocks to generate a set of plots
%{
%{
time_plot = single(linspace(0,length(y_ecg)/1000,length(y_ecg)));
figure(1);
plot(time_plot,y_ecg);
hold on
plot(time_plot(rlocs),qrspeaks,'ro');
plot(time_plot,y_ppg);
plot(time_plot(plocs),pppeaks,'ro');
xlabel('Seconds');
ylabel('Normalized Amplitude');
%title('R Peaks Localized by Wavelet Transform with Automatic Annotations');
%}

%{
time_plot = single(linspace(0,length(ppg)/1000,length(ppg)));
figure(2);
plot(time_plot,ppg);
hold on
plot(time_plot(plocs),ppg(plocs),'ro','MarkerSize',5,'MarkerFaceColor',[0 0 1]);
plot(time_plot,ecg);
plot(time_plot(rlocs),ecg(rlocs),'v','MarkerSize',5,'MarkerFaceColor',[.6 .6 .6]);
%plot(time_plot(pmlocs),ppg(pmlocs),'^','MarkerSize',10,'MarkerFaceColor',[1 .6 .6]);
%plot(time_plot(grad_indx),ppg(grad_indx),'d','MarkerSize',10,'MarkerFaceColor',[1 0 .6]);
%plot(time_plot(inflc_indx),ppg(inflc_indx),'p','MarkerSize',10,'MarkerFaceColor',[0 1 .6]);
xlabel('Seconds');
ylabel('Normalized Amplitude');
%}

%{
figure(3);
tm = 1:1:length(abp);
time_plot = linspace(0,length(abp)/1000,length(abp));
plot(time_plot,abp);
hold on
plot(time_plot(alocs),abp(alocs),'ro');
%plot(time_plot(dblocs),abp(dblocs),'*');
plot(time_plot,y_abp);
xlabel('Seconds');
ylabel('Normalized Amplitude');
%}

%{
figure(4);
plot(time_plot,ppg);
hold on
plot(time_plot(plocs),ppg(plocs),'ro','MarkerSize',10,'MarkerFaceColor',[.6 .6 .6]);
%plot(time_plot(pmlocs),ppg(pmlocs),'^','MarkerSize',10,'MarkerFaceColor',[1 .6 .6]);
%plot(time_plot(grad_indx),ppg(grad_indx),'d','MarkerSize',10,'MarkerFaceColor',[1 0 .6]);
%plot(time_plot(inflc_indx),ppg(inflc_indx),'p','MarkerSize',10,'MarkerFaceColor',[0 1 .6]);
xlabel('Seconds');
ylabel('Normalized Amplitude');

%}

%}

%%%%%%% Selecting 10 random R-peaks from set of R-peaks
rnd_indx = randi((length(rlocs)-2),10,1);

%%%%%%%% Finding corresponding SBP and systolic PPG peaks close to but
%%%%%%%% greater than each of the 10 random R-peaks
indx_pmaxf = single(zeros(1,length(rnd_indx)));
indx_SBP = single(zeros(1,length(rnd_indx)));
for i = 1:length(rnd_indx)
    qw = find(plocs>rlocs(rnd_indx(i)));
    mw = find(alocs>rlocs(rnd_indx(i)));
    indx_pmaxf(i) = plocs(min(qw));
    indx_SBP(i) = alocs(min(mw));
end
indx_pminBf = single(zeros(1,length(indx_pmaxf)));
indx_pminAf = single(zeros(1,length(indx_pmaxf)));
PATp_ini = single(zeros(1,length(indx_pmaxf)));
PATf_ini = single(zeros(1,length(indx_pmaxf)));
grad_indx = single(zeros(1,length(indx_pmaxf)));
PATd_ini = single(zeros(1,length(indx_pmaxf)));
HR_ini = single(zeros(1,length(indx_pmaxf)));
inflc_indx = single(zeros(1,length(indx_pmaxf)));
augindx_ini = single(zeros(1,length(indx_pmaxf)));
LASI_ini = single(zeros(1,length(indx_pmaxf)));
S1_ini = single(zeros(1,length(indx_pmaxf)));
S2_ini = single(zeros(1,length(indx_pmaxf)));
S3_ini = single(zeros(1,length(indx_pmaxf)));
S4_ini = single(zeros(1,length(indx_pmaxf)));
SBP_ini = single(zeros(1,length(indx_pmaxf)));
DBP_ini = single(zeros(1,length(indx_pmaxf)));

%%%%%%% Calculating parameter based features PATp, PATd, PATf,
%%%%%%% HR, Augmentation Index, LASI, S1, S2, S3, S4 for 10 randomly
%%%%%%% selected R-peaks. Also, SBP and DBP features from ABP signal have
%%%%%%% been calculated for these 10 R peaks
for i = 1:length(indx_pmaxf)
  [amp,indx_pmin] = min(ppg(rlocs(rnd_indx(i)):indx_pmaxf(i)));
  indx_pminBf(i) = indx_pmin + rlocs(rnd_indx(i)) -1;
  indx_temp = find(plocs==indx_pmaxf(i));
  indx_temp2 = plocs(indx_temp+1);
  [amp,indx_pmin] = min(ppg(indx_pmaxf(i):indx_temp2));
  indx_pminAf(i) = indx_pmin + indx_pmaxf(i) -1;
  PATp_ini(i) = tm(indx_pmaxf(i)) - tm(rlocs(rnd_indx(i)));
  PATf_ini(i) = tm(indx_pminBf(i)) - tm(rlocs(rnd_indx(i)));
  temp = ppg(indx_pminBf(i):indx_pmaxf(i));
  grad = gradient(temp);
  qw = find(grad==max(grad));
  grad_indx(i) = qw(1) + indx_pminBf(i); 
  PATd_ini(i) = tm(grad_indx(i)) - tm(rlocs(rnd_indx(i)));
  HR_ini(i) = tm(rlocs(rnd_indx(i)+1)) - tm(rlocs(rnd_indx(i)));
  temp = ppg(indx_pmaxf(i):indx_pminAf(i));
  [res_x, idx] = knee_pt(temp,tm(indx_pmaxf(i):indx_pminAf(i)));
  inflc_indx(i) = idx + indx_pmaxf(i);
  augindx_ini(i) = ppg(inflc_indx(i))/ppg(indx_pmaxf(i));
  LASI_ini(i) = tm(inflc_indx(i)) - tm(indx_pmaxf(i));
  S1_ini(i) = trapz(ppg(indx_pminBf(i):grad_indx(i)));
  S2_ini(i) = trapz(ppg(grad_indx(i):indx_pmaxf(i)));
  S3_ini(i) = trapz(ppg(indx_pmaxf(i):inflc_indx(i)));
  S4_ini(i) = trapz(ppg(inflc_indx(i):indx_pminAf(i)));
  SBP_ini(i) = abp(indx_SBP(i));
  indx_temp = find(alocs==indx_SBP(i));
  indx_temp2 = alocs(indx_temp+1);
  DBP_ini(i) = min(abp(indx_SBP(i):indx_temp2));
end


%%%%% Calculating the median of PATp features and finding the feature
%%%%% closest to median
check = median(PATp_ini);
diff = abs(PATp_ini - check);
n_df = find(diff==min(diff));

%%%%%%% Using the median-closest index for rest of the features
PATp = PATp_ini(n_df(1));
PATf = PATf_ini(n_df(1));
PATd = PATd_ini(n_df(1));
HR = HR_ini(n_df(1));
augindx = augindx_ini(n_df(1));
LASI = LASI_ini(n_df(1));
S1 = S1_ini(n_df(1));
S2 = S2_ini(n_df(1));
S3 = S3_ini(n_df(1));
S4 = S4_ini(n_df(1));
SBP = SBP_ini(n_df(1));
DBP = DBP_ini(n_df(1));
%}

%%%%%%%Uncomment block if features from 10 windows are to be used instead of median 
%{
param_based = [PATp_ini
               PATf_ini
               PATd_ini
               HR_ini
               augindx_ini
               LASI_ini
               S1_ini
               S2_ini
               S3_ini
               S4_ini
               SBP_ini
               DBP_ini];
%}

%%%%%Putting final parameter based features including SBP and DBP in a
%%%%%vector.
p_b = [PATp PATf PATd HR augindx LASI S1 S2 S3 S4 SBP DBP];

%%%%%%%%% Using median-closest index to calculate whole based features
indx_temp = find(plocs==indx_pmaxf(n_df(1)));
indx_temp2 = plocs(indx_temp+1);
temp = ppg(indx_pmaxf(n_df(1)):indx_temp2);
w_b_c = temp;


%{
time_plot = single(linspace(0,length(w_b_c)/1000,length(w_b_c)));
figure(10);
plot(time_plot,w_b_c);
xlabel('Time (in s)');
ylabel('Normalized Amplitude');
%}

clearvars -except p_b w_b_c ;

end