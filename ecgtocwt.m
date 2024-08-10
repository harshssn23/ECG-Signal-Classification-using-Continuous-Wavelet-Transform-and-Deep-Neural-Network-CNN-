%Program to create CWT Image database from ECG signals
load('ECGData.mat');
data=ECGData.Data;
labels=ECGData.Labels;

ARR=data (1:30,:); %Taken first 30 samples
CHF=data (97:126,:);
NSR=data (127:156,:);

signallength=500;
fb=cwtfilterbank('SignalLength', signallength, 'Wavelet', 'amor', 'VoicesPeroctave', 12);

mkdir('ecgdataset');
mkdir('ecgdataset\arr');
mkdir('ecgdataset\chf');
mkdir('ecgdataset\nsr');

ecgtype={'ARR', 'CHF', 'NSR'};

ecg2cwtscg(ARR,fb,ecgtype{1});  %Arrhythmias
ecg2cwtscg(CHF,fb,ecgtype{2});  %Congestive Heart Failure
ecg2cwtscg(NSR,fb,ecgtype{3});  %Normal Sinus Rhythm