close all
clear all 

data = readtable("data_complete.csv");

ahu5 = data(strcmp(data.ahu_id,'AHU5'),:);
ahu5 = table2timetable(ahu5);

ahu5.AHU_RUN = strcmp(ahu5.AHU_RUN,'active');

ahu5.FCU_1 = strcmp(ahu5.FCU_1,'active');
ahu5.FCU_2 = strcmp(ahu5.FCU_2,'active');
ahu5.FCU_3 = strcmp(ahu5.FCU_3,'active');
ahu5.FCU_4 = strcmp(ahu5.FCU_4,'active');
ahu5.FCU_5 = strcmp(ahu5.FCU_5,'active');
ahu5.FCU_6 = strcmp(ahu5.FCU_6,'active');
ahu5.FCU_7 = strcmp(ahu5.FCU_7,'active');
ahu5.FCU_8 = strcmp(ahu5.FCU_8,'active');
ahu5.FCU_9 = strcmp(ahu5.FCU_9,'active');
ahu5.FCU_10 = strcmp(ahu5.FCU_10,'active');
ahu5.FCU_11 = strcmp(ahu5.FCU_11,'active');

ahu5.fcus = ahu5.FCU_1 + ahu5.FCU_2 + ahu5.FCU_3 + ahu5.FCU_4 + ahu5.FCU_5 ...
    + ahu5.FCU_6 + ahu5.FCU_7 + ahu5.FCU_8 + ahu5.FCU_9 + ahu5.FCU_10 + ahu5.FCU_11;

dataset = ahu5(:, [
    "current_co2","current_temperature","current_humidity","fcus","AHU_MDT",...
    "AHU_MODE","AHU_OBDM","AHU_RDT","AHU_SDT","AHU_RUN","AHU_TMSET",...
    "temperature_out","humidity_out","solar_radiation"]);

dataset.AHU_OBDM(isnan(dataset.AHU_OBDM)) = 0;
dataset.AHU_RUN = double(dataset.AHU_RUN);


dataset_avg = retime(dataset,'regular','mean','TimeStep',minutes(15));

dataset_avg = fillmissing(dataset_avg,'nearest');


normData = normalize(dataset_avg, 'zscore');


train_data = iddata(normData{:,["AHU_RDT"]}, normData{:,["AHU_SDT","AHU_RUN","fcus"...
    "temperature_out","humidity_out","solar_radiation"]});


train_data.InputName = {"AHU_SDT","AHU_RUN", "fcus","temperature_out","humidity_out","solar_radiation"};

train_data.OutputName = 'AHU_RDT';