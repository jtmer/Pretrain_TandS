# wget -O SleepEEG.zip https://figshare.com/ndownloader/articles/19930178/versions/1
wget -O Epilepsy.zip https://figshare.com/ndownloader/articles/19930199/versions/1
# wget -O FD-A.zip https://figshare.com/ndownloader/articles/19930205/versions/1
# wget -O FD-B.zip https://figshare.com/ndownloader/articles/19930226/versions/1
# wget -O HAR.zip https://figshare.com/ndownloader/articles/19930244/versions/1
# wget -O Gesture.zip https://figshare.com/ndownloader/articles/19930247/versions/1
# wget -O ECG.zip https://figshare.com/ndownloader/articles/19930253/versions/1
# wget -O EMG.zip https://figshare.com/ndownloader/articles/19930250/versions/1

# unzip  SleepEEG.zip -d dataset/SleepEEG/
unzip Epilepsy.zip -d dataset/Epilepsy/
# unzip FD-A.zip -d dataset/FD-A/
# unzip FD-B.zip -d dataset/FD-B/
# unzip HAR.zip -d dataset/HAR/
# unzip Gesture.zip -d dataset/Gesture/
# unzip ECG.zip -d dataset/ECG/
# unzip EMG.zip -d dataset/EMG/

#rm {SleepEEG,Epilepsy,FD-A,FD-B,HAR,Gesture,ECG,EMG}.zip
rm SleepEEG.zip
rm Epilepsy.zip


