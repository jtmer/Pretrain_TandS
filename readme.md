/TimeSiam_forecast/readme.md
/SimMTM_classification/readme.md

时序预测：
进入TimeSiam_forecast文件夹，修改run.sh配置项，运行run.sh

时序分类：
进入SimMTM_classification文件夹
直接训练：
```
python ./code/main.py --training_mode train --target_dataset Epilepsy
```
预训练微调：
```
python ./code/main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy
```