

# 视频全量目标分析和建模比赛VFOST算法


在这个库中，我们发布了中国软件杯2020视频全量目标分析和建模比赛VFOST算法的代码。VFOST能够解决复杂城市场景下语义分割和目标跟踪任务，并能够进行结果的统计。本项目引用了两个开源项目，pointrend和deepsort,这基础上进行了代码优化.


## 引用paper
pointrend: [arxiv](https://arxiv.org/abs/1912.08193).

deepsort:  [arxiv](https://arxiv.org/abs/1703.07402).

## 推理
使用一块GPU进行推理:
	将需要进行推理的视频文件(.mp4, .avi等)放入input文件夹下
	python3 run.py 



