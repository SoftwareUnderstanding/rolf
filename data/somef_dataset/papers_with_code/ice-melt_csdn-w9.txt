# 第9周作业 `csdn-w9`

## 更新日志
- 修改 `run.sh` 脚本中数据集路径
- 修改 `ssd_mobilenet_v1_pets.config` 文件
	- `PATH_TO_BE_CONFIGURED` 路径
	- num_classes: 5
	- num_steps: 100
	- num_examples: 47
	- max_evals: 1
- 修改 `exporter.py` 脚本
	- 将第72行的参数 layout_optimizer 替换为 optimize_tensor_layout
- 新增 `object_detection/dataset_tools/create_w9_tf_record.py` 脚本生成 tfrecord 文件
	- 原数据见 [https://gitee.com/ai100/quiz-w8-data.git](https://gitee.com/ai100/quiz-w8-data.git)
	- 脚本参考 `research/object_detection/dataset_tools/create_pet_tf_record.py`
- 新增 tinymind 上运行时需要的相关代码,见 [https://gitee.com/ai100/quiz-w8-doc](https://gitee.com/ai100/quiz-w8-doc)
- 使用 tensorflow/models/research/下的 object_detection 和 slim 文件夹下的代码对项目初始化

## 命令记录

```
# from ~/Documents/Github/csdn-w9/research
protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/dataset_tools/create_w9_tf_record.py \
--label_map_path=/home/ice-melt/Documents/Github/csdn-w9/research/data/labels_items.txt \
--data_dir=/home/ice-melt/Documents/Github/csdn-w9/research/data \
--output_dir=/home/ice-melt/Documents/Github/csdn-w9/research/data
```