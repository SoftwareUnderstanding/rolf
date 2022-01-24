# Transformer效率优化
- An Attention Free Transformer: https://arxiv.org/abs/2105.14103

# 多模态Transformers
- E2E-VLP: https://arxiv.org/abs/2106.01804 
  - 问题: 论文中transformer decoder generation存在输入信息泄漏
  - 潜在提升点: 
    - 提出一个通用的解决视频 和 文本多模态预训练的统一的Framework, 视频Encoder-decoder解决视频相关的任务， 视频是否匹配
    - 图像任务分离, image Encoder-decoder(目标检测, caption generation, 分类等, 把imagenet 的图像label也利用起来),  text任务分离 MLM,  多模态Encoder(image-text matching)
    - 所有可以获取的数据放一起做预训练(COCO/Visual Genome/Conceptual Captions ) 
    - 在视频预训练的应用: 视频预训练也是在 Kinetics上预训练模型，然后针对视频抽取特征，抽取特征的知识都开源于Kinectics上，可以把视频预训练也变成端到端pipeline, 这样就不用抽取特征
  
- CLIP: https://github.com/openai/CLIP
  - 解决: 图文多模态检索 和 few shot的问题

- End-to-End Video Instance Segmentation with Transformers:https://arxiv.org/abs/2011.14503
  - Transformer在视频实例分割中的应用 

- TAP: Text-Aware Pre-training for Text-VQA and Text-Caption
  - 输入: image, roi, ocr 模态，可以作为roi_model的参考

# 视频预训练
- Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling
  - 架构和attrim-mmbt非常类似
  - 用离散采样的video-clip, 用图片特征抽取代替视频特征抽取 
- Video Transformer Network
  - 单独设计一个时序模块transformer，用于学习视频时间序列 
- VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text
  - 直接从视频像素进行学习

# Transformer同时做视觉和多模态任务
- UniT: Multimodal Multitask Learning with a Unified Transformer: https://arxiv.org/abs/2102.10772

# 文档预训练
- DocFormer: End-to-End Transformer for Document Understanding
  - 增加一个新的图像重建预训练任务

