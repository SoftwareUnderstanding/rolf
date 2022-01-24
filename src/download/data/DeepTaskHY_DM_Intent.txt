# 1. [M2-7] Intention Classifier

## 2. package summary 

Intention Classifier is a module that analyzes the intention of the user’s utterance. This module modifies and combines “bi-RNN” and “Attention mechanism” to implement an Intention classification model. 

- 2.1 Maintainer status: maintained
- 2.2 Maintainer: Yuri Kim, [yurikim@hanyang.ac.kr]()
- 2.3 Author: Yuri Kim, [yurikim@hanyang.ac.kr]()
- 2.4 License (optional): 
- 2.5 Source git: https://github.com/DeepTaskHY/DM_Intent

## 3. Overview

To analyze the intention of the user's utterance, this module consists of two parts: 1)keyword extraction, 2)intention analysis. To extract the keywords of the user's utterance, we used Google Dialogflow. This module combines “bi-RNN” and “Attention mechanism” to implement an Intention classification model.  

![KCC2020-Att-BiRNN](./image/KCC2020-Att-BiRNN.jpg)

## 4. Hardware requirements

None

## 5. Quick start

### 5.1 Install dependency

**ros-melodic**

```
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
$ sudo apt-get update  
$ sudo apt-get install ros-melodic-desktop-full  
$ sudo rosdep init  
$ rosdep update  

# or download dockerimage
$ docker pull ribin7bok/deeptask
$ docker run -e NVIDIA_VISIBLE_DEVICES=0 --volume ~/$workspace_name:/workspace -it --name social_intent ribin7bok/deeptask

```

**requirements**

```
$ mkdir $dir_name
$ cd $dir_name
$ mkdir src
$ catkin_make
$ cd src
$ git clone --recursive https://github.com/DeepTaskHY/DM_Intent.git
$ sudo apt-get update && sudo apt-get install python3-pip 
$ sudo apt-get install default-jdk
$ sudo apt-get install default-jre
$ sudo apt-get install python3-pyyaml

$ cd dm_intent
$ sudo pip3 install -r requirements.txt  
```

**download files**

```
$ cd dm_intent  
$ sh model_download.sh  
```

### 5.2 Start the module

```
$ cd $dir_name
$ source devel/setup.bash  
$ roslaunch src/DM_Intent/dm_intent/launch/dm_intent.launch
```

## 6. Input/Subscribed Topics

```
{  
   "header": {
        "source": "perception",
        "target": ["dialog", "planning"],
        "timestamp": "1563980552.933543682",
        "content": ["human_speech"]
   },
   "human_speech":{ 
      "speech":"안녕하세요",
      "name":"이병현",
   }
}
```

○ header (header/recognitionResult): contain information about published time, publisher name, receiver name and content.  

- timestamp: published time  
- source: publish module name  
- target: receive module name  
- content: role of this ROS topic name  

○ dialog_intent (dialog_intent/recognitionResult): contain human speech and user name.  

- speech: human speech    
- name: user name   
- information: keyword to use for intention classification 

## 7. Output/Published Topics

```
{
    "header": {
        "target": ["planning"], 
        "content": ["dialog_intent"], 
        "timestamp": "1563980561.940629720", 
        "source": "dialog"
    }, 
    "dialog_intent": {
        "speech": "좋아진 것 같아.", 
        "intent": "단순 정보 전달", 
        "name": "이병현",
        "information": {}
    }
}
```

○ header (header/dialog_intent): contain information about published time, publisher name, receiver name and content.  

- timestamp: published time  
- source: publish module name  
- target: receive module name  
- content: role of this ROS topic name  

○ dialog_intent (dialog_intent/dialog_intent): contain intent, human speech, name and information.  

- intent: intention of the human speech  
- speech: human speech  
- name: user name  

## 8. Parameters

There are one category of parameters that can be used to configure the module: deep learning model.  

**8.1 model parameters**  

- ~data_path (string, default: None): The path where data(pickle file) is stored.  
- ~RNN_SIZE (int, default: 192): rnn size  
- ~EMBEDDING_SIZE (int, default: 200): embedding vector dimension  
- ~ATTENTION_SIZE (int, default: 50): attention dimension  
- ~L2_LEG_LAMBDA (float, default: 3.0): l2 leg lambda  
- ~EPOCH (int, default: 50): epoch size  
- ~BATCH_SIZE (int, default: 64): batch size  
- ~N_Label (int, default: 7): label size  
- ~DROPOUT_KEEP_PROB (float, default: 0.5): dropout
- ~BATCH_SIZE (int, default: 64): batch size   
- ~N_Label (int, default: 7): label size   
- ~DROPOUT_KEEP_PROB (float, default: 0.5): dropout  

## 9. Related Applications (Optional)

None

## 10. Related Publications (Optional)

- Wang, Y., Huang, M., & Zhao, L. Attentionbased lstm for aspect-level sentiment classification.In Proceedings of the 2016 conference on empiricalmethods in natural language processing. 606-615, 2016
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. Efficient estimation of word representations in vector space. arXiv:1301.3781. Retrieved from https://arxiv.org/abs/1301.3781 , 2013 
- Zhou, P., Shi, W., Tian, J., Qi, Z., Li, B., Hao, H., & Xu, B. Attention-based bidirectional long shortterm memory networks for relation classification. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, 2. 207-212, 2016
- Jeongmin Yoon and Youngjoong Ko. Speech-Act Analysis System Based on Dialogue Level RNNCNN Effective on the Exposure Bias Problem. Journal of KIISE, 45, 9 (2018), 911-917, 2018
