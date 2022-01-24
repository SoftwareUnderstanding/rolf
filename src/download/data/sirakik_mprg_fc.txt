# mprg_fc
- Proximal Policy Optimization ([arXiv](https://arxiv.org/abs/1707.06347))  
- Google Research Football ([GitHub](https://github.com/google-research/football))

# 次やること
- GCNモデルによる学習の観察   
- graph評価を考える  

# bot vs bot  
``
python3 football.gfootball.play_game_shiraki.py --real_time=False --action_full --players=bot:left_player=1 --level=11_vs_11_hard_stochastic
``

# docker 
``
git clone https://github.com/google-research/football.git
cd football
``  
``
docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.2-gpu-py3 --build-arg DEVICE=gpu . -t gfootball
``  
``
nvidia-docker run -it -e TZ=Asia/Tokyo,LANG=ja_JP.UTF-8 -v foo/:/foo gfootball
``
