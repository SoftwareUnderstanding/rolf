# ALHE-projekt

## Temat 12. Naucz robota chodzić
Korzystając ze środowiska OpenAI Gym: https://gym.openai.com/envs/#box2d zaimplementuj poznany na wykładzie algorytm (np. ewolucyjny/genetyczny) by nauczyć dwunożnego robota chodzić. Zadanie wymaga zaznajomienia z ideą sieci neuronowych, których parametry należy optymalizować w ramach projektu. Uwaga: Proszę nie stosować metod uczenia ze wzmocnieniem, a jedynie algorytmy oparte o populacje / algorytmy przeszukiwania.

**Dokumentacja wstępna:** https://docs.google.com/document/d/1SGi361dUx475NzdJJFqGgjtpwL87CuuYX02KqkK1vzs/edit# 

**Dokumentacja końcowa:** https://docs.google.com/document/d/1bv8zoB7cyC3gjCCWxG4fd6qIAJRbG4XtrsH5_Z5fvVI/edit

## Kroki
1. Sklonuj pliki projektu używając komendy:
```
git clone https://github.com/kklipski/ALHE-projekt.git
```
2. Utwórz nowy projekt w JetBrains PyCharm (IDE używane przez autorów).
3. Dodaj do nowoutworzonego projektu pliki źródłowe ze sklonowanego repozytorium (pliki znajdujące się w folderze [src](src)).
4. Zainstaluj w swoim projekcie pakiety wymienione w pliku [requirements.txt](requirements.txt).
5. Jeśli występują problemy z instalacją pakietu *torch*, skorzystaj z poniższego sposobu:

   W terminalu w PyCharm (warunek: otwarty projekt) użyj (pierwszej) komendy wygenerowanej na stronie: https://pytorch.org/get-started/locally/
	
   Przykład dla konfiguracji: PyTorch Build: *Stable (1.1)*, Your OS: *Windows*, Package: *Pip*, Language: *Python 3.7*, CUDA: *10.0*:
```
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl
```
6. W celu przeprowadzenia uczenia ze wzmocnieniem należy uruchomić skrypt [train.py](src/td3/train.py) z folderu [td3](src/td3).
7. W celu przeprowadzenia uczenia ze wzmocnieniem połączonego z algorytmem ewolucyjnym należy uruchomić skrypt [test_evo.py](src/td3_evo/test_evo.py) z folderu [td3_evo](src/td3_evo).

## Testy
1. standardowe TD3 (z ograniczeniem ilości kroków bez ruchu oraz karami za zatrzymywanie się i utykanie w miejscu)
2. parametry: n_networks = 5, saved_scores = 5, episodes_ready = [5, 7, 10, 15, 20], explore_prob = 0.05, explore_factors = [0.99, 1.01]
3. parametry: n_networks = 5, saved_scores = 10, episodes_ready = [10, 15, 20, 25, 30], explore_prob = 0.5, explore_factors = [0.95, 1.05]
4. parametry: n_networks = 10, saved_scores = 10, episodes_ready = [10, 11, 12, 13, 14, 15, 16, 20, 25, 30], explore_prob = 0.5, explore_factors = [0.95, 1.05]
5. parametry: n_networks = 5, saved_scores = 10, episodes_ready = [10, 15, 20, 25, 30], explore_prob = 0.05, explore_factors = [0.99, 1.01]
6. parametry: n_networks = 10, saved_scores = 10, episodes_ready = [10, 11, 12, 13, 14, 15, 16, 20, 25, 30], explore_prob = 0.05, explore_factors = [0.99, 1.01]

Każdy z testów, oprócz wariantów 1. i 6., zostanie przeprowadzony trzykrotnie w celu walidacji wyników (wariant 1. zostanie powtórzony pięciokrotnie, a wariant 6. - dwukrotnie); w sumie: 19 testów.

## Autorzy
- **Lipski Kamil** - [kklipski](https://github.com/kklipski)
- **Rzepka Karol** - [krzepka](https://github.com/krzepka)

## Źródła
- https://arxiv.org/pdf/1711.09846.pdf - Population Based Training of Neural Networks (article)
- https://deepmind.com/blog/population-based-training-neural-networks/ - Population based training of neural networks (blog post)
- https://github.com/vy007vikas/PyTorch-ActorCriticRL - PyTorch implementation of DDPG algorithm for continuous action reinforcement learning problem (used in the project) (**DEPRECATED**)
- https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2 - Twin Delayed DDPG (TD3) PyTorch solution for Roboschool and Box2d environment (used in the project)

![](https://github.com/kklipski/ALHE-projekt/blob/master/bipedalwalker-v2.gif)
