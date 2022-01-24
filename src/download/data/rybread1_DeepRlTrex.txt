# DeepRlTrex

Reinforcement learning implementation of double-deep-q-learning, dueling network architure and PER to play the Google 
Chrome Trex Game directly from the browser.

![](/assets/trex_demo.gif)

**To run this on your local machine you'll probably have to fine-tune a few parameters:**
- Because the env runs by grabbing screenshots directly from your monitor you will have to ensure the screen capture dimensions are actually working correctly
   * Check the bbox and terminal_bbox: bbox should capture the play area for the actualy game, terminal_bbox should capture some unique identifier for when the game is over
   * Make sure the render dimensions are large enough to fully expand the entire "runway" for the t-rex. If the dimensions of the window rendered are too small, the agent has more difficulty (since it can't see as far ahead)
   * Make sure that the _update_state() function properly reshaping the frames
   * Finally, in agent.py, check the input dimensions for the CNN

**References:**
- Double Deep Q-Network: https://arxiv.org/pdf/1509.06461.pdf
- Dueling Network Architecture: https://arxiv.org/pdf/1511.06581.pdf
- Prioritized Experience Replay: https://arxiv.org/pdf/1511.05952.pdf
