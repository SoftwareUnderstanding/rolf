# Asteroids: An Adventure in Deep Reinforcement Learning
*Classic Arcade Game, Cutting-Edge AI Learning Techniques*

This project is my implementation of the Proximal Policy Optimization algorithm (https://arxiv.org/pdf/1707.06347.pdf). I created a PyGame implementation of the Atari classic "Asteroids" and trained an AI to play it.


# Training

The AI started with an *agressive* policy:

![Full Speed Ahead, Captian](results/Beginning.gif)

It then picked up on the classic Asteroids technique, the "Turn and Shoot":

![The Cosmos are Alive with the Sound of Music and Lasers](results/Middle.gif)

Until finally, it learned to duck and weave well enough to clear the asteroids completely:

![Final Version](results/End.gif)

# Think You Can Beat The Computer?

If you want to take a crack at this version of asteroids,
<ol>
<li> Download these files</li>
<li> Open a console and navigate to the AsteroidsDeepReinforcement file</li>
<li> Enter <code> python game.py </code> to start the game!</li>
</ol>

Controls:
<ul>
<li>Jet Forward:  <code>W</code></li>
<li>Rotate Right: <code>D</code></li>
<li>Rotate Left:  <code>A</code></li>
<li>Fire Laser:   <code>Spacebar</code></li>
</ul>

# Want to Train One Yourself?

If you would like to train your own Asteroid-blasting AI pilot, I recommend the following:

<ol>
  <li>Install Python 3.7 (through Anaconda) on a Cuda-compatible computer</li>
  <li>Install torch, pygame, and numpy libraries</li>
  <li>In the console, navigate to AsteroidsDeepReinforcement/training</li>
  <li>Type 'ipython' to open an IPython kernel</li>
  <li>Run the following commands:
    <ul>
      <li><code>run training/networks</code></li>
      <li><code>run training/train</code></li>
      <li><code>run training/visualize</code></li>
    </ul>
  </li>
  <li>Create a PolicyNetwork object <code>network = PolicyNetwork().cuda()</code></li>
  <li>Train the network for as long as you want using <code>train(network, # )</code> Where "#" is the desired number of iterations. Each iteration should take anywhere from 10-20 seconds on an average GPU. I was able to see the results above after 1800 instances of training (~6 hours).</li>
</ol>
