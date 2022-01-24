# pytorch-toy-GAN

Toy model of a **Generative Adversarial Network** (see: https://arxiv.org/abs/1406.2661) where: 
 - the **generator** is trained to generate fake normal distributions from noise (uniform distributions).  
 - the **discriminator** is trained to detect if the given normal distributions are real or fake/generated.
 
The **uniform distributions** (noise) are generated on-the-fly with a uniform distribution, which is quite a common process. The **normal distributions** -- the *training data* -- are also generated on-the-fly, which has 2 advantages:
 - this simulate infinite data  
 - this makes the code much shorter -- no need to read, format, parse external data -- so we can focus more on the GAN itself!
