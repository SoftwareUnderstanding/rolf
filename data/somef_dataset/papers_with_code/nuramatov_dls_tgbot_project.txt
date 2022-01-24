# Telegram bot for style transfer
GAN being used is the official CycleGAN implementation (style_vangogh) taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
## Commands:
/start - Start the bot
## Using the bot
After you start the bot, it will ask you for the model you want to use (GAN or NST).
Then you send it the content picture, and a style image (if you chose NST) in separate messages.
After you receive the image, you can choose the model again.
## Paper links 
https://arxiv.org/abs/1508.06576 - Neural Style Transfer  
https://arxiv.org/abs/1703.10593 - CycleGAN
## Configuration
Just create a .env file in the root folder with your token in the single line like this:  
"0000000000:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

You may want to change the number of epochs per a NST request
