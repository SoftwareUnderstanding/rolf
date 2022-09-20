# FTbg
Background removal using Fourier Transform


This handy python script takes a FITS image, perform Fourier transform, and separate low- and high-spatial frequency components by a user-specified cut. Both components are then inverse FT back to image domain. It can be used to remove large-scale background/foreground emission in many astrophysical applications.

Input: original FITS image
Outputs: background and structure images

If you find this script useful, please cite our paper:
[Large-scale filaments associated with Milky Way spiral arms](http://adsabs.harvard.edu/abs/2015MNRAS.450.4043W).


![Example](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/mnras/450/4/10.1093/mnras/stv735/2/stv735figa1.jpeg?Expires=1510317012&Signature=dcIJU0AJMEwDFhO-D10ZwH6gCfx5RF5K3kfcOujYSp14aFCyddqK5HZAxCk2gDUAZi5oR6PAtq-B2EMuWNGKMDFHBZLV5yP6x-TeNeYHjtpLrSRGxk0sqYPEf2ZRgqtqX832mMdr8U8Xiam~Eb6cCbWhLimHBw-OulGHVzEW~nlGUBaux7FD9wMKjx1NYcombvBb83ouXTQrG-n-wHc9-Mo1UPEH9ffS4ysuT7a~8T0tihsZ6~hZ3xSYYhxurxNMA1h12fCW29DXL8qV9f7OAc3ZVV~MBmyrNQmBAo~o4WO5yuacmWtrnh1GRYkGpQHR4nk0xGnb3C3dZY8QDtbhMQ__&Key-Pair-Id=APKAIUCZBIA4LVPAVW3Q)
