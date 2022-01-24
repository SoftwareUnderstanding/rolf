# Handbag GAN

In this project I will implement a DCGAN to see if I can generate handbag designs. 

To accomplish this I'm using 3,000 handbag images from Imagenet. These are sample images:

<p></p>
 
<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/sample%20image.PNG" width="400"></p>

# DCGAN

DCGANs were introduced by Alec Radford, Luke Metz and Soumith Chintala in 2016 (paper: https://arxiv.org/pdf/1511.06434.pdf). The following diagram explains the architecture of a DCGAN:

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/images/dcgan%20diagram.png" width="800"></p>

In the paper, the role of the Generator is explained as follows: "A 100 dimensional uniform distribution Z is projected to a small spatial extent convolutional representation with many feature maps. A series of four fractionally-strided convolutions then convert this high level representation into a 64 × 64 pixel image."

# Some results

After 450 epochs of training, here we can see some handbags created by our DCGAN:

<p float="left">
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image1.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image2.png" width="100" /> 
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image3.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image4.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image5.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image6.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image7.png" width="100" />
</p>

<p float="left">
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image8.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image9.png" width="100" /> 
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image10.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image46.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image12.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image13.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image14.png" width="100" />
</p>

<p float="left">
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image15.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image16.png" width="100" /> 
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image17.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image18.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image19.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image20.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image21.png" width="100" />
</p>

<p float="left">
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image22.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image23.png" width="100" /> 
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image24.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image62.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image26.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image27.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image28.png" width="100" />
</p>

<p float="left">
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image59.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image30.png" width="100" /> 
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image60.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image32.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image33.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image34.png" width="100" />
  <img src="https://github.com/prodillo/GAN_project/blob/master/images/image35.png" width="100" />
</p>

# Bonus: Google Cloud Setup

Certainly, GPU saves a lot of time training neural networks, so I had to spend some time to setup a virtual machine in Google Cloud to have access to GPU computing. Given that I didn't find a compehensive tutorial for Windows users, I will share what worked for me.

1\. To setup my virtual machine in Google Cloud, I borrowed a virtual machine image from Stanford’s Convolutional Neural Networks course that installs Anaconda, Pytorch and other useful libraries (thanks guys!). I followed this tutorial: http://cs231n.github.io/gce-tutorial/ . <p></p>
Be careful to select the number of GPUs that you need in order to have access to GPU computing. In my case, I selected 1 NVIDIA Tesla K80 GPU.

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image1-1.png"</p>
 
After finishing the setup of your virtual machine you will get an error message because you don’t have a GPU quota assigned to your virtual machine. 

To solve this, you have to go IAM & admin->Quotas in the Google Cloud console, find and select the NVIDIA K80 GPU of your corresponding zone, click “EDIT QUOTAS” and then request access to the number of GPUs that you selected previously (1 in my case).

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image1-2.png"</p>
 
In my case, it took almost 24 hours to get my quota increased. After that, you are ready to go with your virtual machine!
 
2\. Following the previous tutorial, it is important that you open the terminal:

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image2-1.png"</p>
 
and make sure to run the following command for the first time setup:
 
    $ /home/shared/setup.sh && source ~/.bashrc

3\. Install PuTTY to generate a private key: https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html

4\. Open PuTTY key generator and create a private key: 

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image4-1.png"</p>
 
Make sure to put your Google Cloud username in “Key comment”. After this, save the private key (I saved the key in a file named gcloud_instance_2.ppk)

5\. Go to the virtual machine in the Google Cloud console, make sure it is stopped, and click it:

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image5-1.png"</p>
 
Then click “EDIT”:

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image5-2.png"</p>
 
And go to SSH Keys and click “Add item” then copy and paste the key generated in the previous step: 

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image5-3.png"</p>
 
Finally, save changes:

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image5-4.png"</p>
 
6\. Download WinSCP: https://winscp.net/eng/download.php to transfer files between local and virtual machine.

To connect, use the external IP of the instance, the user name (prodillo) and in Advanced Settings->SSH-> Authenticate, select the private key file created in the previous step.

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image6-1.png"</p>
<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/gcloud_tutorial/image6-2.png"</p>
 
7\. Finally, if you need to install python libraries, open  a SSH terminal as shown in step 2 and type:

    $ sudo su root
    
    $ conda install [package name]

For example, I installed the tqdm package typing:

    $ conda install tqdm
