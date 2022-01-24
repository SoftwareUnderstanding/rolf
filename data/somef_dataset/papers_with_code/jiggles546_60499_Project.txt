# 60499_Project

## Project Goals
The main goal of the project is to develop our programming skills and learn new technologies. Using our newly acquired skills we developed software that processes images and “tags” them. The images and tags are then stored in a database where the information is displayed to the end user for interpretation. This paper will explain the technologies chosen, how they work and why they were chosen. By the end you should understand the overall architecture of the project and how to make additions to it yourself.

# Back-End By Jonathan North
## Overall Architecture
The high-level architecture is very simple for this project, see figure 1.1.

### Figure 1.1
![alt text](Figure-Images/Figure1-1.jpg "The top portion represents the front-end portion of the service. The user uses a web browser to make requests to a file server to the images that belong to his account. The file server then communicates with a Microsoft SQL Cloud Database, set up in Azure to retrieve user and image information.")

The bottom portion of figure 1.1 shows the user setting up a device, this can be any device that’s connected to a camera and has access to the internet. The device takes a picture and sends that picture using a REST API request to our Web Service hosted in Microsoft’s Azure Cloud Service. After processing the image, it should be added to the database previously mentioned along with it’s tags.

## Images & Tags
After an image has been processed two objects will be created. The first being the same image created however, the image will have boxes around identified objects within an image, see figure 2.1.  Each object will have a corresponding colour. The second object after processing is the tags. To accompany the image the Web App will also create a JSON file which will provide more information of the “boxes” from the new image, see figure 2.2. For each object identified a JSON object called “tags” will be created with the following properties. “bbox” represents the four points of the “box” that surround an identified object. “label” provides the user with what object our image processing thinks it has identified. “Score” gives a value from 0 to 1 which represents the likelihood that the label is correct. 1 being the most likely and 0 being unlikely. Generally, the algorithm won’t identify objects that are far below 0.5.

### Figure 2.1
![alt text](Figure-Images/Figure2-1.jpg)

### Figure 2.2
![alt text](Figure-Images/Figure2-2.jpg)

## Web Application Service
To start, we will discuss the Web App hosted on Microsoft’s Azure. If you’re not familiar with Microsoft’s cloud based platform a good starting point is the following link https://docs.microsoft.com/en-us/azure/architecture/cloud-adoption/getting-started/what-isazure.
One of the reasons for using Azure over Amazon’s AWS is that Microsoft has a verbose library of deep learning algorithms which allows us to easily train our programs to recognize objects then process images quickly. The library is called Microsoft Cognitive Toolkit or CNTK (https://www.microsoft.com/en-us/cognitive-toolkit/). There’re several different algorithms provided by the library however, we will be using Faster-RCNN. If you would like to run FasterRCNN on your PC you can do so by following this guide (https://docs.microsoft.com/enus/cognitive-toolkit/Object-Detection-using-Faster-R-CNN). You will need Python 3.5, PIP and Anaconda as your Python Environment. This is a good way to see how the library itself works before we get into implementing it in a Web Application. By using Faster-RCNN we greatly improve the speed at which we can identify objects which is going to allow us to make our software more scalable. If you would like to know more about the algorithm the published paper can be found here (https://arxiv.org/pdf/1506.01497.pdf)


### Steps to deploy Web Application
1.    Now that you know how CNTK and Faster-RCNN work now we can get into the Web
Application. To start you’ll need to create an Azure subscription at
(https://azure.microsoft.com/en-us/free/). If you’re a student, then you’ll get $200 worth of credits which will be more than enough to pay for the base service.  First you should download the content from the repository () and unzip to some folder on your computer. For this you'll only need the files and folders from the "Back-End" folder.

2.    Step 2 requires you to set up Python, it’s environment and the dependencies by CNTK on your local Windows machine. Luckily Microsoft provides us with the following tutorial
(https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-binary-script)

3.    Next open Anaconda and run the following command pip install azure-cli
Azure CLI is Microsoft’s command-line tool for managing Azure resources. We will be using it to deploy and manage our web application

4.    Now run the following command ```az login```. You will be prompted for your Azure account credentials, however, keep in mind that sometimes it takes a few minutes for the prompt to appear.

5.    Now we’re going to set up variable names in our environment as they’ll be used throughout the next couple of steps. Although not required it is highly recommended as this will avoid errors and allow you to copy and paste the commands with no alterations. Run the following commands:
```
set uname=[username]
set pass=[password]
set appn = [web app name]
set resgname = [resource group name]
```
Replace all instances of [] with whatever you wish. I recommend making your resource group name [web app name]_resource_group. I.e. If
appn=60499Project then resgname=60499Project_Resource_Group

6. ```az webapp deployment user set –user-name %uname% --password %pass%``` This command will allow us to deploy our code to Azure

7. Next lets set up a resource group. ```az group create –location eastus –name %rgname%``` Resource groups allows us to manage all aspects of a web app as one entity. For more information please see [here](https://docs.microsoft.com/en-us/azure/architecture/cloud-adoption/getting-started/azure-resource-access)

8.    Now create an Azure App Service Plan and an Azure Web App. For more information about these services please see here and here. Run the following commands:
```
    Az appservice plan create –name %appn% --resource-group %rgname% --sku S1
    Az webapp create –name %appn% --resource-group %rgname% --plan %apppn%
```

9.    By default, web apps only support Python 2.7 & 3.4 but we require 3.5. Therefore, we need to use an extension in our web application environment. To do so go to the Azure Portal at http://www.portal.azure.com and log in using the credentials during step 1. In the left side bar choose App Services and select the web app that you’ve created this will be the main page to configure and monitor your web app. You’ll see a new column with the first entry being “Overview”. In the search bar above it type the following “extension” and select the option “Extensions”. There should be no extensions installed at this time. Choose Add, then under choose extension select “Python 3.5.4 x64”. Accept terms and conditions then you OK. You should get a message that it’s being installed and may take a few minutes.

10.  Now in your Python Environment run the following command
```
az webapp deployment source config-local-git –name %appn% --resource-group %rgname% --query url –output tsv
```
This command will output the URL of your web application. It should look something like this
```https://xxx@yyyyyyyy.scm.azurewebsites.net/zzzzzz.git``` Make sure to copy this website down as it’ll be used in a later step.

11.  Run the following commands
```
Git init
Git remote add azure https://xxx@yyyyyyyy.scm.azurewebsites.net/zzzzzz.git
Git add -A
Git commit -m “init”
Git push azure master
```
There’s a script in the repo (deploy.cmd) that will install all the required dependencies from the requirements.txt file.

12.  You’re all done.  In the Azure Portal restart your application. Wait 5 minutes and you should see the following web page.

## How does the Web App work?
The web application is written in Python and uses the microframework Flask. We chose Flask as it’s less bloated than Django and therefore can be picked up much more quickly. Since we didn’t want to spend a lot of time learning a framework it was the obvious choice. To look at the files you can either look at the repository you download, or you can view the files from the Web App itself. In the Azure Portal and the App Service you should be able to find an option called “Advanced Tools”, if not use the search bar. Select “Go”, this should bring up a new tab which gives some details about the environment. Select the dropdown “Debug Console” then CMD. This will pull up the command prompt where your web app is being held. Feel free to use the command prompt or the file navigator on top to go through the files. Under the directory D:\home\site\wwwroot\ you will file all the files from the repository you downloaded. The most important files are “app.py”, “config.py”,”evaluate.py” and “web.config”.

### Important Files

#### Config.py
This file is very important as it sets the variable names and values that will be used throughout our program. These are variables the will set paths of certain files, file names, etc. Be careful you alter variables from this file as can be used throughout the execution of the program.


#### App.py
This is the file that gets called on startup. You can see with the “@app.route…” calls we specify what actions to take with each URL. Depending on the URL either an image or JSON will be returned. If you wish to add another route, simply use the routes as a template.
***Double check to make sure that that the “os.environ[‘PATH’]” variable is set correctly. Python 2.7 is installed automatically and therefore we need to add the path of the Python version that we added as an extension. The path should be “D:\home\python354x64;”*****

#### Evaluate.py
This is the main driving force behind the image classifier. It uses several functions and several variables from config.py to perform its task. It also uses the helper functions from cntk_helpers.py, plot_helpers.py and scripts from the utils folder. Keep in mind that most are copied from the official CNTK repository on Github.

#### Webconfig.py
Standard web config file. Just make sure that the paths are set correctly.

### Tying it all together

When the web application receives a request it will first go in the App.py file and call the appropriate method based on the url used (/returntags or /returnimages). The method will then call evaluateImage() found in the file Evaluate.py. This is the method that will call most other methods and be the driving force behind evaluating an image. Once completed the method evaluateImage will output either json or a binary image (depending on the url called). The we come back to App.py which will return the output.

## Q&A:
Q: What is a CNTK Model?
A: It’s the file that’s used to train our deep learning algorithm (Faster R-CNN).

Q: What can the current model detect?
A: The current model can detect the following: lamp, toilet, towel, sink, bathtub, tap, bed, pillow and faucet

Q: How can I train my own model?
A: Follow the instructions on this link. Note that you may need to scroll down a little.

Q: Can I use somebody else’s model?
A: There’s no reason why you can’t as long as it’s trained for FASTER R-CNN. Here’s an example of a model you can use. Once you have a model add it to the CNTKMODELS folder. If you wish to only use that model remove the current model and add yours using the exact same name. You can have both files in the folder however you’ll need to make an entry in config.py with the path and name and make configuration changes to all the files mentioned in this document.

## Room for improvement
- The first place to improve would be to automatically add the image and tags to a database rather than just returning the information
- Add a component to the website to allow users to add models


# Front-End By Rahul Sharma
## Technologies
- PHP
- JavaScript
- HTML
- CSS

## Flow

![alt text](Figure-Images/Figure3-1.jpg)

Once the program is executed, the user goes to the homepage. From here, they can either login
or register for an account. If the user registers for an account, they must fill out the form
correctly. The data then gets stored in the MSSQL DB under the User_Account_Info_60499 table.

  The user then logs in with valid credentials. The code in login.php will check if the credentials
  are valid, then take you to the portal (portal.php). If the credentials are invalid, the program will show an
  error message. Once you login, your full name, username, password, and accountID are stored in local
  storage for later use.

  In the portal, the user views a logged-in navigation bar (logged_in_theme.php). The user can
  view details of the captured image (coordinates of where it was taken, item name, match percent).
  The code will select the information stored in the JSONTags_60499 table based on your accountID.
  It returns JSON data which is parsed and displayed into a table

  In the portal, the user can view the images that are captured. The images are stored in the
  Account_Images_60499 table. The code selects the images based on accountID. The images are
  then displayed into a table.

  In the logged-in navigation bar, the user can view their account information. It shows their
  username and full name in table. This is displayed from local storage. It also has an option
  to change the password (change_password.php)




## Screenshots

- Home page for the app
 ![alt text](Figure-Images/Figure3-2.jpg)


- Blank register form

  ![alt text](Figure-Images/Figure3-3.jpg)
- The form validates input for full name, username, password, confirm password. The screenshot
for invalid password is below.  
  ![alt text](Figure-Images/Figure3-4.jpg)
- The form checks if the username exists in the database. If it does, it will display an error message
denoted by the screenshot below
  ![alt text](Figure-Images/Figure3-5.jpg)

- After a successful registration, there is a message telling the user that the registration was successful.
This screen only occurs after the registration.
  ![alt text](Figure-Images/Figure3-6.jpg)

- If the login button is pressed on the index page or the navigation bar, the user is taken to the page below.
If the login is successful, the user is redirected to the portal, if not, there will be an error message displayed.
  ![alt text](Figure-Images/Figure3-7.jpg)

- The portal includes information regarding the details of the image and the actual image. It also displays the
logged-in navigation bar.

  ![alt text](Figure-Images/Figure3-8.jpg)


- If the user pressed the View Details button, they are taken to the page below. This page shows the information regarding
the coordinates, item name and score.

  ![alt text](Figure-Images/Figure3-9.jpg)

- If the user pressed the View Images button, they are taken to the page below. This page shows the images taken by the user.

  ![alt text](Figure-Images/Figure3-10.jpg)


- The user can check his/her account information from the My Account link in the navigation bar. This page allows the user
to view his/her information (username and full name).


  ![alt text](Figure-Images/Figure3-11.jpg)

- The user can change there password as shown below.

  ![alt text](Figure-Images/Figure3-12.jpg)

- After a successful change, the page below is shown indicating to the user the password has been change. Else, it will show an error message.

  ![alt text](Figure-Images/Figure3-13.jpg)



## How to config
In connect.php, change the values of server, database, user, and password to make it work with your database.
The code accesses the database by using pdo(lightweight interface for accessing databases in PHP) since we used the MSSQL server.
This implementation ran the code on the local php server on your computer.


## Room for improvement
- Connect the front-end with the back-end. Right now, it just takes information from the database.
- Add Security to passwords 
