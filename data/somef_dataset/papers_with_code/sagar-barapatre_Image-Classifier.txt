# MERN CRUD Blogging App
# [Live](https://mern-crud-assignment.herokuapp.com)
### Front-End - React + Redux
### Back-End - Node.js(Express.js) & MongoDB

## Tech Stack

#### Front-end

* The front-end client is built as a simple-page-application using React and Redux (for middlewares and reducers).
* React-Router is used for navigation.
* Redux-Thunk is used for processing asynchronous requests.
* Bootstrap 4 is used for page styling.

#### Back-end

* The back-end server is built with Express.js and Node.js in MVC pattern, which provides completed REST APIs for data interaction.
* Passport.js is used as an authentication middleware in the sever.
* JSON Web Token (JWT) is used for signing in user and making authenticated requests.

#### Database

* MongoDB is used as the back-end database, which include different data models/schemas (i.e., User, Post and Comment).
* Mongoose is used to access the MongoDB for CRUD actions (create, read, update and delete).

## Project Outline

1. Create Blog Schema
    Blog (String)
    Published (Bool) ---> For users to see unpublished vs published post
    Comments (Object)
    User (Object)

2. Create Comments Schema
    Comment (String)
    User (Object)
    Date

3. Ceate User Schema:
    Name(String)
    Email (string)
    Password 
    isAdmin (String)

4. Sinup/Logout/Login User

5. View Blogs without signin

6. Write Blog with signin

7. Allow admin to determine which blogs to be published or not

8. Allow other users to comment on the blog

## Set-Up Project in your machine

1. Fork the repo and clone it.
2. Create a new branch.
3. Make sure you have `npm` Node.js installed in your system. MongoAtlas is used, so no need for local MongoDB setup.
4. MongoAtlas Setup
https://www.youtube.com/watch?v=7CqJlxBYj-M&feature=youtu.be&t=293
Set up your .env file and paste in the URI that you get from following the instructions in the video above. Also set token secret to anything, it is used for jwt authentication.

```
MONGO_ATLAS_KEY=mongodb+srv://<dbUser>:<password>@cluster0-m5jph.gcp.mongodb.net/test?retryWrites=true&w=majority
TOKEN_SECRET=your secret key
```
You need to remember to paste in the <dbUser> and <password>. Do NOT share it publicly, and do NOT include the .env file in commits.

5. [Only once] Run (from the root) `npm install` and `cd client && npm install`.
6. Un-comment line 30 in app.js (root folder) and comment line 24 & line 35-37.(This is to run app locally, please suggest better way if you know).
7. Open two terminal windows (one for running Server and other for the UI).
8. To run server, from root folder run `nodemon start` and to run client, go to client directory and run `npm start`.
9. Go to `http://localhost:3000` to see the application running.

## Deploying to Heroku[]()

1. To deploy your application to Heroku, you must have a Heroku account.
2. Go to [their page](https://www.heroku.com/) to create an account. Then go through their documention on how to create a Heroku app. Also check out the [documentation](https://devcenter.heroku.com/articles/heroku-cli) on Heroku CLI.

### Create a Heroku App
First, login to Heroku:

```
heroku login
```
This will redirect you to a URL in the browser where you can log in. Once you're finished you can continue in the terminal.

In the same React project directory, run the following:

```
heroku create
```

This will create a Heroku application and also give you the URL to access the application.

### Configure package.json

Heroku uses your package.json file to know which scripts to run and which dependencies to install for your project to run successfully.

In your `package.json` file, add the following:

```
{
    ...
    "scripts": {
        ...
        "start": "node backend/server.js",
        "heroku-postbuild": "NPM_CONFIG_PRODUCTION=false npm install npm && run build"
    },
    ...
    "engines": {
        "node": "10.16.0"
    }
}
```
Heroku runs a post build, which as you can see installs your dependencies and runs a build of your React project. Then it starts your project with the start script which basically starts your server. After that, your project should work fine.

`engines` specifies the versions of engines like `node` and `npm` to install.

### Push to Heroku

```
git push heroku master
```

This pushes your code to Heroku. Remember to include unnecessary files in `.gitignore`.

After few seconds your site will be ready. If there are any errors, you can check your terminal or go to your dashboard in the browser to view the build logs.

Now you can preview your site at the URL Heroku sent when you ran `heroku create`.



