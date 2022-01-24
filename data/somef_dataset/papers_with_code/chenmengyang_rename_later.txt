# Obstacle Course 
This is a simple exercise to assess the fitness of a candidate for a the role of Machine Learning Engineer. There are two independent tasks that need to be completed. The first task deals with training and deploying a machine learning model, and the second deals with model monitoring.

## Task 1
Use the Iris datasets stored in the `data/` directory to train a simple machine learning model to predict the species of *Iris* given measurements of the length and width of the petal and sepal. Serialize the model and store it in the current working directory.

 Next, define two functions in `score.py` named `init()` and `run()` . The `init()` function is run once when the web service is initialized. It should return 0 if there was no error during initialization or 1 otherwise. The `run()` is executed whenever a call is made to the `/predict` endpoint. It accepts a JSON encoded string corresponding to the request schema and returns a JSON string conforming to the response schema shown below: 

 Request schema:

 ```js
 {
     "sepalWidth": Float,
     "petalWidth": Float,
     "sepalLength": Float,
     "petalLength": Float
 }
 ```

Response schema:

```js
{
    "species": "versicolor"|"virginica"|"setosa"
}
```

Refer to `app.py`, the Flask app that serves the predictions, to understand how these functions are used. In `test_score.py` write unit tests using a well-known Python testing framework to verify that the functions you defined work as they should. Ensure that the service runs locally. Next, complete the `Dockerfile`, build the Docker image, and verify that it also runs locally. 

If you found this exercise trivial, you are encouraged to augment your solution with something cool. For example, you could use JSONSchema for input validation, build out a CI/CD pipeline with Azure Pipelines, Travis  etc., or make the API more secure using JWT and TLS. Feel free to impress us! 

## Task 2
A while ago, your team deployed an Iris model into a test environment and directed a small portion of the traffic to it to test the deployment. You were sent two datasets, `predictions.csv`, which contains the predictions given by the model, and `input.csv` (attached in the email we sent you) which contains requests received by it. Analyse the two datasets, and present your conclusion about whether the deployment was successful and why/why not. 

