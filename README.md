**Car Price Prediction**

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/e2aea0f1-40c6-4635-8461-7f590ce7baf1)

 
This project aims to predict car prices based on various technical and non-technical characteristics of the vehicle. Utilizing Python libraries such as Pandas, NumPy, Matplotlib, and Scikit-learn, a model has been developed for car price prediction. Through this analysis, significant predictors influencing car prices have been identified.

**Dataset Used**

This dataset, sourced from Kaggle, contains 26 columns, encompassing various attributes and the price of different cars. These attributes include car ID, symboling, car name, fuel type, aspiration, number of doors, drive wheel, engine location, wheelbase, car length, car width, car height, curb weight, engine type, cylinder number, engine size, fuel system, bore ratio, stroke, compression ratio, horsepower, peak rpm, city mpg, and highway mpg.

**Dataset Link**: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data 

Tool used: Google Colab

**Process** 

Step 1: Importing the python libraries, and uploading the CSV file


![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/699c2483-5b4e-4727-adc3-d3d98adb8dc2)


**Step 2: Data Exploration**
2.1. Checking the data types of the fields and looking for missing values (if any)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/7a384388-6971-4bc6-95fb-d2940dd27de1)


It's important to note that the dataset comprises 10 categorical fields with object data type and 16 numeric fields with integer and float data type. Upon inspection, it's observed that there are no null values present in the data.

2.2. Checking for duplicate rows

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/7853b4de-efda-4656-b72c-7a80df66c792)

It's evident that there are no duplicate values within the dataset. 

2.3. Checking the descriptive statistics

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/f9e46252-9877-4bca-b11e-372139b66341)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/a1950988-b62a-4105-a894-b6c94e84886a)

Note that some categorical fields have more than 2 unique values which means we may need to use dummy variables in the model. Also note that, the car name column has 147 unique values which can be reduced if we consider the car brand names only instead of car brand with model names.

**Step 3: Feature Engineering **

3.1. Creating the car_brand column and correcting the spelling mistakes in the data

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/b422fc89-bf5c-406b-a9d1-aca15e80fdb0)

3.2. Dropping the unnecessary columns

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/dda15945-30f4-458f-bc74-34860aa77f15)

**Step 4: Data Visualization **

4.1. Using GroupBy method to visualize average car price with respect to different features 

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/0db17477-9c41-46fa-bf51-daace3a52753)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/f25ccd61-0a36-422a-a88e-eabbe4db9bb8)

Similarly we get, 
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/c1711595-a2e5-40e9-9ca3-3ccf667d80a6)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/622657d0-a578-4011-bcbc-c0f12986f25e)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/c3845257-4f34-4017-851a-6679c8756cc2)

**Step 5: Feature Selection **
5.1. Dividing the numerical and categorical columns because we are going to use different methods for selection of numerical and categorical features:

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/47daca7e-f7b7-4288-bd28-4c8197838a83)

5.2. Using a Heatmap and correlation scores to understand which numerical features are highly correlated with the car price:

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/9321c714-efed-4684-a341-ea5a871def6c)

We dropped the features with negligible correlation scores (with price) from the dataframe.

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/911a86a6-f522-4bdf-9a77-f6b01e1aa191)

5.3. Creating a new numerical feature list (without price column) to check for multicollinearity:
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/c56aedc8-d9c1-487e-af9a-16ae14250d2f)

5.4. Using Variable Influence Factor to check for multicollinearity:

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/da4a29a0-ab55-408d-bb71-1369e5899417)

Features with high VIF scores need to be dropped to solve the problem of multicollinearity.

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/e874ebfa-5603-4f9b-a075-7396b64dee3e)

5.5. Using ANOVA test to identify the statistically significant categorical features for car price prediction:

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/398ca34a-a405-4ed9-b59f-1d73c1ca5183)

If the P-value is less than 0.05 then we reject the null hypothesis that there is no significant relationship between the car price and the individual feature/independent variable. This means, we will drop the features with a higher P-value (than 0.05) from the Dataframe.

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/2ba231e7-5466-4d21-a080-8df4a6f456ee)

**Step 6: Dummy Variables**

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/a203b2f1-2ad1-478b-8f9d-46f0823d34ef)

**Step 7: Train-Test split and modelling**

7.1. Defining the X and y followed by split

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/e927e401-de80-49bb-a59e-87b36db3439e)

7.2. Fitting the data in Linear Regression model and predicting car prices using train data

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/b47f7f96-e0f7-4f9c-981c-d2939cc59873)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/eb9fbbcd-78f7-46ac-be9b-7973e3ac226e)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/06addf07-c38c-4311-bae0-158beb1eac61)

Predicting the car prices based on test data using the lr model

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/6b455fa2-b496-46e0-96ec-25eed0eb3a59)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/ee29772e-6d81-402e-9fc4-e816a8850941)

This model seems to work fine.

7.3. Fitting the data in Decision Tree Regressor and Lasso models:

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/25e17769-c048-4725-ba57-e953209d8d9b)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/b7182ca0-c664-43ae-b813-b10147d101a8)

The matrices (comparing dtr, lr and lasso models) shows that the Decision Tree Regressor works better for the car price prediction.

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/a1ffae47-3ee5-4536-aca4-47f6adc7e189)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/1bb2f3d1-fc45-4c66-bbc4-6f086b9ce5bf)

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/facc514c-918a-49f2-95a7-8c047cdaef2f)

If we compare the scatter plots we can understand that the decision Tree Regression model is best suited for the car price prediction. 

**Findings**

1.	The top 5 cars based on average car prices are of Mitsubishi, Mazda, Volvo, Porsche, and Mercury.
2.	The average price of the cars with sedan and hatchback car body and a turbo engine is higher than others
3.	The average price of the cars with four-wheel drive is higher than the same with front wheel and rear wheel. The cars with three or four cylinders and four doors are priced higher on an average.
4.	The average price of rear engine cars is higher than front engine cars.
5.	The higher the Car Length, curb weight, and Horse Power the higher the car prices. Therefore, the higher the highway_mpg the lower the car prices.
   
**Conclusion**

In this project, the car price prediction has been done using a Decision Tree Regressor model. The predicted car prices closely align with the actual car prices, indicated by the model's R squared value exceeding 80%. We have also discussed the consumers’ car preferences based on statistically significant car features. Nonetheless, further enhancements to the model are possible with the addition of more data points.

