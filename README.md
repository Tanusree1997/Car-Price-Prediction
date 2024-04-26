**Car Price Prediction**

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/e2aea0f1-40c6-4635-8461-7f590ce7baf1)

 
This project aims to predict car prices based on various technical and non-technical characteristics of the vehicle. Utilizing Python libraries such as Pandas, NumPy, Matplotlib, and Statsmodels, a linear regression model has been developed for car price prediction. Through this analysis, significant predictors influencing car prices have been identified.

**Dataset Used**

This dataset, sourced from Kaggle, encompasses 26 columns, encompassing various attributes and the price of different cars. These attributes include car ID, symboling, car name, fuel type, aspiration, number of doors, drive wheel, engine location, wheelbase, car length, car width, car height, curb weight, engine type, cylinder number, engine size, fuel system, bore ratio, stroke, compression ratio, horsepower, peak rpm, city mpg, and highway mpg.

Dataset Link: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data

Tool used: Google Colab

**Process**

Step 1: Importing the python libraries, and uploading the CSV file

![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/ad8ff74e-977a-4ae6-9a5e-407826bfec47)

Step 2: Understanding the data and checking for null values
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/36a900fe-4902-4a48-b6f1-ec061909f47e)

 ![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/a538024d-8ae3-49bf-ae72-475cb2a21cef)

It's important to note that the dataset comprises 10 categorical fields with object data type and 16 numeric fields with integer and float data type. Upon inspection, it's observed that there are no null values present in the data.

Step 3: Checking for duplicate rows
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/7853b4de-efda-4656-b72c-7a80df66c792)

It's evident that there are no duplicate values within the dataset. 

Given that our objective involves constructing a linear regression model, it's imperative to convert categorical variables into numeric codes. In other words, we need to create dummy variables to incorporate them into the regression analysis.

Step 4: Creating Dummy variables by converting categorical variables to numeric codes
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/9e330686-5ca0-4758-852b-c9539956e9ba)

I've employed the same approach to generate dummy variables for the remaining categorical variables. It's important to note that this method encompasses a range of numerical values starting from zero for each value in a categorical field.  It consistently assigns zero to one value (e.g., diesel in ‘fueltype’ field) to prevent encountering the dummy variable trap.
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/e4da5740-d0e3-4fbd-b14d-b2c18e393b91)

Step 5: Checking for multicollinearity in the data
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/3f606236-ff87-4dba-b9b3-b20fe314eac3)
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/a9f61645-1425-4f13-8047-6a59d712992e)

Upon observing the correlation table, it became apparent that certain fields exhibit notably high VIF scores, which can pose challenges for the model. In response, several of these fields were dropped to mitigate the issue of multicollinearity. By examining the correlation coefficients and considering their significance within the model, 7 fields were identified for exclusion. Consequently, the final set of independent variables comprises the following:
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/3b05d248-dac3-430c-aa5d-12df6b5f6db3)

Step 5: Dividing the data into test and train data
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/f91f1796-da37-4ad7-8255-759d9c205ade)

Step 6: Fitting the model using train data
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/e05d053c-a5ca-4c00-921a-866103450202)

Step 7: Executing the prediction of car prices from the model
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/df993418-7584-4e67-817c-df9b64b502b8)

Step 8: Checking the goodness of fit (of the model) 
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/e2bbd5e7-5ec7-417e-a671-d4d28ecb3208)
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/93178042-55bd-4574-a39a-621d2b518534)

The R-squared value approximates 0.8856, indicating that the independent variables can explain approximately 89% of the variability observed in car prices, the dependent variable. This suggests a favorable fit of the model.

Step 9: Predicting the car prices based on X_test to check model validity 
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/084af484-be54-4a67-9fd8-14ebafd49275)

The high goodness of fit observed for the test data substantiates the validity of the model.
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/7cf9cf6a-7158-4717-b51d-0cac001e49c1)

Step 10: Finding out the statistical significance of the model
![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/b705ae1b-d1a5-406d-a87f-2ada756b0cd5)

The rule for assessing the statistical significance of each coefficient and intercept is as follows:

1.	If the p-value is greater than 0.05 (considering a 95% confidence interval), we cannot reject the null hypothesis, indicating no statistically significant relationship between the independent variable and the dependent variable.
2.	If the p-value is less than 0.05, we reject the null hypothesis and accept the alternative, indicating a statistically significant relationship between the independent variable and the dependent variable.
Based on this rule, the following variables demonstrate statistically significant influence on car price determination: ‘CarName’, ‘doornumber’, ‘carbody’, ‘drivewheel’, ‘enginelocation’, ‘carheight’, ‘enginesize’, ‘stroke’, ‘compressionratio’, ‘peakrpm’, and ‘highwaympg’. Conversely, ‘symboling’, ‘aspiration’, ‘enginetype’, ‘fuelsystem’, ‘cylindernumber’, and ‘boreratio’ are not statistically significant determinants of car price.
It's essential to consider the mapping performed during data preprocessing (step 4) when we are trying to interpret the coefficients of the categorical variables. For instance, "enginelocation" with "front" mapped as 0 and "rear" mapped as 1 implies that cars with "enginelocation" = 'rear' tend to have a higher price (positive coefficient) than those with "enginelocation" = 'front'.

Step 10: Exploratory Data Analysis
 ![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/2dba8700-0e80-43a5-b5db-818e52085b69)

 ![image](https://github.com/Tanusree1997/Car-Price-Prediction/assets/164666871/ea534ebf-27b5-4085-a99f-70e037377640)

I have used the ANOVA table and plots to determine which factors are responsible for a higher car price. 

1.	There are total 143 types of cars in the data. The top 10 car names with high prices are 'alfa-romero Quadrifoglio', 'alfa-romero giulia', 'alfa-romero stelvio', 'audi 100 ls', 'audi 100ls', 'audi 4000', 'audi 5000', 'audi 5000s (diesel)', 'audi fox', 'bmw 320i'. On the other hand, 'volkswagen rabbit custom', 'volkswagen super beetle', 'volkswagen type 3', 'volvo 144ea', 'volvo 145e (sw)', 'volvo 244dl', 'volvo 245', 'volvo 246', 'volvo 264gl', 'volvo diesel', 'vw dasher' are the cars with least prices. 
2.	Based on car body, the highest car prices have been seen in the case of ‘wagon’ followed by ‘sedan’, ‘hatchback’, ‘hardtop’, and ‘convertible’. 
3.	Consumer preference leans towards two-door cars over four-door options.
4.	The popularity order among drive types is four-wheel drive, followed by front-wheel drive and rear-wheel drive. Additionally, consumers show a preference for cars with rear engines and taller car heights.
5.	Clean technology, high speed, and fuel efficiency are favored by consumers, leading to higher prices for cars with lower stroke and higher compression ratios.
6.	Consumers prioritize speed and efficiency, resulting in higher prices for cars with higher RPM, horsepower, and engine size. Conversely, vehicles with higher mileage per gallon (both highway and city) tend to be priced lower.

Conclusion

In this project, the car price prediction has been done using a MLRM. The predicted car prices closely align with the actual car prices, indicating the model's accuracy exceeds 80%. We have also discussed the consumers’ car preferences based on statistically significant car features. Nonetheless, further enhancements to the model are possible with the addition of more data points.
