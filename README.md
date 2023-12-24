# Housing Price Predictor

![app_main](readme_images/app_main.png)

Housing Price Predictor is a project developed to predict housing price in Ames, Iowa, USA. This is a fifth project from Code Institute's Diploma in Full Stack Software Development in Predictive Analytics. The code is written in Python using Jupyter notebook with Streamlit as a dashboard development environment. The project is deployed using Heroku. 

Live app can be found [here](https://pp5-housing-price-04711200f028.herokuapp.com/) 
(*Ctrl or Cmd + mouse click to open on a new tab*)


## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace. 
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|


## Business Requirements

The client has received an inheritance from a deceased great-grandfather, and needs help if the client could maximize the sales price for the inherited properties. Also the client is interested in predicting the sale price from any house in Ames, Iowa in case of future property ownership in that area.

We are using a public dataset with house prices for Ames, Iowa that the client provided. 

* BR1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* BR2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.


## Hypothesis and how to validate?

* From the 23 variables we needed to collect top 10 variables that correlates the best with out target variable, `SalePrice`. To do this, we have conducted Spearman, Pearson and Power Predictive Score (PPS) tests.
* Top 10 variables were, 
```
'GarageArea', 'GarageYrBlt', 'GrLivArea', 'KitchenQual', 'MasVnrArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd'
```
* PPS study showed below 2 variables to have a correlation with the target variable.
```
'KitchenQual', 'OverallQual'
```
* Since those two variables are also included in the top 10 selection from Spearman and Pearson study, we have finalised that above top 10 variables to have the highest correlation with the SalePrice in creating a prediction ML model. 

* With the given top 10 variables, Exploratory data analysis (EDA) was followed to examine the exact relationship with the SalePrice to derive the hypothesis. 

### Hypothesis

* Size matters: 1stFlrSF, GarageArea, GrLivArea, TotalBsmtSF shows that generally when the size is bigger, the sale price grows higher.

   * We can make our hypothesis 1 as ***The size of a house positively correlates with the sale price***

* Remodeling year matters: YearRemodAdd shows when there was a recent remodel with the house, the sale price is higher.

   * We can make our hypothesis 2 as ***Year of the house remodeling positively correlates with the sale price***

* Quality matters: KitchenQual and OverallQual shows that houses with higher quality have a higher sale price.
   * We can make our hypothesis 3 as ***The quality of the house positively correlates with the sale price***


## The rationale to map the business requirements to the Data Visualisations and ML tasks

* **BR1 - The client is interested in discovering how the house attributes correlate with the sale price.**
   
   * We conduct a ***data visualization and correlation study*** to fulfill the business requirement 1. 
   * For correlation study, we use Spearman and Pearson study to measure the magnitude and the direction between the variables and sale price.
   * The inspection was done with various plots to visualize the correlation. 
   * Details of the study could be found at [Data Analysis Notebook](https://github.com/choyoon88/pp5-housing-issue/blob/1d0feda00aab6bd06dbb6e3ca1954fa4e6e18ed3/jupyter_notebooks/03%20-%20Data-Analysis.ipynb)

* **BR2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.**  

   * Since the client is interested in predicting continuous value, we use ***regression machine learning*** to meet the second business requirement. 
   * After we have collected the 10 most correlated variables to our sale price, we conducted a Grid Search CV to check which regression model should best fit to our project. 
   * `GradientBoostingRegressor` showed the best performance amount the regressor studies with mean score 0.8, and train and test set with each 0.86 and 0.77 R2 Score. 
   * With Gradient Boosing Regressor, the 4 most best variables that correlated to sale price was 'OverallQual', 'TotalBsmtSF', 'GarageArea', '2ndFlrSF'
   * With the selected regressor and the best correlated variables, we predicted the sale price for the 4 inherited houses also created a widget that could predict the sale price by inputting the values. 
   * Details of the study could be found at [Modelling and Evaluation Notebook](https://github.com/frankiesanjana/housing-price-predictor/blob/f6ce8dd8a4334a05213efaf45d96bd102e85cb7d/jupyter_notebooks/05-modelling-evaluation.ipynb)


## ML Business Case

* Machine Learning model is trained to meet the second business model requesting to predict a sale price. 

   * The model should predict the sale price of the client's four houses.
   * Also it should allow to predict the sale price of other houses in Ames, Iowa with the given attributes of the house. 

* The regression model to use is *linear regression* since we are predicting a value from several continuous variables. 
* The ideal outcome is to provide the client with an insight of house attributes that could maximise the sale price.
* We have agreed with the client to have the below matrix:
   
   * R2 score should be at least 0.75
   * The model is considered a failure if the R2 score did not reach at least 0.75. In such case, extra modelling evaluation is needed to find and develop the model. 

* The output is defined as a continuous value for sale price in US dollar. 

   * We will present the predicted sale price for the client's four inherited house and also the summed amount for the price. 
   * We also present a live widget that could predict the sale price by entering values of the house attributes. 

* Our client has an excellent understanding and knowledge of property prices in Belgium.

   * Since the client has some background understanding on housing market, the client will try to use their own knowledge first when predicting the sale price of the inherited houses and the prices in Ames, Iowa. 
   * Although the background knowledge could be helpful in some ways, this could lead to an inaccurate prediction considering that the housing markets have different circumstances across the area and/or country. 
   * Therefore, we proceed with the development of a machine learning model. 

* The training data are a public dataset with house prices in Ames, Iowa. Dataset is provided by the client.

   * Dataset has 1460 house data with 24 features including sale price. 
   * Some variables had too many missing data. So we have dropped 2 variables which are `EnclosedPorch` and `WoodDeckSF`
   * Our target variable is `SalePrice`

## Dashboard Design

### Page 1: Quick Project Summary

<details>
<summary>Project terms and Jargon</summary>

- **Sale price** is the current market price of the house with various attributes.
- **Inherited house** is the house that the client inherited from their grandparents.

</details>

<details>
<summary>Project Dataset</summary>

- The project dataset comes from housing price database in [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data) created by Code Institute.
- The data represents the housing price in Ames, Iowa, USA with 23 aspects of the house such as the size of the house, built year etc. The total number of houses is 1460.
</details>

<details>
<summary>Business Requirements</summary>

The project has two major business requirements.

* BR1: The client is interested in discovering the most relevant variable that correlates with the sale price.
* BR2: The client wants to have a predicting model of the 4 inherited houses, as well as any other houses in Ames, Iowa.
</details>


### Page 2: House Price Correlation Study (BR1)

This page shows,

* Business requirement that was answered by correlation study. 
* Checkbox for inspecting the dataset
* Features with the highest correlation to the sale price
* Checkbox for displaying plots of sale price and each of the variables that have strong correlation. 

<details>
<summary>Spearman Correlation</summary>

![spearman](readme_images/spearman.png)
</details>

<details>
<summary>Pearson Correlation</summary>

![pearson](readme_images/pearson.png)
</details>

<details>
<summary>PPS</summary>

![pps](readme_images/heatmap.png)
</details>

* Summary of the correlation plots
* Checkbox for displaying plots between each variables and the sale price. 
   * Boxplot for categorical variables
   * LMplot for numerical variables
   * Line plot for time variables

<details>
<summary>Data visualisation plots</summary>

![1stlfrsf](readme_images/lmplot_1stflrsf.png)
![garagearea](readme_images/lmplot_garagearea.png)
![garageyrblt](readme_images/lineplot_garageyrblt.png)
![grlivarea](readme_images/lmplot_grlivarea.png)
![kitchenqual](readme_images/boxplot_kitchenqual.png)
![masvnrarea](readme_images/lmplot_masvnrarea.png)
![overallqual](readme_images/boxplot_overallqual.png)
![totalbsmtsf](readme_images/lmplot_totalbsmtsf.png)
![yearbuilt](readme_images/lineplot_yearbuilt.png)
![yearremodadd](readme_images/lineplot_yearremodadd.png)

</details>

### Page 3: Project Hypothesis and Validation

This page contains the hypothesis we can get from data analying. 

![hypo](readme_images/hypothesis.png)


### Page 4: Predict House Price (BR2)

This page contains,

* Answers for the second business requirement
* BR2 part 1: Predict the sale price of client's four inherited houses.

   * Shows a raw dataset for the four inherited houses.
   * Display the predicted sale price for each of the four inherited houses and the summed price.

* BR2 part 2: Predict the sale price of any other house in Ames, Iowa.

   * Widget that the user can input feature values and a button to predict the sale price with the input values.
   * The user can get a predicted sale price by pressing the 'Predict Sale Price' button after entering the relevant values on the widget. 

### Page 5: ML: House Price Predictor

This page contains, 

* Conclusion of the pipeline training
* Selected regressor and the pipeline step
* Final best features and the importance plot for the best features
* Conclusion of the pipeline performance 
* Model evaluation of train and test set
* Regression Performance Plot


## Unfixed Bugs
* As far as to my knowledge, there is no unfixed bugs. 


## Testing
- Conducted Python code testing through [CI Python Linter](https://pep8ci.herokuapp.com/#)
- Several errors popped up such as E501 line too long (80 > 79 characters), W291 trailing whitespace, E128 continuation line under-indented for visual indent, E251 unexpected spaces around keyword / parameter equals and so on. 
- Most of them were not crucial that could hinder the code from running, but I have fixed them for better visialisation.
- See the below image for examples.
- After fixing the mentioned issues, all Python files are now clean with no issues. 

![pytest1](readme_images/testing_cor_study.png)

![pytest](readme_images/testing_predictor.png)


## Deployment

Deployment was done through Heroku. 

### Steps

Before deploying on Heroku, there are files that needed to be included on my repository.

1. Create `setup.sh` file (This was already setup by CI's template)
   1. `setup.sh` file should contain streamlit configuration requirements to be used on Heroku. This will be the same server configuration code for every Streamlit app I deploy on Heroku.
2. Create a `procfile`.
   1. Should contain,
   ```
   web: sh setup.sh && streamlit run app.py
   ```
   
3. Create a `runtime.txt` file
   1. Sets the python environment so that we could reduce conflicts from development to production. Should contain,
   ```
   python-3.8.17
   ```
4. Go to Heroku, and log in.
5. Create a new app.
6. At the Deploy tab, select GitHub as the deployment method.
7. Select my repository name and click Search. Once it is found, click Connect.

Since we will stick with the python 3.8.12 we do the following steps to avoid error message while deploying.

8. In dashboard.heroku.com click on Account Settings 
9. Scroll down to the API Key section and click Reveal. Copy the key.
10. Go back to my IDE workspace, enter the following command in the terminal: `heroku login -i`, and enter my email then API key that I copied when prompted.
10. Then use the command `heroku stack:set heroku-20 -a <pp5-housing-price>`

11. Go back to Heroku and now select the main branch to deploy, then click Deploy Branch.


### Workspace Environment

- Since of the insufficient memory issue with Gitpod, main coding workspace including Jupyter notebook and Streamlit was conducted through Codeanywhere. But writing README file was done on Gitpod. 
- To continuously work on the up-to-dated repository on a different workspace, I used `git fetch` and `git merge`. 
- First command was `git fetch origin main` 
- Once fetching was done, I went through merging the repository by the command `git merge origin/main` 
- This made me to easily work on cross environments.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* In case you would like to thank the people that provided support through this project.


----

## Codeanywhere Template Instructions

Welcome,

This is the Code Institute student template for Codeanywhere. We have preinstalled all of the tools you need to get started. It's perfectly ok to use this template as the basis for your project submissions. Click the `Use this template` button above to get started.

You can safely delete the Codeanywhere Template Instructions section of this README.md file,  and modify the remaining paragraphs for your own project. Please do read the Codeanywhere Template Instructions at least once, though! It contains some important information about the IDE and the extensions we use. 

## How to use this repo

1. Use this template to create your GitHub project repo

1. Log into <a href="https://app.codeanywhere.com/" target="_blank" rel="noreferrer">CodeAnywhere</a> with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and <code>pip3 install -r requirements.txt</code>

1. In the terminal type <code>pip3 install jupyter</code>

1. In the terminal type <code>jupyter notebook --NotebookApp.token='' --NotebookApp.password=''</code> to start the jupyter server.

1. Open port 8888 preview or browser

1. Open the jupyter_notebooks directory in the jupyter webpage that has opened and click on the notebook you want to open.

1. Click the button Not Trusted and choose Trust.

Note that the kernel says Python 3. It inherits from the workspace so it will be Python-3.8.12 as installed by our template. To confirm this you can use <code>! python --version</code> in a notebook code cell.


## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In your Cloud IDE, from the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.



## Business Requirements
As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.


## Hypothesis and how to validate?
* List here your project hypothesis(es) and how you envision validating it (them).


## The rationale to map the business requirements to the Data Visualisations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.


## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.



## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* In case you would like to thank the people that provided support through this project.

