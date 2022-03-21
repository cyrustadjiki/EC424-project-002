---
title: "Project 2 EC 424"
author: "Cyrus Tadjiki"
date: "2/15/2022"
output: 
  html_document:
    theme: flatly
    highlight: monochrome 
    toc: yes
    toc_depth: 4
    toc_float: yes
    keep_md: true
# themes include default, cerulean, journal, flatly, darkly, readable, spacelab, united, cosmo, lumen, paper, sandstone, simplex, and yeti. Pass null for no theme
---



## Part 0: Set Up
#### Loading Packages

```r
library(pacman)
p_load(tidyverse, rvest, lubridate, janitor, 
       data.table, readr, readxl, dplyr, skimr, 
       broom, tidyr, stringr, maps, gridExtra,
       tidymodels, knitr, here, glmnet,
       fastverse, tidymodels, magrittr)
```

#### Loading Data

```r
df = read.csv("election-2016.csv")
df$winner_2016 <- ifelse(df$i_republican_2016 == 1, 'R', 'D')
```



## Part 1: Penalized regression  

#### 01. 5-fold cross validation
Using 5-fold cross validation: tune a Lasso regression model. (Don't forget: You can add interactions, transformations, etc. in your recipe. You'll also want to standardize your variables.)  

```r
lambdas = data.frame(penalty = c(0, 10^seq(5,-2, length = 100)))

lasso_model = linear_reg(
  # CHANGED from model to mode
	mode = 'regression',
	penalty = tune(),
	mixture = 1
) %>% set_engine('glmnet')
# Set up recipe
lasso_rec = recipe(i_republican_2016 ~., data = df) %>%
	step_rm(winner_2016) %>%
	update_role(fips, new_role = 'id variable') %>% 
	update_role(county, new_role = 'id variable') %>% 
	update_role(state, new_role = 'id variable') %>% 
  # ADDED
  # update_role(all_nominal(), new_role = "id variable") %>%
  # # update_role(cupper_points, new_role = "outcome") %>%
  # update_role(
  #   pop_pct_white,
  #   pop_pct_nonenglish,
  #   new_role = "predictor"
  # ) %>%
  # END
	step_normalize(all_numeric_predictors()) %>% 
	step_interact(~ all_predictors() : all_predictors())
# %>% 
	# step_lincomb(all_predictors())
# Set up CV
set.seed(12345)
folds = vfold_cv(df, v = 5)
# Workflow
lasso_wf = workflow() %>% 
	add_recipe(lasso_rec) %>% 
	add_model(lasso_model)
# CV
lasso_cv = lasso_wf %>% 
	tune_grid(
		folds,
		grid = lambdas,
		metrics = metric_set(rmse, mae))

# Find best models
lasso_cv %>% collect_metrics(.metric = 'rmse') %>% arrange(mean)
```

```
## # A tibble: 202 x 7
##    penalty .metric .estimator   mean     n std_err .config               
##      <dbl> <chr>   <chr>       <dbl> <int>   <dbl> <chr>                 
##  1  0.01   mae     standard   0.0848     5 0.00225 Preprocessor1_Model002
##  2  0.0118 mae     standard   0.0855     5 0.00223 Preprocessor1_Model003
##  3  0.0138 mae     standard   0.0864     5 0.00224 Preprocessor1_Model004
##  4  0.0163 mae     standard   0.0876     5 0.00222 Preprocessor1_Model005
##  5  0.0192 mae     standard   0.0895     5 0.00221 Preprocessor1_Model006
##  6  0.0226 mae     standard   0.0922     5 0.00232 Preprocessor1_Model007
##  7  0.0266 mae     standard   0.0958     5 0.00250 Preprocessor1_Model008
##  8  0.0313 mae     standard   0.101      5 0.00269 Preprocessor1_Model009
##  9  0.0368 mae     standard   0.106      5 0.00287 Preprocessor1_Model010
## 10  0.0433 mae     standard   0.112      5 0.00298 Preprocessor1_Model011
## # ... with 192 more rows
```



#### 02. Best Penalty
Q: What is the penalty for your 'best' model? 

A: My best penalty was **0.01** because when $\lambda$=0.1 our MAE was minimized. 

#### 03. How to define "best"   
Q: Which metric did you use to define the 'best' model? Does it make sense in this setting? Explain your answer.  

A: I chose to use the mean absolute error for choosing my best model minimizing the root mean squared error is telling us to use no penalty.  

#### 04. Elasticnet Prediction Model
Now tune an elasticnet prediction model.

```r
# Same as first part but mixture = tune()

lambdas = 10^seq(from = 5, to = -2, length = 100)
alphas = seq(from = 0, to = 1, by = 0.1)

# Set up CV
set.seed(12345)
folds = vfold_cv(df, v = 5)

# Define the elasticnet model
model_net = linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

# Workflow
workflow_net = workflow() %>%
  add_recipe(lasso_rec) %>%
  add_model(model_net)

# CV
cv_lasso = workflow_net %>% 
	tune_grid(
		folds,
		grid = expand_grid(mixture = alphas, penalty = lambdas),
		metrics = metric_set(rmse, mae))

# Find best models
metircs_cv_lasso = cv_lasso %>% collect_metrics() %>% arrange(mean)
metircs_cv_lasso
```

```
## # A tibble: 2,200 x 8
##    penalty mixture .metric .estimator   mean     n std_err .config              
##      <dbl>   <dbl> <chr>   <chr>       <dbl> <int>   <dbl> <chr>                
##  1  0.01       0.7 mae     standard   0.0843     5 0.00244 Preprocessor1_Model0~
##  2  0.01       0.6 mae     standard   0.0844     5 0.00251 Preprocessor1_Model0~
##  3  0.01       0.8 mae     standard   0.0844     5 0.00236 Preprocessor1_Model0~
##  4  0.0118     0.6 mae     standard   0.0845     5 0.00244 Preprocessor1_Model0~
##  5  0.0118     0.5 mae     standard   0.0845     5 0.00252 Preprocessor1_Model0~
##  6  0.01       0.9 mae     standard   0.0846     5 0.00230 Preprocessor1_Model0~
##  7  0.0118     0.7 mae     standard   0.0846     5 0.00235 Preprocessor1_Model0~
##  8  0.01       0.2 mae     standard   0.0848     5 0.00248 Preprocessor1_Model0~
##  9  0.0138     0.5 mae     standard   0.0848     5 0.00245 Preprocessor1_Model0~
## 10  0.0138     0.4 mae     standard   0.0848     5 0.00253 Preprocessor1_Model0~
## # ... with 2,190 more rows
```


#### 05. Ridge vs. Lasso  
Q: What do the chosen hyperparameters for the elasticnet tell you about the Ridge vs. Lasso in this setting?

A: This model is telling us that to minimize out MAE and RMSE our alpha penalty should be somewhere between 0.3 and 0.6.

## Part 2: Logistic regression
#### 06. Logistic regression 5-fold CV
Now fit a logistic regression (logistic_reg() in tidymodels) model—using 5-fold cross validation to get a sense of your model's performance (record the following metrics: accuracy, precision, specificity, sensitivity, ROC AUC).

```r
logistic_model = logistic_reg(
	mode = 'classification',
	penalty = tune()
) %>% set_engine('glm') #%>% tune_args( full = FALSE)

# Set up recipe
logistic_rec = recipe(winner_2016 ~ ., data = df) %>%
	step_rm(i_republican_2016) %>%
	update_role(fips, new_role = 'id variable') %>% 
	update_role(county, new_role = 'id variable') %>% 
	update_role(state, new_role = 'id variable') %>% 
	step_normalize(all_numeric_predictors()) %>% 
  # step_dummy(all_predictors()) %>% 
  # step_dummy(all_nominal()) %>% 
	step_interact(~ all_predictors() : all_predictors()) %>%
	step_lincomb(all_predictors())

# Set up CV
set.seed(12345)
folds = vfold_cv(df, v = 5)

# Workflow
logistic_wf = workflow() %>% 
	add_recipe(logistic_rec) %>% 
	add_model(logistic_model)

# CV
logistic_cv = logistic_wf %>%
	fit_resamples(
		folds,
		# metrics = metric_set(accuracy, roc_auc)
		# control = control_resamples(save_pred = TRUE, save_workflow = TRUE),
	  
		metrics = metric_set(accuracy, roc_auc, sens, spec, yardstick::precision)
		# metrics = metric_set(accuracy, roc_auc, sens, spec)
	) 
```

```
## ! Fold1: preprocessor 1/1, model 1/1: glm.fit: fitted probabilities numerically 0...
```

```
## ! Fold2: preprocessor 1/1, model 1/1: glm.fit: algorithm did not converge, glm.fi...
```

```
## ! Fold3: preprocessor 1/1, model 1/1: glm.fit: algorithm did not converge, glm.fi...
```

```
## ! Fold4: preprocessor 1/1, model 1/1: glm.fit: algorithm did not converge, glm.fi...
```

```
## ! Fold5: preprocessor 1/1, model 1/1: glm.fit: algorithm did not converge, glm.fi...
```

```r
# Find estimated model performance
logistic_cv %>% collect_metrics()
```

```
## # A tibble: 5 x 6
##   .metric   .estimator  mean     n std_err .config             
##   <chr>     <chr>      <dbl> <int>   <dbl> <chr>               
## 1 accuracy  binary     0.926     5 0.00715 Preprocessor1_Model1
## 2 precision binary     0.764     5 0.0383  Preprocessor1_Model1
## 3 roc_auc   binary     0.893     5 0.0113  Preprocessor1_Model1
## 4 sens      binary     0.784     5 0.0144  Preprocessor1_Model1
## 5 spec      binary     0.953     5 0.00919 Preprocessor1_Model1
```
Hint: You can tell tune_grid() or fit_resamples() which metrics to collect via the metrics argument. You'll want to give the argument a metric_set().

#### 07. CV accuracy of logistic model  
Q: What is the cross-validated accuracy of this logistic model?  

A: my CV accuracy for this model was 0.9258581

#### 08. Juding accuracy
Q: Is your accuracy "good"? Explain your answer—including a comparison to the null classifier.  

A: I trust my accuracy is ggo dmy standard error is very small and we are doing a five fold cv process here in our workflow. 

#### 09. Other Metrics   
Q: What do the other metrics tell you about your model? Is it "good"? Are you consistently missing one class of outcomes? Explain.

A: My sensitivity is precision are much lower than my accuracy most likely due to us not using lasso and be able to pick and choose which indicators are best for our model and using all of the instead. In class we saw this data is very heavily skewed to approve most people for loans. 

## Part 3: Logistic Lasso
#### 10. Logistic Lasso regression 5-fold CV 
Now fit a logistic Lasso regression (logistic_reg() in tidymodels, but now tuning the penalty) model—using 5-fold cross validation. Again: record the following metrics: accuracy, precision, specificity, sensitivity, ROC AUC.

```r
ll_model = logistic_reg(
	mode = 'classification',
	# mixture = 1,
	penalty = tune()
) %>% set_engine('glmnet')

# Set up recipe
ll_rec = recipe(winner_2016 ~ ., data = df) %>%
	step_rm(i_republican_2016) %>%
	update_role(fips, new_role = 'id variable') %>%
	update_role(county, new_role = 'id variable') %>%
	update_role(state, new_role = 'id variable') %>%
	step_normalize(all_numeric_predictors()) %>%
	step_interact(~ all_predictors() : all_predictors()) %>%
	step_lincomb(all_predictors())

# Set up CV
set.seed(12345)
folds = vfold_cv(df, v = 5)

# Workflow
ll_wf = workflow() %>%
	add_recipe(ll_rec) %>%
	add_model(ll_model)

# CV
ll_cv = ll_wf %>% tune_grid(
	resamples = folds,
	metrics = metric_set(accuracy, roc_auc, sens, spec, yardstick::precision),
	grid = grid_latin_hypercube(penalty(), size = 5),
	# grid = grid_latin_hypercube(penalty() , mixture(), size = 20),
	control = control_grid(parallel_over = 'resamples')
)

# Find best models
ll_cv %>% collect_metrics() %>% arrange(mean)
```

```
## # A tibble: 25 x 7
##     penalty .metric   .estimator  mean     n std_err .config             
##       <dbl> <chr>     <chr>      <dbl> <int>   <dbl> <chr>               
##  1 4.94e- 2 sens      binary     0.871     5  0.0268 Preprocessor1_Model5
##  2 4.94e- 2 precision binary     0.880     5  0.0246 Preprocessor1_Model5
##  3 3.91e-10 sens      binary     0.886     5  0.0179 Preprocessor1_Model1
##  4 5.89e- 8 sens      binary     0.886     5  0.0179 Preprocessor1_Model2
##  5 8.09e- 6 sens      binary     0.886     5  0.0179 Preprocessor1_Model3
##  6 3.91e-10 precision binary     0.888     5  0.0156 Preprocessor1_Model1
##  7 5.89e- 8 precision binary     0.888     5  0.0156 Preprocessor1_Model2
##  8 8.09e- 6 precision binary     0.888     5  0.0156 Preprocessor1_Model3
##  9 4.09e- 4 sens      binary     0.900     5  0.0139 Preprocessor1_Model4
## 10 4.09e- 4 precision binary     0.919     5  0.0153 Preprocessor1_Model4
## # ... with 15 more rows
```


#### 11. Performance of Model
Q: How does the performance of this logistic Lasso compare to the logistic regression in Part 2?

A: This part is slightly harder to analyze because we have 27 different penalties. Our metrics seem fairly more consistent and higher rates. 

#### 12. Room for imporvment?   
Q: Do you think moving to a logistic elasticnet would improve anything? Explain.

A: I think moving to an elasticnet woul dmake this table more complicated to read.     
## Part 4: Reflection  

#### 13. Preference
Why might we prefer Lasso to elasticnet (or vice versa)?

Lasso lets us have the "best" indicating variables for predicting models and is less flexible and elasticnet if more flexible with training data and we have to use every variable in our model. 

#### 14. Differences  
Q: What the the differences between logistic regression and linear regression? What are the similarities?  

A: Logistic regression have different interpretations of our coefficients than linear regressions. The obvious difference is one is linear and one is logarithmic but looking closer at it logarithmic regression produce a probability value for our model with the glm() function and linear regression are much less flexible and produce any number as an output interpenetration with our lm() function. 

#### 15. Final Question
Q: Imagine you worked for a specific political party. Which metric would you use for assessing model performance?

A: I think this depends on which political party I am working for. I'm not too sure what this question is asking. Maybe the dilemma is whether to advertise your accuracy versus your precision? Those to concepts get confused especially when they are thrown around without evidence. 







