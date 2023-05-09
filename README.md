# Proyecto_google

## Target:
ACME Insurance Inc. provides affordable health insurance to thousands of customers across the United States. They want to launch an attractive customer acquisition campaign. Your task is to study the data provided by the company, such as age, gender, BMI, children, smoking habits and region of residence of current customers and propose possible strategies.

## Introduction:
Analysis of available data looking for possible trend lines to estimate the annual medical expenditure of new and existing customers, establishing the reduced annual premium price as the main attraction, while the company promotes healthy habits among its customers by reducing the risk of unpredictable extra expenses.

## Method:
-Data analysis and visualisations: Python and R
-Model: Estimating annual medical expenditure using available information (Python, lazypredict, skalear)

## Conclusions:
Our data includes mostly overweight, middle-aged clients with healthy smoking habits and no children.
The trend lines of annual premiums are mostly conditioned by smoking and age (remembering that an unhealthy BMI is always the basis in most cases, and is an important driver of premiums).
Of these, smoking (difficult to monitor) and BMI (easily monitored) are variables.
Our model, is able to predict with accuracy: 0.826 and mean squared error (MSE) on test set: 31916978 the annual insurance premium with these variables.

## Proposal:
Creation of an interactive campaign in which potential new and existing customers can calculate the reduction in the amount of their annual insurance  premium by enrolling in a weight reduction programme and maintaining their reduced BMI almost for a year (supervised) to achieve this reduction in the annual insurance  premium, taking into account future ages for the calculation.
