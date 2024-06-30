# ML Online Retail Analysis & Customer Segmentation

This repository contains the Machine Learning Customer Segmentation Project, which is part of my MSc Business Mathematics. 
It focuses on leveraging machine learning techniques to analyze and segment customers based on their purchasing behaviors from the Online Retail Data Set available at the UCI Machine Learning Repository.

***

## Table of Contents
   [The Data Set](#the-data-set)
1. [Part I](#1-part-i)
   - [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
   - [Exploratory Analysis](#exploratory-analysis)
   - [NLP-Driven Product Categorization](#nlp-driven-product-categorization)
2. [Part II](#2-part-ii)
   - [RFM Analysis](#rfm-analysis)
   - [Customer Segmentation using RFM Quantile Scores and Category Spending Patterns](#customer-segmentation-using-rfm-quantile-scores-and-category-spending-patterns)
3. [Part III](#3-part-iii)
   - [Association Rules Mining](#association-rules-mining)
   - [Simple Recommendation System](#simple-recommendation-system)
  
***

### The Data Set

The project utilizes the Online Retail Data Set from the UCI Machine Learning Repository. This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

**Access the Data Set:** [Online Retail UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail)

**Variables Table**

| Variable Name | Role       | Type         | Description                                                        |
|---------------|------------|--------------|--------------------------------------------------------------------|
| InvoiceNo     | ID         | Categorical  | A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation or a reversal. |
| StockCode     | ID         | Categorical  | A 5-digit integral number uniquely assigned to each distinct product. |
| Description   | Feature    | Categorical  | Product name.                                                      |
| Quantity      | Feature    | Integer      | The quantities of each product (item) per transaction.              |
| InvoiceDate   | Feature    | Date         | The day and time when each transaction was generated.               |
| UnitPrice     | Feature    | Continuous   | Product price per unit.                                            |
| CustomerID    | Feature    | Categorical  | A 5-digit integral number uniquely assigned to each customer.       |
| Country       | Feature    | Categorical  | The name of the country where each customer resides.                |

***

## 1. Part I

### Data Preparation & Feature Engineering

#### Deduplication, Missing CustomerID Removal and Data Type Conversion
- Remove duplicate entries to ensure the uniqueness of each transaction in the dataset.
- Entries without a 'CustomerID' are removed since they are essential for customer-specific analyses.
- Convert data types for better compatibility with analysis tools. For instance, ensuring that dates are in datetime format and categorical data are treated as such.

#### Handling Cancellations and Returns
- **Cancellations**: Identified by 'InvoiceNo' starting with 'C', these transactions are checked against the product descriptions. Entries marked as 'Discount' but having negative quantities are treated as cancellations unless verified otherwise.
- **Returns**: Transactions with negative quantities are examined to confirm if there is a corresponding previous transaction with a positive quantity for the same 'CustomerID' and 'StockCode'. If such a transaction exists, the negative entry is treated as a return. Otherwise, if no prior purchase is found, the transaction is considered a cancellation. Genuine returns without a recorded prior purchase are removed to maintain data accuracy.

#### Data Enhancement
- **Date Features**: Extract additional features from 'InvoiceDate' such as day of the week, month, and hour to uncover patterns related to time.
- **Adjusted Prices**: Compute adjusted prices for transactions, particularly handling discounts and bulk purchases.
- **Invoice Total**: Calculate the total amount for each invoice to facilitate revenue analysis.

### Exploratory Analysis
This phase involved a deep dive into the dataset to understand the distribution of variables, detect outliers, and uncover patterns to inform subsequent analyses and feature engineering efforts.

### NLP-Driven Product Categorization
Implemented Natural Language Processing techniques to systematically categorize product descriptions, enhancing the granularity of product data for improved segmentation and analysis.

***

## 2. Part II

### RFM Analysis
Applied Recency, Frequency, Monetary (RFM) metrics to segment customers based on their purchasing patterns, identifying key customer groups for targeted marketing strategies.

### Customer Segmentation using RFM Quantile Scores and Category Spending Patterns
Further segmentation was refined by applying quantile scoring to RFM metrics, coupled with an analysis of spending patterns across different product categories to provide detailed insights into customer behaviors.

***

## 3. Part III

### Association Rules Mining
Advanced data mining techniques were used to discover frequent itemsets and robust association rules among products, providing insights into customer purchasing patterns and product associations.

### Simple Recommendation System
A straightforward recommendation system was developed, utilizing the insights from association rules mining to suggest products to customers based on their historical purchasing patterns, aiming to enhance customer satisfaction and increase sales potential.
