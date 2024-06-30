# ML Online Retail Analysis & Customer Segmentation

This repository contains the Machine Learning Customer Segmentation Project, which is part of my MSc Business Mathematics. 
It focuses on leveraging machine learning techniques to analyze and segment customers based on their purchasing behaviors from the Online Retail Data Set available at the UCI Machine Learning Repository.

***

## Table of Contents
   [The Data Set](#the-data-set)
1. [Part I](#1-part-i)
   - [Data Preparation & Feature Engineering](#1.1-data-preparation--feature-engineering)
   - [Exploratory Analysis](#1.2-exploratory-analysis)
   - [NLP-Driven Product Categorization](#1.3-nlp-driven-product-categorization)
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

### 1.1 Data Preparation & Feature Engineering

#### `Deduplication, Missing CustomerID Removal and Data Type Conversion`
- Remove duplicate entries to ensure the uniqueness of each transaction in the dataset.
- Entries without a 'CustomerID' are removed since they are essential for customer-specific analyses.
- Convert data types for better compatibility with analysis tools. For instance, ensuring that dates are in datetime format and categorical data are treated as such.

#### `Handling Cancellations and Returns`
The process for handling cancellations and returns is crucial to ensure data integrity and accuracy in analysis:

- **Cancellations**: Identified primarily by an 'InvoiceNo' that starts with 'C'. These entries typically represent transactions that were cancelled after being initiated. However, not all cancellations directly imply errors or unwanted transactions. For example, transactions described as 'Discount' with negative quantities are automatically treated as cancellations unless further verification suggests otherwise. This categorization helps in distinguishing between genuine transaction cancellations and adjustments made for other reasons such as promotions.

- **Potential Returns**: Transactions with negative quantities are scrutinized to determine if they are genuine returns:
  - A return is considered 'potential' if there exists a corresponding transaction prior to it with a positive quantity for the same 'StockCode' and 'CustomerID'. This matching confirms that the negative transaction is likely a genuine return, as it reverses a part of a previous purchase.
  - If no prior matching transaction is found, the negative quantity is treated as a cancellation. This distinction is important because it identifies entries that may not necessarily represent actual product returns but rather corrections or cancellations without a prior sale.

- **Handling Unmatched Returns**: In cases where a negative transaction (potential return) does not match any previous positive transaction, a challenge arises in confirming whether these are genuine returns or data irregularities. For data cleanliness and to avoid analysis skew:
  - These entries are often removed from the dataset unless additional information (e.g., customer communications or detailed transaction logs) justifies their inclusion.
  - It's crucial to note that some of these unmatched returns might actually be legitimate returns where the original purchase was not recorded due to issues like data loss or transaction errors.

This careful examination and handling of cancellations and returns ensure that the dataset used for analysis does not include misleading data, thus maintaining the reliability of the insights derived from subsequent analyses.

#### `Data Enhancement`
- **Date Features**: Extract additional features from 'InvoiceDate' such as day of the week, month, and hour to uncover patterns related to time.
- **Adjusted Prices**: Compute adjusted prices for transactions, particularly handling discounts and bulk purchases.
- **Invoice Total**: Calculate the total amount for each invoice to facilitate revenue analysis.

### 1.2 Exploratory Analysis
This phase involved a deep dive into the dataset to understand the distribution of variables, detect outliers, and uncover patterns to inform subsequent analyses and feature engineering efforts.

### 1.3 NLP-Driven Product Categorization
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
