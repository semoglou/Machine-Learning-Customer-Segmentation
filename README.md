# ML Online Retail Analysis & Customer Segmentation

This repository contains the Machine Learning Customer Segmentation Project, which is part of my MSc Business Mathematics. 
It focuses on leveraging machine learning techniques to analyze and segment customers based on their purchasing behaviors from the Online Retail Data Set available at the UCI Machine Learning Repository.

***

## Table of Contents
   [The Data Set](#the-data-set)
1. [Part I](#1-part-i)
   - 1.1 [Data Preparation & Feature Engineering](#1-1-data-preparation--feature-engineering)
   - 1.2 [Exploratory Analysis](#1-2-exploratory-analysis)
   - 1.3 [NLP-Driven Product Categorization](#1-3-nlp-driven-product-categorization)
2. [Part II](#2-part-ii)
   - 2.1 [RFM Analysis](#2-1-rfm-analysis)
   - 2.2 [Customer Segmentation using RFM Quantile Scores and Category Spending Patterns](#2-2-customer-segmentation-using-rfm-quantile-scores-and-category-spending-patterns)
3. [Part III](#3-part-iii)
   - 3.1 [Association Rules Mining](#3-1-association-rules-mining)
   - 3.2 [Simple Recommendation System](#3-2-simple-recommendation-system)
  
***

### The Data Set

The project utilizes the Online Retail Data Set from the UCI Machine Learning Repository. This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

- **Access the Data Set:** [Online Retail UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail)

- **Variables Table**

| Variable Name | Role       | Type         | Description                                                        |
|---------------|------------|--------------|--------------------------------------------------------------------|
| InvoiceNo     | ID         | Categorical  | A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation or a reversal. |
| StockCode     | ID         | Categorical  | A 5-digit integral number uniquely assigned to each distinct product. |
| Description   | Feature    | Categorical  | Product name.                                                      |
| Quantity      | Feature    | Integer      | The quantities of each product (item) per transaction.              |
| InvoiceDate   | Feature    | Date         | The day and time when each transaction was generated.               |
| UnitPrice     | Feature    | Continuous   | Product price per unit.                                            |
| CustomerID    | ID         | Categorical  | A 5-digit integral number uniquely assigned to each customer.       |
| Country       | Feature    | Categorical  | The name of the country where each customer resides.                |

***

## 1. Part I
[View Part I Notebook](https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/Part_1/part_1.ipynb)

<a id="1-1-data-preparation--feature-engineering"></a>
### 1.1 Data Preparation & Feature Engineering

#### `Deduplication, Missing CustomerID Removal and Data Type Conversion`
- Remove duplicate entries to ensure the uniqueness of each transaction in the dataset.
- Entries without a 'CustomerID' are removed since IDs are essential for customer-specific analyses.
- Convert data types for better compatibility with analysis tools. For instance, ensuring that dates are in datetime format and categorical data are treated as such.

#### `Handling Cancellations and Returns`
The process for handling cancellations and returns is crucial to ensure data integrity and accuracy in analysis:

Negative 'quantity' transactions are considered potential returns if there exists a corresponding prior positive transaction with the same 'product_id' from the same 'customer_id'. This ensures that the negative transaction is a genuine return and not an error or irregularity.
If no corresponding positive transaction exists before the negative one, it is treated as a cancellation. It is important to note that some cancellations might be genuine returns without a recorded prior purchase. For such cancellations, we remove these entries from the dataset to correct the data and avoid skewing our analysis.

#### `Data Enhancement`
- **Date Features**: Extract additional features from 'InvoiceDate' such as day of the week, month, and hour to uncover patterns related to time.
- **Invoice Total**: Calculate the total amount for each invoice to facilitate revenue analysis.

<a id="1-2-exploratory-analysis"></a>
### 1.2 Exploratory Analysis
This phase involved a deep dive into the dataset to understand the distribution of variables.
#### `Key Areas of Focus`
- **Sales Distribution by Time**: Analyzed sales data to uncover trends across different timescales—hourly, daily, and monthly. This helps in understanding peak shopping hours, busiest shopping days, and seasonal trends which are essential for inventory and marketing strategies.
  
- **Customer Demographics**: Explored demographic data, enhancing the understanding of our customer base.

- **Revenue by Country**: Mapped out revenue generation across different countries to pinpoint high-value markets and assess the global reach of the business.

- **Product Popularity**: Investigated the most frequently purchased items by analyzing the 'Description' field.

<a id="1-3-nlp-driven-product-categorization"></a>
### 1.3 NLP-Driven Product Categorization
Implemented Natural Language Processing techniques to systematically categorize product descriptions, enhancing the granularity of product data for improved segmentation and analysis.

***

## 2. Part II
[View Part II Notebook](https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/Part_2/part_2.ipynb)

<a id="2-1-rfm-analysis"></a>
### 2.1 RFM Analysis
Applied Recency, Frequency, Monetary (RFM) metrics to segment customers based on their purchasing patterns, identifying key customer groups for targeted marketing strategies.

<a id="2-2-customer-segmentation-using-rfm-quantile-scores-and-category-spending-patterns"></a>
### 2.2 Customer Segmentation using RFM Quantile Scores and Category Spending Patterns
Further segmentation was refined by applying quantile scoring to RFM metrics, coupled with an analysis of spending patterns across different product categories to provide detailed insights into customer behaviors.

***

## 3. Part III
[View Part III Notebook](https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/Part_3/part_3.ipynb)

<a id="3-1-association-rules-mining"></a>
### 3.1 Association Rules Mining
Advanced data mining techniques were used to discover frequent itemsets and robust association rules among products, providing insights into customer purchasing patterns and product associations.

<a id="3-2-simple-recommendation-system"></a>
### 3.2 Simple Recommendation System
A straightforward recommendation system was developed, utilizing the insights from association rules mining to suggest products to customers based on their historical purchasing patterns, aiming to enhance customer satisfaction and increase sales potential.
