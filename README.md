# ML Online Retail Analysis & Customer Segmentation

This repository contains the Machine Learning Customer Segmentation Project, which is part of my MSc Business Mathematics. 
It focuses on leveraging machine learning techniques to analyze and segment customers based on their purchasing behaviors from the Online Retail Data Set available at the UCI Machine Learning Repository.

***

## Table of Contents
   [The Data Set](#the-data-set)
1. [Part I](#part-i)
   - [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
   - [Exploratory Analysis](#exploratory-analysis)
   - [NLP-Driven Product Categorization](#nlp-driven-product-categorization)
2. [Part II](#part-ii)
   - [RFM Analysis](#rfm-analysis)
   - [Customer Segmentation using RFM Quantile Scores and Category Spending Patterns](#customer-segmentation-using-rfm-quantile-scores-and-category-spending-patterns)
3. [Part III](#part-iii)
   - [Association Rules Mining](#association-rules-mining)
   - [Simple Recommendation System](#simple-recommendation-system)
  
***

### The Data Set

The project utilizes the Online Retail Data Set from the UCI Machine Learning Repository, which includes transactions (invoices) from a UK-based non-store online retail. 
The dataset spans from December 2010 to December 2011 and includes a variety of products mainly sold to wholesalers.

**Access the Data Set:** [Online Retail UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail)

***

## 1. Part I

### Data Preparation & Feature Engineering
Initial data handling involved deduplication, handling missing values, and employing complex methods to identify and exclude cancellations and returns. Feature engineering focuses on deriving new variables that could enhance the predictive models.

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
