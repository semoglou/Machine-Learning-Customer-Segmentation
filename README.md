# ML Online Retail Analysis & Customer Segmentation

This repository contains the Machine Learning Customer Segmentation Project, which is part of my MSc Business Mathematics. 
It focuses on leveraging machine learning techniques to analyze and segment customers based on their purchasing behaviors from the Online Retail Data Set available at the UCI Machine Learning Repository.

***

### The Data Set

The project utilizes the Online Retail Data Set from the UCI Machine Learning Repository, which includes transactions (invoices) from a UK-based non-store online retail. 
The dataset spans from December 2010 to December 2011 and includes a variety of products mainly sold to wholesalers.

**Access the Data Set:** [Online Retail UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail)

***

## Table of Contents
- **Part I**
  - [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
  - [Exploratory Analysis](#exploratory-analysis)
  - [NLP-Driven Product Categorization](#nlp-driven-product-categorization)
- **Part II**
  - [RFM Analysis](#rfm-analysis)
  - [Customer Segmentation using RFM Quantile Scores and Category Spending Patterns](#customer-segmentation-using-rfm-quantile-scores-and-category-spending-patterns)
- **Part III**
  - [Association Rules Mining](#association-rules-mining)
  - [Simple Recommendation System](#simple-recommendation-system)

***

## Data Preparation & Feature Engineering

Initial data handling involved deduplication, handling missing values (specifically CustomerID), and data type conversions. Significant features like date extraction and adjusted pricing were developed to enhance model accuracy and insight depth.

## NLP-Driven Product Categorization via Clustering

We aim to organize a large set of product descriptions — totaling 3878 entries — into distinct categories. This is done by applying a combination of text preprocessing techniques:
- **Text Preprocessing & Key Term Extraction**: Techniques include keyword extraction, noun identification, and price range analysis to prepare the data for effective clustering.
- **Clustering**: Using algorithms like K-Means to categorize each description into distinct groups based on textual and contextual similarities, facilitating more efficient data management and enhancing search functionality, ultimately improving our customer segmentation efforts.

## Customer Segmentation

Utilized RFM scoring to quantify customer value and applied clustering techniques to segment customers into meaningful groups. This segmentation helps tailor marketing strategies to different customer segments based on their behavior and value to the business.

## Association Rules Mining

Implemented the FP-Growth algorithm to efficiently find frequent itemsets and generate association rules between products. This analysis helps in identifying products that are frequently bought together, enabling effective cross-selling strategies.

## Product Recommendation System

Developed a simple and effective product recommendation system using the insights gained from association rules mining. This system suggests products to customers based on items they have previously bought, enhancing the shopping experience and increasing sales potential.
