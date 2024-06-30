# ML Online Retail Analysis & Customer Segmentation

This repository contains the Machine Learning Customer Segmentation Project, which is part of my MSc Business Mathematics. 
It focuses on leveraging machine learning techniques to analyze and segment customers based on their purchasing behaviors from the Online Retail Data Set available at the UCI Machine Learning Repository.

***

## Overview
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
| Description   | Feature    | Categorical  | Product name.                                                       |
| Quantity      | Feature    | Integer      | The quantities of each product (item) per transaction.              |
| InvoiceDate   | Feature    | Date         | The day and time when each transaction was generated.               |
| UnitPrice     | Feature    | Continuous   | Product price per unit.                                             |
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
Organized a vast array of product descriptions, amounting to 3,878 entries, into well-defined categories. This categorization is driven by the goal to uncover inherent groupings that reveal subtle patterns and similarities not immediately apparent in the raw data.

#### `Methodology`
Our approach involves a process of text preprocessing and key term extraction to prepare the data for robust clustering:

#### `Text Preprocessing & Key Term Extraction`
- **Identifying Key Parts of Speech**: We specifically target nouns in product descriptions as they often indicate key features or elements.
- **Condensing Words to Their Roots**: Applying stemming techniques, we reduce words to their root forms.
- **Extracting and Counting Terms**: From the stemmed nouns, we compile a frequency map to measure the significance of each term within the set of texts.
- **Selecting Representative Terms**: For groups of words sharing the same root, we choose the shortest term as the representative for simplicity and clarity.
  
#### `Keyword Filtering Strategy`
- **Exclusion Criteria**: We filter out terms based on their commonality, rarity, and informativeness. Only terms that contribute meaningfully to product differentiation are retained.
- **Threshold Settings**: We implement a word length threshold to ensure focus on substantial terms, excluding words with non-contributive characters like '+' or '/'.
  
#### `Constructing a Data Matrix for Clustering`
- **Binary Variable Transformation**: Each qualifying keyword is converted into a binary variable for each product description, indicating the presence or absence of that keyword.
- **Integration of Price Range Data**: By including price segmentation, we add an economic dimension to the clustering process, aligning it with both qualitative and quantitative attributes.
  
#### `Clustering Output`
The data matrix, comprising rows of individual products and columns of features (keywords and price ranges), serves as the input for clustering algorithms. The outcome is a set of clusters, each labeled with an indicative name reflecting the thematic essence captured by the cluster:

<div align="center">
<table>
  <tr>
    <th>Cluster Number</th>
    <th>Cluster Name</th>
  </tr>
  <tr>
    <td>0</td>
    <td>Vintage Design</td>
  </tr>
  <tr>
    <td>1</td>
    <td>Classic Artistry</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Urban Home & Jewellery</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Accessories</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Holiday Essentials</td>
  </tr>
</table>
</div>


**Note:** The names assigned to each cluster are not absolute but serve as labels to facilitate easier analysis and discussion.

***

## 2. Part II
[View Part II Notebook](https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/Part_2/part_2.ipynb)

<a id="2-1-rfm-analysis"></a>
### 2.1 RFM Analysis
Applied Recency, Frequency, Monetary (RFM) metrics to segment customers based on their purchasing patterns, identifying key customer groups.
- **R (Recency)**: Recency measures how recently a customer made a purchase. This metric helps to identify customers who have engaged with the brand recently, under the assumption that the more recent the purchase, the more likely the customer will remain engaged.
- **F (Frequency)**: Frequency measures how often a customer makes a purchase within a defined time period. A higher frequency indicates a higher engagement level and loyalty.
- **M (Monetary Value)**: Monetary value measures how much money a customer has spent with the brand over a period of time. It helps in identifying the highest spending customers who are contributing more to the revenue.

<div align="center">
<table>
  <thead>
    <tr>
      <th>Customer Category</th>
      <th>Average Recency</th>
      <th>Average Frequency</th>
      <th>Average Monetary</th>
      <th>Customer Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>At-Risk</td>
      <td>249.5</td>
      <td>1.7</td>
      <td>408.8</td>
      <td>1019</td>
    </tr>
    <tr>
      <td>Loyal Customers</td>
      <td>30.4</td>
      <td>9.1</td>
      <td>3224.9</td>
      <td>714</td>
    </tr>
    <tr>
      <td>Potential Loyalists</td>
      <td>48.1</td>
      <td>2.9</td>
      <td>720.2</td>
      <td>2404</td>
    </tr>
    <tr>
      <td>VIPs</td>
      <td>10.6</td>
      <td>25.5</td>
      <td>5356.5</td>
      <td>202</td>
    </tr>
  </tbody>
</table>
</div>


<a id="2-2-customer-segmentation-using-rfm-quantile-scores-and-category-spending-patterns"></a>
### 2.2 Customer Segmentation using RFM Quantile Scores and Category Spending Patterns
In this stage of our analysis, we further refine our customer segmentation by applying quantile scoring to the RFM metrics and analyzing spending patterns across various product categories. This dual approach allows us to gain a more nuanced understanding of customer behaviors and preferences.

#### `Features`
- **Revenue Percentage per Category per Customer:** We calculate the revenue percentage for each product category per customer by dividing the revenue generated by a customer in a specific category by their total spending across all categories. This metric is expressed as a percentage, highlighting the proportion of total spending dedicated to each category.
- **"Dominant" Category per Customer:** The 'Dominant' Category for each customer is determined by identifying which product category has the highest total spending or transaction count for that customer.
- **Recency, Frequency & Monetary Scores (0-99) using Quantiles:** Each customer receives a score from 0 to 99 for each RFM metric based on their quantile rank. These scores standardize the RFM metrics to other percentage-based measures, improving the integration and comparability of analyses.

***

## 3. Part III
[View Part III Notebook](https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/Part_3/part_3.ipynb)

<a id="3-1-association-rules-mining"></a>
### 3.1 Association Rules Mining
Advanced data mining techniques were used to discover frequent itemsets and robust association rules among products, providing insights into customer purchasing patterns and product associations.

<a id="3-2-simple-recommendation-system"></a>
### 3.2 Simple Recommendation System
A straightforward recommendation system was developed, utilizing the insights from association rules mining to suggest products to customers based on their historical purchasing patterns, aiming to enhance customer satisfaction and increase sales potential.
