# ML Online Retail Analysis & Customer Segmentation

This repository contains the Machine Learning Project, which is part of my MSc in Business Mathematics. The project focuses on leveraging machine learning techniques for customer segmentation. By analyzing and segmenting customers based on their purchasing behaviors, this project aims to derive valuable insights from the Online Retail Data Set available at the UCI Machine Learning Repository. 

**Supervising Professor:** [D. Panagopoulos](https://www.linkedin.com/in/dpanagopoulos/)

***

## Overview
-    [The Data Set](#the-data-set)
- [Part I](#1-part-i)
   - 1.1 [Data Preparation & Feature Engineering](#1-1-data-preparation--feature-engineering)
   - 1.2 [Exploratory Analysis](#1-2-exploratory-analysis)
   - 1.3 [NLP-Driven Product Categorization](#1-3-nlp-driven-product-categorization)
- [Part II](#2-part-ii)
   - 2.1 [RFM Analysis](#2-1-rfm-analysis)
   - 2.2 [Customer Segmentation using RFM Quantile Scores and Category Spending Patterns](#2-2-customer-segmentation-using-rfm-quantile-scores-and-category-spending-patterns)
- [Part III](#3-part-iii)
   - 3.1 [Association Rules Mining](#3-1-association-rules-mining)
   - 3.2 [Simple Recommendation System](#3-2-simple-recommendation-system)
   - 3.3 [Network Graph Analysis of Product Invoices and Recommendations](#3-3-network-graph-analysis-of-product-invoices-and-recommendations)
  
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
- **Date Features**\
   Extract additional features from 'InvoiceDate' such as day of the week, month, and hour to uncover patterns related to time.
- **Invoice Total**\
   Calculate the total amount for each invoice to facilitate revenue analysis.

<a id="1-2-exploratory-analysis"></a>
### 1.2 Exploratory Analysis
This phase involved a deep dive into the dataset to understand the distribution of variables.

#### `Key Areas of Focus`
- **Sales Distribution by Time** \
   Analyzed sales data to uncover trends across different timescales—hourly, daily, and monthly. This helps in understanding peak shopping hours, busiest shopping days, and seasonal trends which are essential for inventory and marketing strategies.
##### `Sales by Time of the Day`
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/salesvtime.png" alt="Sales vs Time" />
</div>

##### `Sales by Day of the Month`
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/salesvday.png" alt="Sales vs Day" />
</div>

##### `Sales by Month`
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/salesvmonth.png" alt="Sales vs Month" />
</div>
  
- **Customer Demographics** \
   Explored demographic data, enhancing the understanding of our customer base.

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/dem.png" alt="Demographics" />
</div>

- **Revenue by Country** \
   Mapped out revenue generation across different countries to pinpoint high-value markets and assess the global reach of the business.
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/revcou.png" alt="EDA Output" />
</div>

- **Product Popularity** \
   Investigated the most frequently purchased items by analyzing the 'Description' field.
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/eda.png" alt="EDA Output" />
</div>

<a id="1-3-nlp-driven-product-categorization"></a>
### 1.3 NLP-Driven Product Categorization
Organized a vast array of product descriptions, amounting to 3,878 entries, into well-defined categories. This categorization is driven by the goal to uncover inherent groupings that reveal subtle patterns and similarities not immediately apparent in the raw data.

#### `Methodology`
Our approach involves a process of text preprocessing and key term extraction to prepare the data for robust clustering:

#### `Text Preprocessing & Key Term Extraction`
- **Identifying Key Parts of Speech** \
   We specifically target nouns in product descriptions as they often indicate key features or elements.
- **Condensing Words to Their Roots** \
   Applying stemming techniques, we reduce words to their root forms.
- **Extracting and Counting Terms** \
   From the stemmed nouns, we compile a frequency map to measure the significance of each term within the set of texts.
- **Selecting Representative Terms** \
   For groups of words sharing the same root, we choose the shortest term as the representative for simplicity and clarity.

#### `Words Occurrence`
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/wordcloud.png" alt="Word Cloud" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/treemap.png" alt="Treemap" />
</div>
  
#### `Keyword Filtering Strategy`
- **Exclusion Criteria** \
   We filter out terms based on their commonality, rarity, and informativeness. Only terms that contribute meaningfully to product differentiation are retained.
- **Threshold Settings** \
   We implement a word length threshold to ensure focus on substantial terms, excluding words with non-contributive characters like '+' or '/'.
  
#### `Constructing a Data Matrix for Clustering`
- **Binary Variable Transformation** \
   Each qualifying keyword is converted into a binary variable for each product description, indicating the presence or absence of that keyword.
- **Integration of Price Range Data** \
   By including price segmentation, we add an economic dimension to the clustering process, aligning it with both qualitative and quantitative attributes.
  
#### `K-Means Clustering`
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/silhouettewords.png" alt="Silhouette Words" />
</div>
  
#### `Clustering Output`
The (scaled) data matrix, comprising rows of individual products and columns of features (keywords and price ranges), serves as the input for clustering algorithms. The outcome is a set of clusters, each labeled with an indicative name reflecting the thematic essence captured by the cluster:

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

#### `t-SNE Visualization of Clusters`
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/tsnenlpwords.png" alt="t-SNE Visualization of Clusters" />
</div>

#### `Word Clouds of Clusters`
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/cluster0.png" alt="Cluster 0 Word Cloud" />
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/cluster1.png" alt="Cluster 1 Word Cloud" />
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/cluster2.png" alt="Cluster 2 Word Cloud" />
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/cluster3.png" alt="Cluster 3 Word Cloud" />
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/cluster4.png" alt="Cluster 4 Word Cloud" />
</div>


**Note:** The names assigned to each cluster are not absolute but serve as labels to facilitate easier analysis and discussion.

#### `Product Categories`
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/categoriestreemap.png" alt="Categories Tree Map" />
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/categoriesbarh.png" alt="Categories Analysis" />
</div>

#### `Further Handling of Negative Values using Product Categories`
To further address negative total prices in the dataset, we identify transactions with negative amounts, group the data by customer and product category, and filter out instances with negative total spending in any category. Specific invoices contributing to this "negativity" are isolated, and their total prices are set to zero. This adjustment ensures accurate and reliable records by preventing distortions in analysis caused by returns, discounts, or data entry errors.

***

## 2. Part II
[View Part II Notebook](https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/Part_2/part_2.ipynb)

<a id="2-1-rfm-analysis"></a>
### 2.1 RFM Analysis
Applied Recency, Frequency, Monetary (RFM) metrics to segment customers based on their purchasing patterns, identifying key customer groups.
- **R (Recency)** \
      Recency measures how recently a customer made a purchase. This metric helps to identify customers who have engaged with the brand recently, under the assumption that the more recent the purchase, the more likely the customer will remain engaged.
- **F (Frequency)** \
      Frequency measures how often a customer makes a purchase within a defined time period. A higher frequency indicates a higher engagement level and loyalty.
- **M (Monetary Value)** \
      Monetary value measures how much money a customer has spent with the brand over a period of time. It helps in identifying the highest spending customers who are contributing more to the revenue.

#### `RFM Distributions`

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/distin.png" alt="Initial Distribution" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/beforeclippingrfm.png" alt="RFM Distributions" />
</div>

#### `Outlier Handling - Upper Clipping`

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/distfin.png" alt="Final Distribution" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/afterclipping.png" alt="Outlier Handling - Upper Clipping" />
</div>

#### `K-Means Clustering`
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/elbowrfm.png" alt="K-Means Clustering" />
</div>

#### `Clustering Results`

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


<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/treerfm.png" alt="RFM Treemap" /> <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/pierfm.png" alt="RFM Pie Chart" />
</div>

<a id="2-2-customer-segmentation-using-rfm-quantile-scores-and-category-spending-patterns"></a>
### 2.2 Customer Segmentation using RFM Quantile Scores and Category Spending Patterns
In this stage of our analysis, we further refine our customer segmentation by applying quantile scoring to the RFM metrics and analyzing spending patterns across various product categories. This dual approach allows us to gain a more nuanced understanding of customer behaviors and preferences.

#### `Features`
- **Revenue Percentage per Category per Customer** \
   We calculate the revenue percentage for each product category per customer by dividing the revenue generated by a customer in a specific category by their total spending across all categories. This metric is expressed as a percentage, highlighting the proportion of total spending dedicated to each category.
- **"Dominant" Category per Customer** \
   The 'Dominant' Category for each customer is determined by identifying which product category has the highest total spending or transaction count for that customer.
- **Recency, Frequency & Monetary Scores (0-99) using Quantiles** \
   Each customer receives a score from 0 to 99 for each RFM metric based on their quantile rank. These scores standardize the RFM metrics to other percentage-based measures, improving the integration and comparability of analyses.

#### `Clustering Evaluation Methods`

- **Feature Importance Analysis using Random Forest** \
     Trained a Random Forest on the cluster labels. This analysis helps identify which attributes are most influential in defining customer segments.
- **Performance Metrics** \
     Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index
- **t-SNE Visualization** \
     Visualization of High-Dimensional data in two dimensions, highlighting the distribution of clusters 

#### `Clustering Techniques Applied`

Following the evaluation of feature importance and performance metrics, we adopted a multi-step clustering approach to optimize our segmentation strategy. This involved the sequential application of different clustering algorithms to refine our clusters and improve the granularity of our customer segmentation. All features were scaled appropriately before applying the algorithms and the results were then assessed back on the initial dataset to ensure consistency and reliability.

- **K-Means Clustering** \
   Initially, we applied the K-Means clustering algorithm to establish a baseline for segmentation. K-Means was chosen for its efficiency and effectiveness in grouping large data sets into k distinct clusters based on attribute similarity.

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/1kmeanselbow.png" alt="K-Means Elbow" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/1featimp.png" alt="Feature Importance" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/1tsne.png" alt="t-SNE Visualization" />
</div>
   
- **Hierarchical Clustering**\
   To further refine our cluster definitions and potentially identify a more optimal number of clusters, we then employed Hierarchical Clustering. This method allowed us to visualize and assess different cluster possibilities through a dendrogram, providing insight into how data points are grouped at various levels of granularity.

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/2dend.png" alt="Dendrogram" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/2featimp.png" alt="Feature Importance" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/2tsne.png" alt="t-SNE Visualization" />
</div>

- **Revised K-Means Clustering** \
   Based on the insights gained from Hierarchical Clustering, specifically the number of clusters suggested by the dendrogram, we performed a second round of K-Means clustering. This time, we used the cluster count obtained from the hierarchical method as the input for 'k'. This refined approach allowed us to fine-tune our segmentation, leading to more distinct and actionable customer groups.

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/3featimp.png" alt="Feature Importance" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/3tsne.png" alt="t-SNE Visualization" />
</div>

This iterative clustering strategy — starting with K-Means, refining with Hierarchical Clustering, and concluding with a revised K-Means — proved to be highly effective. It enabled us to leverage the strengths of both methods: the computational efficiency of K-Means and the detailed insight provided by Hierarchical Clustering. The final iteration of K-Means, using the informed choice of 'k', yielded the most meaningful and practical customer segments.

Below is a breakdown of our customer segmentation analysis, showcasing the distribution of average percentage spending per category, average dominant category indicators, and average RFM scores for each defined customer segment:

| Customer Segment                             | % Accessories | % Classic Artistry | % Holiday Essentials | % Urban Home & Jewellery | % Vintage Design | Dominant Accessories | Dominant Classic Artistry | Dominant Holiday Essentials | Dominant Urban Home & Jewellery | Dominant Vintage Design | Recency Score | Frequency Score | Monetary Score | Number of Customers |
|----------------------------------------------|---------------|--------------------|----------------------|--------------------------|------------------|----------------------|---------------------------|-----------------------------|--------------------------------|------------------------|---------------|-----------------|----------------|---------------------|
| Active Urban Home & Jewellery Enthusiasts    | 14.93         | 16.67              | 9.15                 | 43.54                    | 15.72            | 0.0                  | 0.0                       | 0.0                         | 1.0                            | 0.0                    | 51.78         | 47.07           | 46.82          | 554                 |
| High-Value Accessory Enthusiasts             | 40.13         | 18.10              | 8.84                 | 16.30                    | 16.64            | 1.0                  | 0.0                       | 0.0                         | 0.0                            | 0.0                    | 66.37         | 72.63           | 70.53          | 534                 |
| Low-Engagement Accessory Seekers             | 54.44         | 13.18              | 6.51                 | 12.78                    | 12.08            | 1.0                  | 0.0                       | 0.0                         | 0.0                            | 0.0                    | 26.66         | 26.70           | 28.70          | 594                 |
| Low-Engagement Vintage Design Shoppers       | 11.90         | 16.72              | 9.28                 | 11.08                    | 51.01            | 0.0                  | 0.0                       | 0.0                         | 0.0                            | 1.0                    | 33.92         | 30.36           | 30.28          | 453                 |
| Occasional Holiday Essentials Shoppers       | 9.89          | 19.77              | 47.86                | 9.38                     | 13.10            | 0.0                  | 0.0                       | 1.0                         | 0.0                            | 0.0                    | 45.52         | 42.58           | 38.49          | 392                 |
| Premium Vintage Design Lovers                | 14.70         | 21.52              | 10.52                | 14.48                    | 38.78            | 0.0                  | 0.0                       | 0.0                         | 0.0                            | 1.0                    | 70.68         | 76.25           | 76.77          | 455                 |
| Understated Classic Artistry Enthusiasts     | 11.16         | 50.08              | 13.13                | 10.46                    | 15.17            | 0.0                  | 1.0                       | 0.0                         | 0.0                            | 0.0                    | 33.59         | 26.80           | 29.05          | 635                 |
| Valued Classic Artistry Connoisseurs         | 13.87         | 37.18              | 15.01                | 15.43                    | 18.51            | 0.0                  | 1.0                       | 0.0                         | 0.0                            | 0.0                    | 66.64         | 71.89           | 71.95          | 722                 |


<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/3treemap.png" alt="Treemap" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/3pie.png" alt="Pie Chart" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/finalcateg.png" alt="Final Categories" />
</div>

<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/finalrfmcat.png" alt="Final RFM Categories" />
</div>

***

## 3. Part III
[View Part III Notebook](https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/Part_3/part_3.ipynb)

<a id="3-1-association-rules-mining"></a>
### 3.1 Association Rules Mining
Association rules mining is a key technique used to discover interesting relationships between variables in large databases. We utilized this method to uncover patterns and associations between different product categories in our dataset. Here's an overview of the metrics and the algorithm used:

#### `Metrics Explained`
- **Support:** This metric measures how frequently an itemset appears in the dataset. A higher support indicates that the itemset is more common.

- **Confidence:** Confidence assesses the likelihood that an item Y is purchased when item X is purchased, expressed as a percentage. It indicates the strength of the implication found in the data.

- **Lift:** Lift compares the likelihood of Y being purchased when X is purchased against the likelihood of Y being purchased independently. It helps identify itemsets that are more likely to be bought together than separately.

#### `FP-Growth Algorithm`
For our analysis, we chose the FP-Growth Algorithm due to its efficiency in mining frequent itemsets without candidate generation, which is particularly useful for large datasets.

#### `Configuration Parameters`
- **Minimum Support:** 0.005  
This threshold filters out itemsets that appear in less than 0.5% of all transactions, focusing analysis on more commonly purchased items.

- **Minimum Lift Threshold:** 1.2  \
Sets a baseline to find item pairs at least 20% more likely to be purchased together than independently, highlighting potentially strong associations.

- **Max Itemset Length:** 2  \
Restricts itemsets to pairs, simplifying the complexity of analysis and making the findings more actionable for marketing strategies like product bundling or promotions.


<a id="3-2-simple-recommendation-system"></a>
### 3.2 Simple Recommendation System
A straightforward recommendation system was developed, utilizing the insights from association rules mining to suggest products to customers based on their historical purchasing patterns, aiming to enhance customer satisfaction and increase sales potential.

#### `"Because you Liked this Item" Recommendations Overview`

<div align="center">

<table>
<thead>
<tr>
<th>Products in Basket</th>
<th>Recommendations</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>JAM MAKING SET WITH JARS</strong></td>
<td>
<strong>JAM MAKING SET PRINTED</strong><br>
Confidence (Probability): 32.25%, Lift: 8<br>
<hr>
<strong>SET OF 3 CAKE TINS PANTRY DESIGN</strong><br>
Confidence: 25.75%, Lift: 4.5<br>
<hr>
<strong>RECIPE BOX PANTRY YELLOW DESIGN</strong><br>
Confidence: 20.29%, Lift: 5<br>
<hr>
<strong>SET OF 4 PANTRY JELLY MOULDS</strong><br>
Confidence: 19.45%, Lift: 5
</td>
</tr>
<tr>
<td><strong>RED KITCHEN SCALES</strong></td>
<td>
<strong>IVORY KITCHEN SCALES</strong><br>
Confidence (Probability): 58%, Lift: 20<br>
<hr>
<strong>MINT KITCHEN SCALES</strong><br>
Confidence: 26.62%, Lift: 18
</td>
</tr>
<tr>
<td><strong>HAND WARMER OWL DESIGN</strong></td>
<td>
<strong>HAND WARMER RED LOVE HEART</strong><br>
Confidence (Probability): 48%, Lift: 20
</td>
</tr>
</tbody>
</table>

</div>

<a id="3-3-network-graph-analysis-of-product-invoices-and-recommendations"></a>
### 3.3 Network Graph Analysis of Product Invoices and Recommendations

In our analysis, we employed association rules mining for both products and product categories to construct a directed graph that encapsulates the relationships between products based on invoice data. 
The graph is constructed using both direct product-to-product associations and broader category-to-category relationships as a fallback mechanism.
This approach aids in identifying key influencers within the product network and simplifies the understanding of complex interdependencies in purchasing behavior.

#### Detailed Process

- **`Initialize Nodes`**
   - **Description:** Initializes graph nodes using unique product descriptions from the data.
   - **Implementation:** Iterates over unique descriptions and adds each as a node to the graph.

- **`Add Edge Attributes`**
   - **Description:** Adds edges between nodes with attributes such as confidence, support, and lift, which signify the strength and significance of the relationships.
   - **Implementation:** For each pair of connected nodes, adds an edge and assigns attributes based on derived association rules. If attributes are missing, defaults to predefined minimal values.

- **`Precompute Descriptions to Categories`**
   - **Description:** Caches category information for each product description to speed up graph construction.
   - **Implementation:** Compiles a dictionary mapping each product description to its respective category, utilizing data from the dataset to facilitate quick access during graph operations.

- **`Build Graph`**
   - **Description:** Constructs the network graph using both direct product-to-product associations and broader category-to-category relationships.
   - **Implementation:** 
     - Processes groups of products by invoices to determine co-purchase relationships.
     - Directly connects products with strong association rules.
     - Where direct connections are lacking, uses category-level associations as a fallback to ensure comprehensive network connectivity.

- **`Connect Isolated Nodes`**
   - **Description:** Integrates nodes that remain isolated after the initial construction into the broader network to maintain connectivity.
   - **Implementation:** 
     - Identifies isolated nodes using NetworkX functionalities.
     - For each isolated node, evaluates potential category connections and links each to the most central node in the most connected category, determined by the highest confidence measures and using degree centrality to assess centrality within categories.

#### Personalized PageRank Recommendations

Personalized PageRank adapts the original PageRank algorithm to focus on a specific node (product), emphasizing paths that start from this node. This provides a personalized ranking of all nodes in the graph based on their relevance to the starting node.
By tailoring the PageRank to a starting node, we ensure that the recommendations are highly relevant to the user's current interest or recent activity.
Personalized PageRank considers the entire graph's structure, which helps identify both direct and indirect relationships between products. This holistic view can reveal less obvious but potentially valuable recommendations.

- **`Example`**
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/PRrec.png" alt="Personalized PageRank Recommendations" />
</div>

#### A* Search Algorithm Recommendations ("Path" of recommended products)

A* search algorithm finds the shortest path between nodes in a graph. In this context, the "shortest path" is based on the confidence of co-purchase edges, with higher confidence leading to lower costs.
The path offers a personalized sequence of products that a customer is likely to purchase. This journey can make the shopping experience more engaging.

- **`Example`**
   - Start Node: 'LUNCH BAG RED RETROSPOT'
   - Goal Node: 'HAND WARMER OWL DESIGN'
<div align="center">
  <img src="https://github.com/semoglou/Machine-Learning-Customer-Segmentation/blob/main/images_outputs/APathrec.png" alt="A* Search Algorithm (Path) Recommendations" />
</div>
