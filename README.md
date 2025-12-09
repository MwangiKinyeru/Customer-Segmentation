# Customer-Segmentation
This project segments retail customers into behavioral groups using RFM analysis and K-Means clustering

## Overview

A comprehensive unsupervised learning solution that segments retail customers into behavioral groups using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering. The system identifies distinct customer segments for targeted marketing strategies and personalized engagement.

## Features

>> * RFM Feature Engineering: Computes Recency, Frequency, and Monetary metrics from transaction data
>> * Outlier Detection: Identifies high-value customers using statistical methods (IQR)
>> * Dual Segmentation: Combines rule-based labeling for outliers with K-Means clustering for regular customers
>> * Web Application: Interactive interface for real-time customer segmentation
>> * Actionable Insights: Provides business recommendations for each customer segment

## Architecture

<p align="center">
  <img src="https://github.com/MwangiKinyeru/Customer-Segmentation/blob/main/Images/Project%20flow.png" width="80%" alt="Customer Segmentation Workflow Diagram" />
</p>

## Technical Stack
>> * Data Processing: Python, Pandas, NumPy
>> * Machine Learning: Scikit-learn (K-Means, StandardScaler)
>> * Visualization: Matplotlib, Seaborn
>> * Web Framework: Flask
>> * Deployment: Render

## How it works
This project implements a tiered customer segmentation system that combines rule-based logic with machine learning. It first applies business rules to identify high-value outliers—customers exceeding spending or frequency thresholds are classified as High-Spenders, Power-Shoppers, or Elite VIPs. For regular customers falling within normal ranges, the system uses a trained K-Means clustering model to assign them to one of four behavioral segments (Regular, Lapsed, Occasional, or Premium). The entire process is packaged into a web application where users input customer metrics and instantly receive segment classifications with actionable business insights, enabling data-driven marketing strategies and personalized customer engagement.

## UI Preview

<p float="left">
  <img src="https://github.com/MwangiKinyeru/Customer-Segmentation/blob/main/Images/Image.png" width="45%" />
  <img src="https://github.com/MwangiKinyeru/Customer-Segmentation/blob/main/Images/image2.png" width="45%" />
</p>


## Deployment

The Customer segmentatiion system was Deployed via render as a web application. To access the the finance bot click the link below
>>> [Deployment Link](https://customer-segmentation-g0b9.onrender.com)

