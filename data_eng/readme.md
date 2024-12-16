This folder contains Data Engineering project on NFL and NSL KDD datasets, with a primary focus on building and implementing data pipelines to handle various data processing and engineering tasks. Below is a detailed description of the tasks and the data pipeline workflows I created:


Task 1: Protocol Type Analysis
Pipeline Objective: Extract insights from the NSL-KDD dataset by analyzing protocol types for logged-in and non-logged-in users.
Steps in the Pipeline:
Load the NSL-KDD dataset into a SPARK DataFrame.
Perform filtering and grouping based on user login status and protocol_type.
Output the counts for each protocol_type.

Task 2: Data Transformation
Pipeline Objective: Transform the NSL-KDD dataset to focus on normal traffic data and modify its attributes for further use.
Steps in the Pipeline:
Filter rows to include only "normal" traffic.
Override the protocol_type column to set all values to "tcp".
Generate a transformed DataFrame.
Verify the results by displaying a sample of the transformed data.


Task 3: Cloud Computing with Spark
Pipeline Objective: Scale the above pipelines to a distributed environment using Apache Spark on a cloud-hosted cluster.
Steps in the Pipeline:
Load the NSL-KDD dataset into a Spark DataFrame.
Recreate the protocol type analysis and data transformation pipelines in Spark.
Execute the workflows on the cloud-hosted cluster.
Capture the results and cloud environment details via screenshots.


Task 4: Database Integration Pipeline
Pipeline Objective: Ingest both training and testing datasets from NSL-KDD into a PostgreSQL database for centralized storage.
Steps in the Pipeline:
Load the training and testing datasets into Spark.
Add a new column to differentiate between training and testing data.
Create a PostgreSQL table schema to store the combined data.
Write the merged dataset into PostgreSQL using an automated ingestion pipeline.
Retrieve and verify the data with a query, including sample outputs.


Task 5: Feature Engineering for NFL Pro Bowl Data
Pipeline Objective: Create a data preprocessing and feature engineering pipeline for predictive modeling using the NFL Pro Bowl Plays.csv dataset.
Steps in the Pipeline:
Phase 1 – Data Cleaning and Preprocessing:
Handle missing values and remove unnecessary columns.
Create new features relevant to predicting the PlayResult column.
Phase 2 – Feature Scaling and Transformation:
Scale the dataset to normalize feature ranges.
Use pipelines to standardize preprocessing steps.
Final Output: The resulting scaled dataset is ready for machine learning model training.

