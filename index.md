# Data Science Portfolio
---

## Statistics and Exploratory Data Analyics (EDA)

### Lending Club Case Study

This risk analytic assignment involved solving a case study to gain an understanding of how real business problems are addressed using Exploratory Data Analysis (EDA). It aimed to provide insights into risk analytics in the banking and financial services domain, demonstrating how data can be leveraged to minimize the risk of financial losses in lending.

The primary focus was to identify loan applicants who posed a higher risk of default, as such applicants are the primary contributors to credit loss. 

The objective was to analyze the data and determine the key driving factors (or driver variables) that indicated a higher likelihood of loan default. Understanding these variables and their significance  would enable the company to reduce risky loans and mitigate credit loss. The insights gained could be utilized for effective portfolio management and risk assessment.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/LendingClubCaseStudy)

<center><img src="images/fraud_detection.jpg"/></center>

## Machine Learning 1 (ML1)

### Bike Sharing Case Study

The assignment involved building a multiple linear regression model to predict the demand for shared bikes using the provided independent variables. It aimed to understand how the demand varied with different features and the influence of various factors on bike-sharing demand in the American market.

The primary objectives were to identify the significant variables affecting bike-sharing demand, develop a quantitative model to predict demand, and evaluate how accurately these variables could explain and predict the demand. The analysis provided insights to help the company understand the factors driving demand and optimize their bike-sharing system accordingly.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/BikeSharingMLGCaseStudy.git)

<center><img src="images/fraud_detection.jpg"/></center>

## Machine Learning 1 (ML2)

### Telecom Churn Study

In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition. For many incumbent operators, retaining high profitable customers is the number one business goal. To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. 

In this project, we analyze customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn. The goal is to build a machine learning model that is able to predict churning customers based on the features provided for their usage.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/BikeSharingMLGCaseStudy.git)

<center><img src="images/fraud_detection.jpg"/></center>

## Deep Learning (DL)

### Melanoma Detection Assignment

In this assignment, the focus was on addressing melanoma, a type of cancer responsible for 75% of skin cancer deaths and potentially fatal if not detected early. The objective was to create a solution or model capable of evaluating medical images to assist dermatologists in identifying the presence of melanoma. By doing so, the model aimed to reduce the significant manual effort involved in diagnosis and improve early detection.

The input dataset consisted of 2,357 images spanning nine malignant and benign oncological disease classes. These images were sourced from the International Skin Imaging Collaboration (ISIC), providing a diverse and reliable dataset for training and evaluation.

To enhance the model's performance, data augmentation techniques were applied to address underfitting, overfitting, and class imbalances in the dataset. This ensured the model's robustness and improved its predictive accuracy.

Through this project,a powerful model was built, applying advanced techniques to tackle a critical real-world problem in the medical field. This hands-on experience not only strengthened technical skills but also showcased the potential of AI in improving healthcare outcomes. 

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/LendingClubCaseStudy)

<center><img src="images/fraud_detection.jpg"/></center>

### Gesture Recognition Assignment

In this assignment, the goal was to design a smart-TV feature capable of recognizing five different user gestures to control the TV without a remote. By detecting and interpreting gestures, users could interact with the TV seamlessly, enhancing convenience and accessibility.

To achieve this, videos of gestures were analyzed using neural networks. Two prominent architectures were commonly used for video processing:

* #### CNN + RNN Architecture  - 

This approach leveraged the strengths of both Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). Each frame of the video was passed through a CNN, which extracted a feature vector representing the key information from that image. These feature vectors, representing the entire sequence of frames, were then fed into an RNN to model the temporal dependencies and analyze the sequence as a whole. This method was something you were already familiar with in theory.
 
* #### 3D Convolutional Network Architecture - 

A natural evolution of standard CNNs, 3D Convolutional Networks extended the convolution operation to three dimensions (height, width, and time). This allowed the model to simultaneously capture spatial and temporal features from video data. Unlike the CNN + RNN combination, 3D convolutions directly processed video frames as a block, making it a more unified approach for video-based tasks.
 
In this project, both architectures were explored to build the gesture recognition feature. This hands-on project offered an exciting opportunity to apply theoretical knowledge to create an innovative, real-world application.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/LendingClubCaseStudy)

<center><img src="images/fraud_detection.jpg"/></center>

## Natural Language Processing (NLP)

### Identifying Entities in Healthcare Data

The assignment involved developing a custom Named Entity Recognition (NER) model to extract diseases and their corresponding treatments from a medical text dataset. The task simulated a scenario for a health tech company,  aiming to organize and interpret unstructured medical data generated through online consultations and prescriptions.

The dataset included sentences with implicit mentions of diseases and treatments, requiring the application of a Conditional Random Field (CRF) model to identify and map these entities. Key steps involved processing the data into sentence format, defining features for the CRF model, training the model using the provided training dataset, and evaluating its performance on test data.

Finally, the extracted information was structured into a dictionary where diseases were keys and their probable treatments were values, enabling better data organization and usability for medical applications.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/LendingClubCaseStudy)

<center><img src="images/fraud_detection.jpg"/></center>

### Automatic Ticket Classification

The assignment involved automating a financial company's customer support ticket system. The goal was to classify unstructured customer complaints into categories based on the company's products and services, enabling faster issue resolution and improved customer satisfaction.

Using non-negative matrix factorization (NMF) for topic modeling, patterns and recurring words in the complaints were analyzed to identify the key features for each category. The complaints were segregated into five clusters: credit card/prepaid card, bank account services, theft/dispute reporting, mortgages/loans, and others.

The insights gained from topic modeling were then used to train supervised models such as logistic regression or decision trees, enabling the classification of new customer complaints into their relevant categories for efficient handling.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/LendingClubCaseStudy)

<center><img src="images/fraud_detection.jpg"/></center>

## ML Operations (MLOps)

### Lead Generation Pipeline Automation

The assignment involved building a lead-scoring system for an EdTech startup, to reduce customer acquisition costs (CAC) and improve the efficiency of their sales process. The primary objective was to categorize leads based on their propensity to purchase EdTech courses, removing junk leads and streamlining the lead conversion process. The system focused on predicting the L2AC (Leads to Application Completion) flag by analyzing lead origins and interactions with the platform.

The assignment emphasized three key MLOps principles:
* **Reproducibility:** Ensured by maintaining version-controlled code, creating consistent data pipelines, and standardizing model training and evaluation processes.
* **Automation:** Achieved through automated data preprocessing, model training pipelines, and deployment processes, reducing manual intervention.
* **Collaboration:** Facilitated by clear communication and coordination between the data science team and the sales team to align on business metrics and requirements.

The developed system helped the sales team prioritize leads more effectively, addressing inefficiencies caused by junk leads and contributing to  long-term profitability.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/LendingClubCaseStudy)

<center><img src="images/fraud_detection.jpg"/></center>

## Data Strategy

### Data Science Adoption Strategy

The assignment involved designing a Data Science (DS) adoption strategy for an e-commerce company, to align with its business goals of increasing active customers, revenue, service efficiency, and customer experience. The project focused on prioritizing use cases based on feasibility, complexity, and value while addressing the constraints of a newly formed DS team with limited resources.

The use cases considered included delivery date prediction, sentiment analysis, customer churn prediction, customer acquisition cost optimization, fraud detection, and price optimization. Each use case was analyzed based on its impact on the organisations objectives, feasibility, data and skill requirements, potential changes to current processes, and expected monetary benefits.

**Techniques Used:**

* **Prioritization Framework:** A systematic approach was used to evaluate and rank the use cases, balancing complexity and business impact.
* **Data Strategy:** Leveraged the provided dataset and data architecture to assess the available data and define data requirements.
* **Proof of Concept (PoC):** Proposed the development of PoCs to validate the feasibility and effectiveness of DS solutions.
* **Monetary Benefit Estimation:** Developed a framework for quantifying the expected benefits of each use case based on assumptions and data-driven insights.
* **Success Metrics:** Defined appropriate KPIs, such as increased L2AC, reduced customer churn, optimized pricing, and more accurate delivery predictions, to evaluate the success of each project.
 
The assignment highlighted the strategic role of DS in transforming  organisations operations and achieving its objectives. It demonstrated how to align technical solutions with business goals, propose actionable roadmaps, and quantify potential benefits to secure buy-in from senior management.


[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/LendingClubCaseStudy)

<center><img src="images/fraud_detection.jpg"/></center>

## Deployment and Capstone (DC)

### Credit Card Fraud Detection

The assignment focused on developing a machine learning solution to predict fraudulent credit card transactions using a dataset from Kaggle, comprising 284,807 transactions, of which only 492 (0.172%) were fraudulent. The primary challenge lay in addressing the highly imbalanced nature of the dataset, which is critical for building accurate and reliable models.

**Techniques Used:**

* **Data Understanding:**
Analyzed the dataset features, including principal components derived from PCA for confidentiality, and identified relevant variables such as transaction amount, time, and fraud class.
* **Exploratory Data Analysis (EDA):**
Performed univariate and bivariate analyses to understand data distributions.
Addressed skewness where necessary to ensure robustness in model performance.
* **Train/Test Split & Validation:**
Employed k-fold cross-validation to ensure the minority class (frauds) was adequately represented in all test folds, improving model reliability on unseen data.
* **Sampling Techniques:**
Applied techniques like oversampling (SMOTE) or undersampling to handle class imbalance and improve the model's ability to detect fraud accurately.
* **Model Building & Hyperparameter Tuning:**
Built various machine learning models such as Logistic Regression, Random Forest, and Gradient Boosting, fine-tuning their hyperparameters to achieve optimal performance.
* **Model Evaluation:**
Evaluated models using metrics suited for imbalanced datasets, such as precision, recall, F1-score, and ROC-AUC, with a focus on accurately identifying fraudulent transactions over non-fraudulent ones.

The project demonstrated the application of machine learning to solve a critical business problem for banks, highlighting its potential to proactively detect fraudulent activities, minimize financial losses, and strengthen customer trust. This approach reduces manual effort, mitigates chargebacks, and enhances the overall fraud prevention mechanism in the banking industry. 

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/abe-twistedtech/LendingClubCaseStudy)

<center><img src="images/fraud_detection.jpg"/></center>

