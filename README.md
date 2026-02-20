A Comprehensive Framework for Writing Data Science and Machine Learning Research Articles
Adam Abdellaoui, Lakya Knox, Brandon McLean
    Master's of Computer Science
    Georgia Southern University
Background: Throughout the world countries use various measures and means to compare productivity, prosperity, and to drive policy. Of the measures used gross domestic product (GDP), is among the most trusted indicators of economic prosperity.

Objective: This study aims to determine the accuracy and feasibility of forecasting methods in creating a predictive analysis of the GDP of select G7 nations.

Methods:Data provided by World Bank, US Census datasets, St. Louis Federal Reserve Economic Data.
Results: Predictive analysis, forecasting, and now casting are viable avenues for predicting economic conditions and growth.
Conclusion: Main takeaway and significance.
Keywords: machine-learning, predictive analysis, gross domestic product, forecasting

Funding and Conflicts of Interest
No conflicts of interest, project is unfunded.

Introduction
    Background and Motivation
        From the early 2010s to present day there has been an explosion of wealth generated, primarily in first world economies. This increase in value has largely been made possible by developments in technology and furthermore finance. With an ever more connected world money is circulating faster than ever, with the rise in value of alternative forms of capital such as cryptocurrency the possible gains or losses have never been greater for individuals as well as nations. Predicting best, worst, and average scenarios for GDP would enable policymakers to make more informed decisions to benefit their organization.
    Problem Definition
        Can GDP be predicted up to 90\% using machine learning techniques? 
    Objectives and Contributions
        To develop a model that can be fed economic, time-series data that will provide predictions of no less than 85% accurate results
   
Literature Review        
        To forecast GDP a wide range of predictive approaches must be employed including: time-series models, linear and non-linear regression, as well as Artificial Neural Networks (ANN). Traditional econometric models often struggle when presented with non-linear trends in data, a function that ANN is especially adept at. Artificial neural networks have been proven to be effective at modeling high-dimensional, non-linear relationships.
              
        This study synthesizes findings from more than 100 peer-reviewed articles (2014 - 2023) examining the application of machine learning (ML) and artificial intelligence (AI) in financial forecasting across four major asset classes: equities, cryptocurrencies, commodities, and foreign exchange markets. Using the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) framework, we have identified Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRU), and XGBoost as the most frequently employed predictive models. Root Mean Square Error (RMSE) and Mean Absolute Percentage Error (MAPE) emerge as the most reliable standard evaluation metrics. 

        A review of 64 peer-reviewed journal articles -- selected from an initial pool of 2,977 studies using PRISMA guidelines and a hybrid quantitative-qualitative methodology revealed that 75\% of empirical research used multivariate models that incorporated forward-looking indicators such as search engine trends and economic sentiment measures. LSTM, Random Forest, and Gradient Boosting methods have been identified as the most effective algorithms, 40\% of studies reporting ANN-based models as the best performing. Key implementation strategies include demand decomposition into baseline and volatile components, rolling window validation techniques, and volatility-based model selection using the Coefficient of Variation. The primary limitations reported were: data quality concerns, limited generalizability, interpretability challenges introduced from black-box models, and the disconnect from theoretical framework to real-world volatility.

        Findings from 21 studies examining ML applications for economic forecasting through the lens of Sustainable Development Goal 8 (Decent Work and Economic Growth) evaluates methodologies across three dimensions:  economic (GDP, inflation, exchange rates), social (unemployment, employability), and environmental (carbon dioxide emissions, climate variables, renewable energy adoption). Predominant techniques include Gradient Boosting (XGBoost, LightGBM), Random Forest, Support Vector Machines, and ARIMA, with RMSE, MAE, and MAPE serving as the primary evaluation metrics. A significant research gap emerges from the lack of integrated modeling frameworks that simultaneously address economic growth and decent work outcome. Exisiting studies typically analyze macroeconomic performance or labor market indicators in isolation, overlooking the dynamic interdependencies between these domains.
   
