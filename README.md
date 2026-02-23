# Gold-Trading-Actions
Gold Trading Action Classification (Academic Project in Kaggle Competition)

Time-Series Multi-Class Prediction | Log Loss Optimization

----> Overview :

      This repository contains our solution to a Kaggle-style competition conducted as part of a Machine Learning course at the Department of Management Studies, National Institute of Technology Calicut.
      
      The objective was to build a predictive model that classifies each trading day of Gold Futures (GC=F) into one of three actions:
      0 → Hold / No Action
      1 → Buy
      2 → Sell
      
      The evaluation metric was Multi-Class Log Loss, requiring probabilistic predictions for each class.
      Our final model achieved a Log Loss of 0.7269, securing 23rd position out of 200 participants.

----> Problem Statement : 

      Gold futures are closely monitored by traders and investors.
            Given historical daily market data, the task was to:
            
                  1) Analyze time-series trends
                  2) Engineer predictive features
                  3) Train a classification model
                  4) Output class probabilities (not hard labels)
                  5) Each submission required probabilities for all three classes summing to 1.0.

----> My Contribution :

      This was a team project (4 members). 
            My responsibilities included:
    
                  1) Model selection and tuning
                  2) Designing the final Random Forest configuration
                  3) Implementing time-series-aware data splitting
                  4) Final model training and submission generation
                  5) Editing and structuring the final presentation
        
----> Dataset Description :

      The dataset consisted of:
            Training file: Features + Action label
            Test file: Features only (Action to be predicted)
            
      Sample submission: Required probability format
      Each row represented a daily trading observation.
      
----> The model was evaluated using:

      Log Loss
      Since Log Loss penalizes overconfident incorrect predictions, probability calibration was critical.
            
----> Methodology :

    1. Exploratory Data Analysis (EDA)
          Checked class distribution and imbalance
          Analyzed gold price volatility trends
          Performed ADF and KPSS test on gold price percent change
          Examined temporal behavior of price movements
          Identified correlation patterns
          
    2. Feature Engineering:
          Feature engineering significantly improved model performance.
          Lag Features
          Previous-day price differences
          Multi-day rolling windows
          Volatility-based signals
          Lag-based indicators played a major role in reducing Log Loss.
          External Signal Integration
          Incorporated S&P 500 index data as a macro-market indicator
          Merged aligned time-series data using trading dates
          External market signals helped improve generalization.
          
    3. Time-Series Validation Strategy
          To prevent data leakage:
              We avoided random shuffling
              Used chronological splitting
              Ensured training data preceded validation data
              This preserved the real-world forecasting constraint.
              
    4. Model Development
          The final model was a Random Forest Classifier.
              rf_final = RandomForestClassifier(
                  n_estimators=400,
                  max_depth=20,
                  min_samples_split=10,
                  min_samples_leaf=5,
                  max_features='sqrt',
                  random_state=42,
                  class_weight='balanced',
                  n_jobs=-1
              )
              
          Due to computational limitations, hyperparameter tuning was performed through controlled manual experimentation rather than exhaustive grid search.
          
----> Key considerations:

      Handling class imbalance using class_weight='balanced'
      Controlling model complexity via depth and leaf constraints
      Parallel computation (n_jobs=-1) for efficiency
      
----> Results :

      Final Validation Log Loss: 0.7269
      Competition Rank: 23 / 200 participants
      
      Performance improved substantially after:
      Introducing lag features
      Applying time-aware validation
      Tuning tree depth and sampling parameters
      
----> Key Insights :

      1) Short-term lagged price signals significantly influence trading action classification
      2) Proper time-series validation is essential to avoid misleading performance
      3) Macro-market indicators (S&P 500) provide useful contextual signals
      4) Log Loss optimization requires well-calibrated probability outputs rather than high raw accuracy
      
----> Future Improvements :

      1) Incorporating additional macroeconomic indicators (CPI, bond yields)
      2) Testing gradient boosting models (XGBoost, LightGBM)
      3) Exploring sequence models (LSTM) for temporal dependencies
      4) Probability calibration techniques (Platt scaling, isotonic regression)
      
----> Presentation :
      A 10-minute structured presentation explaining our methodology and findings is available in the /presentation directory.
