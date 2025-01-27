import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from io import BytesIO

try:
    # Streamlit title
    st.title("Income Prediction - Model Training & Prediction")

    # Sidebar options
    option = st.sidebar.selectbox("Select an option", ["Train Model", "Make Prediction"])

    # Function to convert dataframe to CSV for downloading
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    # Function to train the model
    def train_model(data):
        # Drop unnecessary columns
        data = data.drop(['ID'], axis=1, errors='ignore')
        data = data.drop(["class", "education_institute", "unemployment_reason", "is_labor_union", "occupation_code_main",
                          "under_18_family", "veterans_admin_questionnaire", "residence_1_year_ago", "old_residence_reg",
                          "old_residence_state", "migration_prev_sunbelt"], axis=1, errors='ignore')

        # Convert gender: Male to 1, Female to 0
        data['gender'] = data['gender'].apply(lambda x: 1 if x != ' Female' else 0)

        # Convert income_above_limit: Below limit to 0, Above limit to 1
        data["income_above_limit"] = data["income_above_limit"].apply(lambda x: 1 if x == "Above limit" else 0)

        # Create new features
        data['wwpy+te'] = data["working_week_per_year"] + data["total_employed"]
        data['wwpy-oc'] = data["working_week_per_year"] - data["occupation_code"]

        # Final dataset columns
        final_data = data[['working_week_per_year', 'gains', 'total_employed', 'industry_code', 'stocks_status', 'wwpy+te', 'wwpy-oc', 'income_above_limit']]

        # Create two different dataframes for majority and minority class
        df_majority = final_data[final_data['income_above_limit'] == 0]
        df_minority = final_data[final_data['income_above_limit'] == 1]

        # Upsample minority class to match the majority class
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
        final_df = pd.concat([df_minority_upsampled, df_majority])

        # Split the data into X and y for training
        X = final_df.drop(["income_above_limit"], axis=1)  # Features
        y = final_df["income_above_limit"]  # Target

        # Train-test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # Initialize RandomForestClassifier
        rfc = RandomForestClassifier(random_state=42)

        # Hyperparameters for GridSearchCV
        param_grid = {
            'n_estimators': [5, 15],
            'max_features': ['sqrt'],  # Corrected from 'auto' to 'sqrt'
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True]
        }

        # Perform GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_rfc = grid_search.best_estimator_

        return best_rfc, grid_search.best_params_

    # Option 1: Train Model
    if option == "Train Model":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load the uploaded CSV file
            data = pd.read_csv(uploaded_file)

            # Check if model is already trained
            if "trained_model" not in st.session_state:
                with st.spinner("Training the model... Please wait."):
                    try:
                        # Train the model
                        st.session_state.trained_model, best_params = train_model(data)

                        # Display the best hyperparameters
                        st.subheader("Best Hyperparameters")
                        st.write(best_params)

                        # Provide download button for the trained model
                        model_file = BytesIO()
                        joblib.dump(st.session_state.trained_model, model_file)
                        model_file.seek(0)

                        st.download_button(
                            label="Download Trained Model",
                            data=model_file,
                            file_name="trained_model.joblib",
                            mime="application/octet-stream"
                        )

                        st.success("Model downloaded successfully!")
                    except Exception as e:
                        st.error(f"Error while training the model: {str(e)}")
            else:
                st.warning("Model is already trained.")

    # Option 2: Make Prediction
    if option == "Make Prediction":
        model_file = st.sidebar.file_uploader("Upload your trained model (.joblib)", type=["joblib"])
        test_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])

        if model_file is not None and test_file is not None:
            # Load the trained model
            model = joblib.load(model_file)

            # Load the test data
            test_data = pd.read_csv(test_file)

            # Show a preview of the test dataset
            st.subheader("Test Data Preview")
            st.write(test_data.head())

            # Drop unnecessary columns (ensure they match the training process)
            test_data = test_data.drop(['ID'], axis=1, errors='ignore')

            # Convert gender: Male to 1, Female to 0
            test_data['gender'] = test_data['gender'].apply(lambda x: 1 if x != ' Female' else 0)

            # Create the same features ('wwpy+te' and 'wwpy-oc') for the test data as done during training
            test_data['wwpy+te'] = test_data['working_week_per_year'] + test_data['total_employed']
            test_data['wwpy-oc'] = test_data['working_week_per_year'] - test_data['occupation_code']

            # Ensure test data has the same features as the training data
            test_features = test_data[['working_week_per_year', 'gains', 'total_employed', 'industry_code', 'stocks_status', 'wwpy+te', 'wwpy-oc']]

            # Predict with the loaded model
            predictions = model.predict(test_features)

            # Display only the predicted values
            prediction_df = pd.DataFrame(predictions, columns=["Predicted Income Above Limit"])

            st.subheader("Prediction Results")
            st.write(prediction_df)  # Display predicted values

            # Option to download the predictions
            st.download_button(
                label="Download Predictions as CSV",
                data=convert_df_to_csv(prediction_df),
                file_name="predictions.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please upload both the trained model and test data to proceed.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
