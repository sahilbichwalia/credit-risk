import streamlit as st
import pandas as pd
import pickle
import numpy as np
from io import StringIO

# Configure page
st.set_page_config(
    page_title="Customer Priority Classification",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stFileUploader>div>div>div>div {
            color: #4CAF50;
        }
        .stSelectbox>div>div>div {
            color: #4CAF50;
        }
        .stNumberInput>div>div>input {
            color: #4CAF50;
        }
        .title {
            color: #4CAF50;
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #e8f5e9;
        }
        .prediction-box {
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
            background-color: #e8f5e9;
            border-left: 5px solid #4CAF50;
        }
        .high-risk {
            border-left: 5px solid #f44336;
            background-color: #ffebee;
        }
        .prediction-details {
            background-color: white;
            border-radius: 5px;
            padding: 1rem;
            margin-top: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .feature-importance {
            margin-top: 1rem;
        }
        .metric-box {
            background-color: white;
            border-radius: 5px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Load model and encoder
@st.cache_resource
def load_model():
    with open("xgb_model_and_encoder.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["label_encoder"]

model, encoder = load_model()

# Preprocessing function
def preprocess_data(df):
    df_copy = df.copy()
    
    # Convert EDUCATION to numeric
    education_mapping = {
        'SSC': 1,
        '12TH': 2,
        'GRADUATE': 3,
        'UNDER GRADUATE': 3,
        'POST-GRADUATE': 4,
        'OTHERS': 1,
        'PROFESSIONAL': 3
    }
    df_copy['EDUCATION'] = df_copy['EDUCATION'].map(education_mapping).fillna(1).astype(int)
    
    # One-hot encode categorical variables
    categorical_cols = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
    df_encoded = pd.get_dummies(df_copy, columns=categorical_cols)
    
    # Ensure all expected columns are present
    expected_columns = [
        'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_closed_L12M',
        'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL',
        'Other_TL', 'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
        'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'num_times_60p_dpd',
        'num_std_12mts', 'num_sub', 'num_sub_6mts', 'num_sub_12mts', 'num_dbt',
        'num_dbt_12mts', 'num_lss', 'recent_level_of_deliq', 'CC_enq_L12m',
        'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m', 'NETMONTHLYINCOME',
        'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'pct_PL_enq_L6m_of_ever',
        'pct_CC_enq_L6m_of_ever', 'HL_Flag', 'GL_Flag', 'EDUCATION',
        'MARITALSTATUS_Married', 'MARITALSTATUS_Single', 'GENDER_F', 'GENDER_M',
        'last_prod_enq2_AL', 'last_prod_enq2_CC', 'last_prod_enq2_ConsumerLoan',
        'last_prod_enq2_HL', 'last_prod_enq2_PL', 'last_prod_enq2_others',
        'first_prod_enq2_AL', 'first_prod_enq2_CC', 'first_prod_enq2_ConsumerLoan',
        'first_prod_enq2_HL', 'first_prod_enq2_PL', 'first_prod_enq2_others'
    ]
    
    # Add missing columns with 0
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match training data
    df_final = pd.DataFrame(index=df_encoded.index)
    for col in expected_columns:
        if col in df_encoded.columns:
            df_final[col] = df_encoded[col]
        else:
            df_final[col] = 0
    
    return df_final

    # Prediction function with probabilities
def make_predictions(data):
    predictions = model.predict(data)
    prediction_labels = encoder.inverse_transform(predictions)
    probabilities = model.predict_proba(data)
    
    # Map to priority class based on risk probability
    def get_priority_class(good_prob):
        if good_prob >= 0.85:  # 85% or higher probability of being good
            return "P1"
        elif good_prob >= 0.70:
            return "P2"
        elif good_prob >= 0.55:
            return "P3"
        else:
            return "P4"
    
    # Calculate priority classes
    priority_classes = [get_priority_class(prob[0]) for prob in probabilities]
    
    return prediction_labels, probabilities, priority_classes

# Function to display feature importance
def display_feature_importance():
    # Replace with actual feature importance from your model if available
    important_features = [
        ("Tot_Missed_Pmnt", "High"),
        ("max_recent_level_of_deliq", "High"),
        ("NETMONTHLYINCOME", "Medium"),
        ("Time_With_Curr_Empr", "Medium"),
        ("num_times_60p_dpd", "High"),
        ("CC_Flag", "Low"),
        ("PL_Flag", "Low"),
        ("EDUCATION", "Low"),
        ("MARITALSTATUS", "Low"),
        ("GENDER", "Low")
    ]
    
    st.subheader("Key Influencing Factors")
    for feature, importance in important_features:
        st.markdown(f"🔹 **{feature}** (*{importance} impact*)")

# Main app
def main():
    st.title("🏦 Customer Priority Classification Model")
    st.markdown("Classify customers into priority segments (P1-P4) based on credit risk assessment.")
    
    # Sidebar
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio("Select Input Method", 
                               ["Upload File", "Manual Input"])
    
    if app_mode == "Upload File":
        st.header("📁 Batch Prediction via File Upload")
        st.markdown("Upload a CSV or Excel file containing applicant data for batch predictions.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success("File uploaded successfully!")
                
                # Show preview
                st.subheader("Data Preview")
                st.write(df.head())
                
                if st.button("Run Predictions"):
                    with st.spinner("Processing data and making predictions..."):
                        try:
                            # Preprocess data
                            df_processed = preprocess_data(df)
                            
                            # Make predictions
                            predictions, probabilities, priority_classes = make_predictions(df_processed)
                            
                            # Add predictions to dataframe
                            df['Predicted_Risk'] = predictions
                            df['Risk_Probability'] = probabilities[:, 1]  # Probability of "Bad" risk
                            df['Priority_Class'] = priority_classes
                            
                            # Show results
                            st.subheader("Prediction Results")
                            st.write(df)
                            
                            # Download button
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name="credit_risk_predictions.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            st.error(f"Processed dataframe shape: {df_processed.shape}")
                            st.error(f"Missing columns: {set(expected_columns) - set(df_processed.columns)}")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    else:  # Manual Input
        st.header("✍️ Single Record Prediction")
        st.markdown("Enter applicant details manually for individual prediction.")
        
        with st.form("manual_input_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Personal Information")
                marital_status = st.selectbox("Marital Status", ["Married", "Single"])
                education = st.selectbox("Education Level", 
                                       ["12TH", "GRADUATE", "SSC", "POST-GRADUATE", 
                                        "UNDER GRADUATE", "OTHERS", "PROFESSIONAL"])
                gender = st.radio("Gender", ["M", "F"])
                net_income = st.number_input("Net Monthly Income (₹)", min_value=0, value=50000)
                employment_time = st.number_input("Time With Current Employer (months)", min_value=0, value=12)
                
            with col2:
                st.subheader("Credit History")
                cc_flag = st.checkbox("Credit Card Holder")
                pl_flag = st.checkbox("Personal Loan Holder")
                hl_flag = st.checkbox("Home Loan Holder")
                gl_flag = st.checkbox("Gold Loan Holder")
                last_product = st.selectbox("Last Product Enquired", 
                                          ["PL", "ConsumerLoan", "AL", "CC", "others", "HL"])
                first_product = st.selectbox("First Product Enquired", 
                                           ["PL", "ConsumerLoan", "others", "AL", "HL", "CC"])
                missed_payments = st.number_input("Total Missed Payments", min_value=0, value=0)
                
            st.subheader("Financial Metrics")
            col3, col4 = st.columns(2)
            
            with col3:
                pct_tl_open = st.number_input("% Trade Lines Open (Last 6 Months)", min_value=0.0, max_value=100.0, value=50.0)
                pct_tl_closed = st.number_input("% Trade Lines Closed (Last 6 Months)", min_value=0.0, max_value=100.0, value=10.0)
                tot_tl_closed = st.number_input("Total Trade Lines Closed (Last 12 Months)", min_value=0, value=1)
                pct_tl_closed_12m = st.number_input("% Trade Lines Closed (Last 12 Months)", min_value=0.0, max_value=100.0, value=15.0)
                
            with col4:
                cc_tl = st.number_input("Credit Card Trade Lines", min_value=0, value=0)
                home_tl = st.number_input("Home Loan Trade Lines", min_value=0, value=0)
                pl_tl = st.number_input("Personal Loan Trade Lines", min_value=0, value=0)
                secured_tl = st.number_input("Secured Trade Lines", min_value=0, value=0)
            
            # More metrics in an expander for better UI
            with st.expander("Additional Metrics"):
                col5, col6 = st.columns(2)
                
                with col5:
                    unsecured_tl = st.number_input("Unsecured Trade Lines", min_value=0, value=0)
                    other_tl = st.number_input("Other Trade Lines", min_value=0, value=0)
                    age_oldest_tl = st.number_input("Age of Oldest Trade Line (months)", min_value=0, value=24)
                    age_newest_tl = st.number_input("Age of Newest Trade Line (months)", min_value=0, value=6)
                    time_since_payment = st.number_input("Time Since Recent Payment (days)", min_value=0, value=30)
                    max_delinquency = st.number_input("Max Recent Level of Delinquency", min_value=0, value=0)
                    num_deliq_6_12mts = st.number_input("Number of Delinquencies (6-12 months)", min_value=0, value=0)
                    num_times_60p_dpd = st.number_input("Number of Times 60+ Days Past Due", min_value=0, value=0)
                
                with col6:
                    cc_enquiries = st.number_input("Credit Card Enquiries (Last 12 Months)", min_value=0, value=0)
                    pl_enquiries = st.number_input("Personal Loan Enquiries (Last 12 Months)", min_value=0, value=0)
                    recent_enquiry_time = st.number_input("Time Since Recent Enquiry (days)", min_value=0, value=90)
                    enquiries_3m = st.number_input("Enquiries (Last 3 Months)", min_value=0, value=0)
                    pct_pl_enq = st.number_input("% PL Enquiries Last 6M of Ever", min_value=0.0, max_value=100.0, value=0.0)
                    pct_cc_enq = st.number_input("% CC Enquiries Last 6M of Ever", min_value=0.0, max_value=100.0, value=0.0)
                    num_std_12mts = st.number_input("Number of Standard Accounts (12 months)", min_value=0, value=0)
                    recent_level_of_deliq = st.number_input("Recent Level of Delinquency", min_value=0, value=0)
                
                col7, col8 = st.columns(2)
                
                with col7:
                    num_sub = st.number_input("Number of Substandard Accounts", min_value=0, value=0)
                    num_sub_6mts = st.number_input("Number of Substandard Accounts (6 months)", min_value=0, value=0)
                    num_sub_12mts = st.number_input("Number of Substandard Accounts (12 months)", min_value=0, value=0)
                
                with col8:
                    num_dbt = st.number_input("Number of Doubtful Accounts", min_value=0, value=0)
                    num_dbt_12mts = st.number_input("Number of Doubtful Accounts (12 months)", min_value=0, value=0)
                    num_lss = st.number_input("Number of Loss Accounts", min_value=0, value=0)
            
            submit_button = st.form_submit_button("Classify Customer Priority")
            
            if submit_button:
                # Create a single-row dataframe with all required features
                data = {
                    'pct_tl_open_L6M': pct_tl_open,
                    'pct_tl_closed_L6M': pct_tl_closed,
                    'Tot_TL_closed_L12M': tot_tl_closed,
                    'pct_tl_closed_L12M': pct_tl_closed_12m,
                    'Tot_Missed_Pmnt': missed_payments,
                    'CC_TL': cc_tl,
                    'Home_TL': home_tl,
                    'PL_TL': pl_tl,
                    'Secured_TL': secured_tl,
                    'Unsecured_TL': unsecured_tl,
                    'Other_TL': other_tl,
                    'Age_Oldest_TL': age_oldest_tl,
                    'Age_Newest_TL': age_newest_tl,
                    'time_since_recent_payment': time_since_payment,
                    'max_recent_level_of_deliq': max_delinquency,
                    'num_deliq_6_12mts': num_deliq_6_12mts,
                    'num_times_60p_dpd': num_times_60p_dpd,
                    'num_std_12mts': num_std_12mts,
                    'num_sub': num_sub,
                    'num_sub_6mts': num_sub_6mts,
                    'num_sub_12mts': num_sub_12mts,
                    'num_dbt': num_dbt,
                    'num_dbt_12mts': num_dbt_12mts,
                    'num_lss': num_lss,
                    'recent_level_of_deliq': recent_level_of_deliq,
                    'CC_enq_L12m': cc_enquiries,
                    'PL_enq_L12m': pl_enquiries,
                    'time_since_recent_enq': recent_enquiry_time,
                    'enq_L3m': enquiries_3m,
                    'NETMONTHLYINCOME': net_income,
                    'Time_With_Curr_Empr': employment_time,
                    'CC_Flag': int(cc_flag),
                    'PL_Flag': int(pl_flag),
                    'pct_PL_enq_L6m_of_ever': pct_pl_enq,
                    'pct_CC_enq_L6m_of_ever': pct_cc_enq,
                    'HL_Flag': int(hl_flag),
                    'GL_Flag': int(gl_flag),
                    'MARITALSTATUS': marital_status,
                    'EDUCATION': education,
                    'GENDER': gender,
                    'last_prod_enq2': last_product,
                    'first_prod_enq2': first_product
                }
                
                df = pd.DataFrame(data, index=[0])
                
                try:
                    # Debug: Show dataframe before preprocessing
                    with st.expander("Raw Input Data"):
                        st.write(df)
                    
                    # Preprocess data
                    df_processed = preprocess_data(df)
                    
                    # Debug: Show processed dataframe
                    with st.expander("Processed Data"):
                        st.write(df_processed)
                        st.write(f"Shape: {df_processed.shape}")
                    
                    # Make prediction
                    prediction, probabilities, priority_classes = make_predictions(df_processed)
                    risk_label = prediction[0]
                    good_prob = probabilities[0][0] * 100
                    bad_prob = probabilities[0][1] * 100
                    priority_class = priority_classes[0]
                    
                    # Display result
                    st.subheader("Prediction Result")
                    
                    # Determine box color and styling based on priority class
                    if priority_class == "P1":
                        box_class = "prediction-box"
                        color = "#4CAF50"  # Green
                        icon = "✅"
                        risk_text = "Very Low Risk"
                    elif priority_class == "P2":
                        box_class = "prediction-box"
                        color = "#8BC34A"  # Light Green
                        icon = "✓"
                        risk_text = "Low Risk"
                    elif priority_class == "P3":
                        box_class = "prediction-box high-risk"
                        color = "#FFC107"  # Amber
                        icon = "⚠️"
                        risk_text = "Medium Risk"
                    else:  # P4
                        box_class = "prediction-box high-risk"
                        color = "#f44336"  # Red
                        icon = "⚠️"
                        risk_text = "High Risk"
                    
                    st.markdown(f"""
                    <div class="{box_class}">
                        <h3 style="color:{color};">{icon} {priority_class} Customer ({risk_text})</h3>
                        <p>This applicant has a <b>{good_prob:.1f}%</b> probability of being a good customer.</p>
                        <p>Priority Classification: <b>{priority_class}</b> - {risk_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed prediction metrics
                    with st.expander("Detailed Prediction Analysis"):
                        col5, col6 = st.columns(2)
                        
                        with col5:
                            st.metric("Probability of Good Credit", f"{good_prob:.1f}%")
                            st.metric("Probability of Bad Credit", f"{bad_prob:.1f}%")
                        
                        with col6:
                            st.metric("Priority Class", priority_class)
                            st.metric("Risk Category", risk_label)
                            
                        # Add priority class explanation
                        st.markdown("""
                        **Priority Class Explanation:**
                        - **P1**: Very Low Risk (≥85% good probability) - Prime customers
                        - **P2**: Low Risk (70-84% good probability) - Good customers
                        - **P3**: Medium Risk (55-69% good probability) - Average risk
                        - **P4**: High Risk (<55% good probability) - Higher risk
                        """)
                        
                        # Feature importance
                        display_feature_importance()
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

if __name__ == "__main__":
    main()