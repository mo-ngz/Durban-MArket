import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_save_path = "dbnmarket.pkl"
with open(model_save_path, 'rb') as file:
    loaded_model = pickle.load(file)


# Sidebar with input field descriptions
st.sidebar.header("Description of The Required Input Fields")
st.sidebar.markdown("**Province**: The provinces producing Onion brown.")
st.sidebar.markdown("**Size_Grade**: The sizes of the brown onion packages.")
st.sidebar.markdown("**Weight_Kg**: The weight of onion brown in kilogram.")
st.sidebar.markdown("**Low_Price**: The lowest price the onion brown cost in the market.")
st.sidebar.markdown("**Sales_Total**: The total price purchase onion brown.")
st.sidebar.markdown("**Stock_On_Hand**: The onion brown stock currently available in the warehouse.")


# Streamlit interface
st.title("POTATO(WASHED)MONDIAL Average Price Prediction")
st.image("https://images.tridge.com/1600x800/story-thumbnail/39/18/93/39189378bdd22d8ca1bdbc4f98ddd78f51f6944c/blob.jpeg")

# Function to preprocess user inputs and make predictions
def predict_price(Province,Size_Grade,Weight_Kg,Sales_Total,Low_Price,High_Price,Total_Kg_Sold,Stock_On_Hand):
    # Assuming label encoding mappings are known
    province_mapping = {'NORTHERN CAPE':5, 'WESTERN CAPE - CERES':11, 'WEST COAST':10,'SOUTH WESTERN FREE STATE':7, 'WESTERN FREESTATE':12,'KWAZULU NATAL':1,
                        'OTHER AREAS':6, 'TRANSVAAL':9, 'EASTERN FREESTATE':0,'MPUMALANGA':2,'NORTH EASTERN CAPE':3,'NORTH WEST':4,'SOUTHERN CAPE':8} 
   # Replace with actual mappings

    size_grade_mapping = {'1L':0, '1M':1, '1R':2, '1S':3, '1U':4,'1X':5,'1Z':6,'2L':7,'2M':8,'2R':9,'2U':10,'2X':11,'2Z':12,'3L':13,'3M':14,'3R':15,'3S':16,'3U':17,'3X':18,'3X':19,'3Z':20,'4L':21,'4M':22,'4R':23,'4S':24,'4U':25,'4Z':26}
    # Convert categorical inputs to numerical using label encoding
    province_encoded = province_mapping.get(Province,-1)  # Use -1 for unknown categories
    size_grade_encoded = size_grade_mapping.get(Size_Grade,-1)  # Use -1 for unknown categories
    

    # Prepare input data as a DataFrame for prediction
    input_data = pd.DataFrame([[province_encoded,size_grade_encoded,Weight_Kg,Sales_Total,Low_Price,High_Price,Total_Kg_Sold,Stock_On_Hand]])
     # Rename columns to string names
     # Make sure the feature names match the model's expectations
    input_data.columns = ['Province','Size_Grade','Weight_Kg','Sales_Total','Low_Price','High_Price','Total_Kg_Sold','Stock_On_Hand']

    # Make prediction
    predicted_price = loaded_model.predict(input_data)

    return predicted_price[0]

col1,col2 = st.columns(2)
with col1:
    Province= st.selectbox('Province',['EASTERN FREESTATE','KWAZULU NATAL','MPUMALANGA','NORTH EASTERN CAPE','NORTH WEST','NORTHERN CAPE', 'OTHER AREAS','SOUTH WESTERN FREE STATE','SOUTHERN CAPE','TRANSVAAL','WEST COAST','WESTERN CAPE - CERES', 'WESTERN FREESTATE'])
    Size_Grade= st.selectbox('size grade',['1L', '1M', '1R', '1S', '1U','1X','1Z','2L','2M','2R','2U','2X','2Z','3L','3M','3R','3S','3U','3X','3X','3Z','4L','4M','4R','4S','4U','4Z'])
    Weight_Kg = st.number_input("weight per kilo", min_value=0.0)
    Low_Price=st.number_input("Low_Price", min_value=0)
    
with col2:
    Sales_Total= st.number_input('total sale', min_value=0)
    Stock_On_Hand= st.number_input('stock on hand', step=1)
    High_Price=st.number_input("High_Price", min_value=0)
    Total_Kg_Sold=st.number_input('Total_Kg_Sold',min_value=0)


# Make prediction
if st.button("Predict"):
     # Call the prediction function
    prediction_price=predict_price(Province,Size_Grade,Weight_Kg,Sales_Total,Low_Price,High_Price,Total_Kg_Sold,Stock_On_Hand)
    st.success(f'Predicted Average Price of POTATO(WASHED)MONDIAL: R{prediction_price:.2f}')

