import streamlit as st
import pandas as pd
import pickle
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore")

# Model and scaler file paths
xgb_filename = 'atlanta_xgb_model.sav'
svr_filename = 'atlanta_svr_model.sav'
rf_filename = 'atlanta_rf_model.sav' 
ols_filename = 'atlanta_ols_model.sav'
scaler_filename = "atlanta_scaler.pkl"

# Load models and scaler
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))
loaded_xgb_model = pickle.load(open(xgb_filename, 'rb'))
loaded_svr_model = pickle.load(open(svr_filename, 'rb'))
loaded_ols_model = pickle.load(open(ols_filename, 'rb'))

# Feature details
all_cols = [
    'bed', 'bath', 'area_sqft', 'walkScore', 'transitScore', 'bikeScore', 
    'Wooden Floor Feature', 'Sundeck Features', 'Free Wifi', 'Controlled Access', 
    '24 Hour Facilities', 'Accessibility Features', 'Air Conditioning', 
    'Bike/Indoor Bike Storage', 'Ceiling Fan', 'Clubhouse', 'Closets', 
    'Energy Efficiency Features', 'EV Charging', 'Garden', 'Granite Countertops', 
    'Guest Features', 'Gym', 'High Speed Internet Features', 'Historic Feature', 
    'Icemaker', 'Keyfob', 'Kitchen Island', 'Large Windows', 'Laundry Facilities', 
    'Media and Game Room', 'Package Service', 'Parking/Garage/Reserved Parking', 
    'Patio/Balcony', 'Pet Facilities', 'Picnic Area', 'Pool Features', 
    'Rooftop Features', 'Security Features', 'Sewer and Trash', 'Skylight', 
    'Stainless Steel Appliances', 'Storage Facilities', 'Tile Features', 
    'Trash/Garbage Services', 'Tub/Shower', 'Views', 'Vinyl Flooring', 
    'Walking/Biking Trails', 'Washer and Dryer', 'Wheelchair Access', 
    'newBuiltFactor', 'Zipcode_30306', 'Zipcode_30307', 'Zipcode_30308', 
    'Zipcode_30309', 'Zipcode_30310', 'Zipcode_30312', 'Zipcode_30313', 
    'Zipcode_30314', 'Zipcode_30315', 'Zipcode_30316', 'Zipcode_30318', 
    'Zipcode_30324', 'Zipcode_30332', 'Zipcode_30363', 'propType_Apartment', 
    'propType_Condo'
]

independent_num_columns = ['bed', 'bath', 'area_sqft', 'walkScore', 'transitScore', 'bikeScore']
categorical_columns = [
    'Wooden Floor Feature', 'Sundeck Features', 'Free Wifi', 'Controlled Access', 
    '24 Hour Facilities', 'Accessibility Features', 'Air Conditioning', 
    'Bike/Indoor Bike Storage', 'Ceiling Fan', 'Clubhouse', 'Closets', 
    'Energy Efficiency Features', 'EV Charging', 'Garden', 'Granite Countertops', 
    'Guest Features', 'Gym', 'High Speed Internet Features', 'Historic Feature', 
    'Icemaker', 'Keyfob', 'Kitchen Island', 'Large Windows', 'Laundry Facilities', 
    'Media and Game Room', 'Package Service', 'Parking/Garage/Reserved Parking', 
    'Patio/Balcony', 'Pet Facilities', 'Picnic Area', 'Pool Features', 
    'Rooftop Features', 'Security Features', 'Sewer and Trash', 'Skylight', 
    'Stainless Steel Appliances', 'Storage Facilities', 'Tile Features', 
    'Trash/Garbage Services', 'Tub/Shower', 'Views', 'Vinyl Flooring', 
    'Walking/Biking Trails', 'Washer and Dryer', 'Wheelchair Access', 
    'newBuiltFactor', 'Zipcode_30306', 'Zipcode_30307', 'Zipcode_30308', 
    'Zipcode_30309', 'Zipcode_30310', 'Zipcode_30312', 'Zipcode_30313', 
    'Zipcode_30314', 'Zipcode_30315', 'Zipcode_30316', 'Zipcode_30318', 
    'Zipcode_30324', 'Zipcode_30332', 'Zipcode_30363', 'propType_Apartment', 
    'propType_Condo'
]

# Feature details
all_cols = [
    'bed', 'bath', 'area_sqft', 'walkScore', 'transitScore', 'bikeScore', 
    'Wooden Floor Feature', 'Sundeck Features', 'Free Wifi', 'Controlled Access', 
    '24 Hour Facilities', 'Accessibility Features', 'Air Conditioning', 
    'Bike/Indoor Bike Storage', 'Ceiling Fan', 'Clubhouse', 'Closets', 
    'Energy Efficiency Features', 'EV Charging', 'Garden', 'Granite Countertops', 
    'Guest Features', 'Gym', 'High Speed Internet Features', 'Historic Feature', 
    'Icemaker', 'Keyfob', 'Kitchen Island', 'Large Windows', 'Laundry Facilities', 
    'Media and Game Room', 'Package Service', 'Parking/Garage/Reserved Parking', 
    'Patio/Balcony', 'Pet Facilities', 'Picnic Area', 'Pool Features', 
    'Rooftop Features', 'Security Features', 'Sewer and Trash', 'Skylight', 
    'Stainless Steel Appliances', 'Storage Facilities', 'Tile Features', 
    'Trash/Garbage Services', 'Tub/Shower', 'Views', 'Vinyl Flooring', 
    'Walking/Biking Trails', 'Washer and Dryer', 'Wheelchair Access', 
    'newBuiltFactor', 'Zipcode_30306', 'Zipcode_30307', 'Zipcode_30308', 
    'Zipcode_30309', 'Zipcode_30310', 'Zipcode_30312', 'Zipcode_30313', 
    'Zipcode_30314', 'Zipcode_30315', 'Zipcode_30316', 'Zipcode_30318', 
    'Zipcode_30324', 'Zipcode_30332', 'Zipcode_30363', 'propType_Apartment', 
    'propType_Condo'
]

independent_num_columns = ['bed', 'bath', 'area_sqft', 'walkScore', 'transitScore', 'bikeScore']
categorical_columns = [
    'Wooden Floor Feature', 'Sundeck Features', 'Free Wifi', 'Controlled Access', 
    '24 Hour Facilities', 'Accessibility Features', 'Air Conditioning', 
    'Bike/Indoor Bike Storage', 'Ceiling Fan', 'Clubhouse', 'Closets', 
    'Energy Efficiency Features', 'EV Charging', 'Garden', 'Granite Countertops', 
    'Guest Features', 'Gym', 'High Speed Internet Features', 'Historic Feature', 
    'Icemaker', 'Keyfob', 'Kitchen Island', 'Large Windows', 'Laundry Facilities', 
    'Media and Game Room', 'Package Service', 'Parking/Garage/Reserved Parking', 
    'Patio/Balcony', 'Pet Facilities', 'Picnic Area', 'Pool Features', 
    'Rooftop Features', 'Security Features', 'Sewer and Trash', 'Skylight', 
    'Stainless Steel Appliances', 'Storage Facilities', 'Tile Features', 
    'Trash/Garbage Services', 'Tub/Shower', 'Views', 'Vinyl Flooring', 
    'Walking/Biking Trails', 'Washer and Dryer', 'Wheelchair Access', 
    'newBuiltFactor', 'Zipcode_30306', 'Zipcode_30307', 'Zipcode_30308', 
    'Zipcode_30309', 'Zipcode_30310', 'Zipcode_30312', 'Zipcode_30313', 
    'Zipcode_30314', 'Zipcode_30315', 'Zipcode_30316', 'Zipcode_30318', 
    'Zipcode_30324', 'Zipcode_30332', 'Zipcode_30363', 'propType_Apartment', 
    'propType_Condo'
]

# Streamlit app
st.set_page_config(layout="wide")
st.title("Atlanta Rental Price Prediction")

# Layout with columns
col1, col2 = st.columns([3, 1])  # Adjust width ratio as needed

# Input features in the left column
with col1:
    st.header("Input Features")
    selected_zipcode = st.selectbox('Zipcode', [
        'Zipcode_30306', 'Zipcode_30307', 'Zipcode_30308', 'Zipcode_30309', 
        'Zipcode_30310', 'Zipcode_30312', 'Zipcode_30313', 'Zipcode_30314', 
        'Zipcode_30315', 'Zipcode_30316', 'Zipcode_30318', 'Zipcode_30324', 
        'Zipcode_30332', 'Zipcode_30363'
    ])

    bed_options = {"Studio": 0.5, "1 Bed": 1, "2 Beds": 2, "3 Beds": 3, "4 Beds": 4, "5 Beds": 5}
    selected_bed_label = st.selectbox('Number of Beds', list(bed_options.keys()))
    selected_bed_value = bed_options[selected_bed_label]

    new_data_point = {
        'bed': selected_bed_value,
        'bath': st.selectbox('Number of Baths', [1, 2, 3, 4, 5], index=0),
        'area_sqft': st.number_input('Area (sqft)', value=855),
        'walkScore': st.number_input('Walk Score', value=80),
        'transitScore': st.number_input('Transit Score', value=50),
        'bikeScore': st.number_input('Bike Score', value=80),
        'Wooden Floor Feature': st.selectbox('Wooden Floor Feature', [0, 1], index=1),
        'Granite Countertops': st.selectbox('Granite Countertops', [0, 1]),
        'Sundeck Features': st.selectbox('Sundeck Features', [0, 1]),
        'Vinyl Flooring': st.selectbox('Vinyl Flooring', [0, 1]),
        'Stainless Steel Appliances': st.selectbox('Stainless Steel Appliances', [0, 1], index=1),
        'Large/Walk-in Closets': st.selectbox('Large/Walk-in Closets', [0, 1], index=1),
        'Laundry Facilities': st.selectbox('Laundry Facilities', [0, 1]),
        'Washer and Dryer': st.selectbox('Washer and Dryer', [0, 1], index=1),
        'Gym': st.selectbox('Gym', [0, 1], index=1),
        'Pool': st.selectbox('Pool', [0, 1]),
        'Views': st.selectbox('Views', [0, 1]),
        'newBuiltFactor': st.selectbox('New Built Factor', [0, 1]),
        'propType_Condo': st.selectbox('Property Type Condo', [0, 1]),
        'propType_Apartment': st.selectbox('Property Type Apartment', [0, 1], index=1),
        'Energy Efficiency Features': st.selectbox('Energy Efficiency Features', [0, 1]),
        'EV Charging': st.selectbox('EV Charging', [0, 1]),
        '24 Hour Facilities': st.selectbox('24 Hour Facilities', [0, 1], index=1),
        'Clubhouse': st.selectbox('Clubhouse', [0, 1]),
        selected_zipcode: 1
    }

# Convert new data point to DataFrame
new_data_df = pd.DataFrame([new_data_point])

# Add missing columns with value 0
for col in all_cols:
    if col not in new_data_df.columns:
        new_data_df[col] = 0

# Ensure the new DataFrame columns are in the same order as the original DataFrame columns
new_data_df = new_data_df[all_cols]

# Scale numerical features
new_data_num_scaled = loaded_scaler.transform(new_data_df[independent_num_columns])
new_data_num_scaled_df = pd.DataFrame(new_data_num_scaled, columns=independent_num_columns)

# Concatenate scaled numerical features and categorical features
new_data_final = pd.concat([new_data_num_scaled_df, new_data_df[categorical_columns]], axis=1)

# Make predictions
xgb_prediction = loaded_xgb_model.predict(new_data_final)[0]
svr_prediction = loaded_svr_model.predict(new_data_final)[0]
ols_prediction = loaded_ols_model.predict(new_data_final)[0]

# Display predictions in the right column
with col2:
    st.header("Predicted Rental Prices")
    st.write(f"**XGBoost Prediction:** ${xgb_prediction:.2f}")
    st.write(f"**SVR Prediction:** ${svr_prediction:.2f}")
    st.write(f"**OLS Regression Prediction:** ${ols_prediction:.2f}")

