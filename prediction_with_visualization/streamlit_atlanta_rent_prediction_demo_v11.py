import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from PIL import Image

# Suppress specific warnings
warnings.filterwarnings("ignore")

# Model and scaler file paths
xgb_filename = 'atlanta_xgb_model.sav'
svr_filename = 'atlanta_svr_model.sav'
ols_filename = 'atlanta_ols_model.sav'
scaler_filename = "atlanta_scaler.pkl"

# Load models and scaler
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))
loaded_xgb_model = pickle.load(open(xgb_filename, 'rb'))
loaded_svr_model = pickle.load(open(svr_filename, 'rb'))
loaded_ols_model = pickle.load(open(ols_filename, 'rb'))

# Load CSV file
df = pd.read_csv('clean_df_demo_atlanta.csv')  # Update with your CSV file name

# Feature details
independent_num_columns = ['bed', 'bath', 'area_sqft', 'walkScore', 'transitScore', 'bikeScore']
categorical_columns = [
    'Wooden Floor Feature',
    'Sundeck Features',
    'Free Wifi',
    '24 Hour Facilities',
    'Accessibility Features',
    'Air Conditioning',
    'Bike/Indoor Bike Storage',
    'Car Services',
    'Ceiling Fan',
    'Closets',
    'Clubhouse',
    'Co-working/Working Spaces/Facilities',
    'Controlled Access',
    'Corner Unit',
    'Courtyard',
    'Den/Den Features',
    'Designer Features',
    'Dishwasher',
    'Energy Efficiency Features',
    'EV Charging',
    'Fire place/pit',
    'Garden',
    'Granite Countertops',
    'Guest Features',
    'Grill Stations/Areas',
    'Gym',
    'Heating Features',
    'High Speed Internet Features',
    'Historic Feature',
    'Icemaker',
    'Keyfob',
    'Kitchen Island',
    'Large Windows',
    'Laundry Facilities',
    'Media and Game Room',
    'Package Service',
    'Parking/Garage/Reserved Parking',
    'Patio/Balcony',
    'Pet Facilities',
    'Picnic Area',
    'Pool Features',
    'Rooftop Features',
    'Security Features',
    'Sewer and Trash',
    'Skylight',
    'Stainless Steel Appliances',
    'Storage Facilities',
    'Storage Additional Spaces',
    'Tile Features',
    'Trash/Garbage Services',
    'Tub/Shower',
    'Views',
    'Valet Services',
    'Vinyl Flooring',
    'Walking/Biking Trails',
    'Washer and Dryer',
    'Wheelchair Access',
    'Yoga Facilities',
    'Zipcode_30306',
    'Zipcode_30307',
    'Zipcode_30308',
    'Zipcode_30309',
    'Zipcode_30310',
    'Zipcode_30312',
    'Zipcode_30313',
    'Zipcode_30314',
    'Zipcode_30315',
    'Zipcode_30316',
    'Zipcode_30318',
    'Zipcode_30324',
    'Zipcode_30332',
    'Zipcode_30363',
    'propType_Apartment',
    'propType_Condo',
    'newBuiltFactor'
]

all_cols = independent_num_columns + categorical_columns

# Streamlit app
st.set_page_config(layout="wide")

# Page selector
page = st.sidebar.selectbox("Select Page", ["Rental Price Prediction", "Visualization"])

if page == "Rental Price Prediction":
    st.title("Atlanta Rental Price Prediction")

    # Add logo
    st.image('images/logo.png', width=200)

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
            'Closets': st.selectbox('Large/Walk-in Closets', [0, 1], index=1),
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
    new_data_df = pd.concat([new_data_num_scaled_df, new_data_df[categorical_columns]], axis=1)

    # Predictions
    xgb_pred = loaded_xgb_model.predict(new_data_df)
    svr_pred = loaded_svr_model.predict(new_data_df)
    ols_pred = loaded_ols_model.predict(new_data_df)

    with col2:
        st.header("Predictions")
        st.write(f"XGBoost Prediction: ${xgb_pred[0]:,.2f}")
        st.write(f"SVR Prediction: ${svr_pred[0]:,.2f}")
        st.write(f"OLS Regression Prediction: ${ols_pred[0]:,.2f}")

elif page == "Visualization":
    st.title("Visualization Dashboard")

    # Interactive filters
    st.sidebar.header("Filter Data")
    selected_beds = st.sidebar.multiselect('Select Number of Beds', options=df['bed'].unique(), default=df['bed'].unique())
    selected_baths = st.sidebar.multiselect('Select Number of Baths', options=df['bath'].unique(), default=df['bath'].unique())
    selected_years = st.sidebar.slider('Select Year Built Range', min_value=int(df['yearBuilt'].min()), max_value=int(df['yearBuilt'].max()), value=(int(df['yearBuilt'].min()), int(df['yearBuilt'].max())))

    # Filter DataFrame based on user selections
    filtered_df = df[(df['bed'].isin(selected_beds)) & (df['bath'].isin(selected_baths)) & (df['yearBuilt'].between(selected_years[0], selected_years[1]))]

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(filtered_df.describe())

    st.subheader("Interactive Chart")

    # Interactive Chart Section (Added Dropdown Menus)
    # Dropdowns for x and y axis selection
    x_axis_col = st.selectbox("Select X-axis Column", options=filtered_df.columns.tolist(), index=filtered_df.columns.get_loc('bed'))
    y_axis_col = st.selectbox("Select Y-axis Column", options=filtered_df.columns.tolist(), index=filtered_df.columns.get_loc('Rent'))

    if x_axis_col and y_axis_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plotting based on selected x and y columns
        sns.scatterplot(data=filtered_df, x=x_axis_col, y=y_axis_col, ax=ax)
        ax.set_title(f'{y_axis_col} vs {x_axis_col}')
        st.pyplot(fig)
    else:
        st.write("Please select valid columns to generate the interactive chart.")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    # Column selection for correlation heatmap
    st.sidebar.header("Select Columns for Correlation Heatmap")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_columns = st.sidebar.multiselect('Select Columns', options=numeric_columns, 
                                              default=[
                                                  'Rent', 
                                                  'bed', 
                                                  'bath', 
                                                  'area_sqft', 
                                                  'yearBuilt', 
                                                  'newBuiltFactor', 
                                                  'walkScore',
                                                  'transitScore',
                                                  'bikeScore',
                                                  '24 Hour Facilities',
                                                  'Accessibility Features',
                                                  'Air Conditioning',
                                                  'Garden',
                                                  'Granite Countertops',
                                                  'Wooden Floor Feature',
                                                  'Skylight',
                                                  'Stainless Steel Appliances',
                                                  'Storage Facilities',
                                                  'Views'
                                                  ])

    if selected_columns:
        # Calculate the correlation matrix
        corr_matrix = df[selected_columns].corr()

        # Create a heatmap with auto-adjusted figure dimensions
        fig, ax = plt.subplots(figsize=(len(selected_columns)*0.75, len(selected_columns)*0.75))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt=".2f", ax=ax,
                    cbar_kws={"shrink": .5})
        plt.title('Correlation Heatmap of Selected Features')

        # Automatically adjust the layout
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Display the heatmap in Streamlit
        st.pyplot(fig)
    else:
        st.write("Please select at least one column to generate the correlation heatmap.")



# streamlit run streamlit_atlanta_rent_prediction_demo_v11.py