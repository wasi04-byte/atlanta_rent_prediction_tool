# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:54:16 2024

@author: Asus
"""
# conda create --name rent_price_ML
import pickle
xgb_filename = 'atlanta_xgb_model.sav'
svr_filename = 'atlanta_svr_model.sav'
rf_filename = 'atlanta_rf_model.sav' 
ols_filename = 'atlanta_ols_model.sav'
scaler_filename = "atlanta_scaler.pkl"



import json
import pandas as pd
import re
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None) 


import os

# Get the absolute path of the current script
current_file_path = os.path.abspath(__file__)

# Get the directory name from the absolute path
current_directory = os.path.dirname(current_file_path)


##### **********************************************************************

def merge_json_files(file_paths):
    dataframes = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding="utf8") as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        dataframes.append(df)
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df

# List of file paths
file_paths = [
    'unitDictionaryAptDotComAtlanta_20240107_v1.json',
    'unitDictionaryAptDotComAtlanta_20240715_v1.json',
    'unitDictionaryAptDotComAtlanta_20240715_v2.json',
    'unitDictionaryAptDotComAtlanta_20240715_v3.json',
    'unitDictionaryAptDotComAtlanta_20240715_v4.json',
]

# Get the merged DataFrame
merged_df = merge_json_files(file_paths)

##### **********************************************************************



### ********** Removing observations with no info on singleRentCheck **************\
    
#expected_row_deletion = (merged_df ['checkSingleRent'] == 'Could not find').sum()


# Filter out the rows where 'checkSingleRent' is 'Could not find'
filtered_df_v1 = merged_df [merged_df ['checkSingleRent'] != 'Could not find']

# Replace last layer Could Not Find NaN
filtered_df_v1['extractedFloorDetails'] = filtered_df_v1['extractedFloorDetails'].fillna('Could not find')


### ***************************** Checking Values *******************************
column_name = 'extractedAddress' #'extractedFloorDetails #'extracted_string' #extractedAddress
column_names = ['checkSingleRent', 'bedbathDetailsWithTypes', 'extractedFloorDetails'] #'extractedFloorDetails', 'extracted_string', 'rentRange', "actualRentValues" #'extractedFloorDetails', 'extracted_string'
row_index = 1 #820  #42 #33 #162,181

# Get the column index by column name
column_indices = [filtered_df_v1.columns.get_loc(name) for name in column_names]
column_index = filtered_df_v1.columns.get_loc(column_name)


# Access the specific value
test_values = filtered_df_v1.iloc[row_index, column_indices]
print("\nNew test: ",test_values)



## *************************************************************************************

clean_df_v1 = filtered_df_v1.copy()

# Function to extract bed information
# Function to extract bed information
def extract_beds(details, alternate_details=None):
    """
    Extracts the number of beds from the given details.

    Args:
        details (str): The original details string.
        alternate_details (str, optional): An alternate details string to use if 'details' is NaN or None.

    Returns:
        int or str or None: The extracted bed information.
    """
    if pd.isna(details) or details is None:
        # Use the value from 'alternate_details' if 'details' is NaN or None
        if alternate_details:
            # Look for "Bedrooms" or "Bedroom" and grab the next line
            bed_alternate_match = re.search(r'(\d+ beds|Studio|\d+ bed|\d+ bd)', alternate_details)
            if bed_alternate_match:
                beds = bed_alternate_match.group(1)
                if beds == "Studio" or beds == "studio":
                    return float(0.5)
                return float(beds.split()[0])
        return None

    bed_match = re.search(r'(\d+ beds|Studio|1 bed)', details)
    if bed_match:
        beds = bed_match.group(1)
        if beds == "Studio" or beds == "studio":
            return float(0.5)
        return float(beds.split()[0])
    return None

# Example usage:
clean_df_v1['bed'] = clean_df_v1.apply(lambda row: extract_beds(row['bedbathDetailsWithTypes'], row['checkSingleRent']), axis=1)


# Function to extract bath information
def extract_baths(details, alt_details):
    # Helper function to find bath information in a given text
    def find_baths(text):
        for line in text.splitlines():
            bath_match = re.search(r'(\d+(\.\d+)? baths?|(\d+(\.\d+)? bas?))', line, re.IGNORECASE)
            if bath_match:
                return float(bath_match.group(1).split()[0])
        return None

    # Try to extract from the main details first
    if pd.notna(details):
        baths = find_baths(details)
        if baths is not None:
            return baths
    
    # Try to extract from the alternative details if main details are not available
    if pd.notna(alt_details):
        return find_baths(alt_details)
    
    return None

# Apply the functions to create new columns
#clean_df_v1['bed'] = clean_df_v1.apply(lambda row: extract_beds(row['bedbathDetailsWithTypes']), axis=1)
clean_df_v1['bath'] = clean_df_v1.apply(lambda row: extract_baths(row['bedbathDetailsWithTypes'], row['checkSingleRent']), axis=1)

nan_bed_check = clean_df_v1['bed'].isna().sum()



## *************************************************************************************

### ***************************** Checking Values *******************************
column_name = 'extractedAddress' #'extractedFloorDetails #'extracted_string' #extractedAddress
column_names = ['checkSingleRent', 'bedbathDetailsWithTypes', 'propType', 'bed', 'bath'] #'extractedFloorDetails', 'extracted_string', 'rentRange', "actualRentValues" #'extractedFloorDetails', 'extracted_string'
row_index = 813 #820  #42 #33 #162,181

# Get the column index by column name
column_indices = [clean_df_v1.columns.get_loc(name) for name in column_names]
column_index = clean_df_v1.columns.get_loc(column_name)


# Access the specific value
test_values = clean_df_v1.iloc[row_index, column_indices]
print("\nNew test: ",test_values)

# ******************************************************************


# Locate the index of the specific value '20' in the 'bed' column

bed_counts = clean_df_v1['bed'].value_counts()
bath_counts = clean_df_v1['bath'].value_counts()
    
specific_value = "20"
indices = clean_df_v1[clean_df_v1['bed'] == specific_value].index.tolist()

# Display the indices
print(f"Indices of bed value {specific_value}: {indices}")

### ******************* Cleaning Rent before Extraction #########################

def filter_rows_by_regex(df, column_name, regex_pattern):
    """
    Filters rows in a DataFrame based on a regex pattern in a specific column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): Name of the column to check.
        regex_pattern (str): Regular expression pattern to match.
    
    Returns:
        pd.DataFrame: Filtered DataFrame without rows containing the specified pattern.
    """
    filtered_df = df[~df[column_name].str.contains(regex_pattern, na=False)]
    return filtered_df

# Example usage:
# Assuming 'clean_df_v2' is your DataFrame
clean_df_v2_0 = filter_rows_by_regex(clean_df_v1, 'bedbathDetailsWithTypes', r"/ Person")



# Function to extract rent from details, alt_details, or checkSingleRent columns
def extract_rent(details, alt_details, check_single_rent):
    # Helper function to extract rent from a given text
    def extract_from_text(text):
        if pd.notna(text):
            for line in text.splitlines():
                if "$" in line:
                    rent_matches = re.findall(r'\$([\d,]+)', line)
                    if rent_matches:
                        rents = [float(rent.replace(',', '')) for rent in rent_matches]
                        return ','.join(map(str, rents))
        return None
    
    # Try to extract rent from the main details first
    rent = extract_from_text(details)
    if rent is not None:
        return rent
    
    # If not found, try to extract from the alternative details
    rent = extract_from_text(alt_details)
    if rent is not None:
        return rent
    
    # If not found, try to extract from the checkSingleRent column
    return extract_from_text(check_single_rent)

# Function to extract area from details, alt_details, or checkSingleRent columns
def extract_area(details, alt_details, check_single_rent):
    # Helper function to extract area from a given text
    def extract_from_text(text):
        if pd.notna(text):
            for line in text.splitlines():
                if "square feet" in line.lower() or "sq ft" in line.lower():
                    area_match = re.search(r'(\d{1,3}(?:,\d{3})*)(?=\s*sq ft)', line, re.IGNORECASE)
                    if area_match:
                        area_str = area_match.group(1)
                        area_float = float(area_str.replace(',', ''))
                        return area_float
        return None
    
    # Try to extract area from the main details first
    area = extract_from_text(details)
    if area is not None:
        return area
    
    # If not found, try to extract from the alternative details
    area = extract_from_text(alt_details)
    if area is not None:
        return area
    
    # If not found, try to extract from the checkSingleRent column
    return extract_from_text(check_single_rent)
    

### **************************** New DF Copied ****************

clean_df_v2 = clean_df_v2_0.copy()

# Apply the functions to create new columns
clean_df_v2['rent'] = clean_df_v2.apply(lambda row: extract_rent(row['extractedFloorDetails'], row['bedbathDetailsWithTypes'], row['checkSingleRent']), axis=1)
clean_df_v2['area_sqft'] = clean_df_v2.apply(lambda row: extract_area(row['extractedFloorDetails'], row['bedbathDetailsWithTypes'], row['checkSingleRent']), axis=1)


### ***************************************

### ***************************** Checking Values *******************************
column_name = 'extractedAddress' #'extractedFloorDetails #'extracted_string' #extractedAddress
column_names = ['bed', 'propType', 'rent', 'area_sqft'] #'extractedFloorDetails', 'extracted_string', 'rentRange', "actualRentValues" #'extractedFloorDetails', 'extracted_string'
row_index = 815 #810 #120 #691 #67 #820  #42 #33 #162,181

# Get the column index by column name
column_indices = [clean_df_v2.columns.get_loc(name) for name in column_names]
column_index = clean_df_v2.columns.get_loc(column_name)


# Access the specific value
test_values = clean_df_v2.iloc[row_index, column_indices]
print("\nNew test: ",test_values)

# ******************************************************************


### **************** DELETING unnecessary entries **************************

def delete_rows_with_phrases(df, column_phrase_dict):
    """
    Delete rows from the DataFrame if any of the specified phrases are found in the corresponding columns.

    :param df: The DataFrame to be filtered.
    :param column_phrase_dict: A dictionary where the keys are column names and the values are lists of phrases to search for.
    :return: The filtered DataFrame with the specified rows deleted.
    """
    mask = pd.Series([False] * len(df))
    
    for column, phrases in column_phrase_dict.items():
        if column in df.columns:
            for phrase in phrases:
                mask = mask | df[column].astype(str).str.contains(phrase, case=False, na=False)
    
    filtered_df = df[~mask]
    return filtered_df

column_phrase_dict = {
    'bedbathDetailsWithTypes': ["Call for Rent", "Not Available", "Show Unavailable Floor"],
    'bed': ["None"]
}

# Apply the function to delete rows with specified phrases
clean_df_v3 = delete_rows_with_phrases(clean_df_v2, column_phrase_dict)

columns_list = clean_df_v3.columns.tolist()

### ************************ Building Name with my experimental regex *************************

building_name_regex = re.compile(r'^.*(?= media gallery)', re.MULTILINE)

# building_name_matches = building_name_regex.findall(clean_df_v3['extractedBuildingName']) ## will lead to: TypeError: expected string or bytes-like object, got 'Series'

clean_df_v3['propName'] = clean_df_v3['extractedBuildingName'].apply(lambda x: building_name_regex.findall(x)[0])

#print(clean_df_v3['propName'].iloc[500:])
### ***************************************************************************



### ********************** Simple standalone function to get rid of rows with unwanted strings ##############################
# Identify records with "Call for Rent"
'''
contains_call_for_rent = clean_df_v3['bedbathDetailsWithTypes'].str.contains("Call for Rent", case=False, na=False)

# Delete those rows
clean_df_v4 = clean_df_v3[~contains_call_for_rent]

# Display the clean DataFrame
print("\n**** Shape now: \n", clean_df_v4.shape)
'''
### *****************************************************************************************




nan_bed_check_v3 = clean_df_v3['bed'].isna().sum()
nan_rent_check_v3 = clean_df_v3['rent'].isna().sum()

### **************************************************************************

def average_comma_separated_values(input_string):
    # Split the input string based on commas
    values = input_string.split(',')

    # Convert the split values to floats and filter out non-numeric values
    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(value.strip()))
        except ValueError:
            continue

    # Calculate the average of the numeric values
    if numeric_values:
        average_value = sum(numeric_values) / len(numeric_values)
        return average_value
    else:
        return None  # or 0 if you prefer to return 0 when no numeric values are found
    
clean_df_v3['cleanRent'] = clean_df_v3['rent'].apply(average_comma_separated_values)

### ***************************************************************************


### ******************* Checking Random Statistics ***************************
print(clean_df_v3['area_sqft'].max())
print(clean_df_v3['area_sqft'].min())

print(clean_df_v3['bed'].value_counts())

#clean_df_v3['rent'].isna().sum()
# Determine the unique data types in the 'rent' column
unique_types = set(clean_df_v3['cleanRent'].apply(type))
print("\n***** Unique rent tpyes are: \n", unique_types)
#df.dtypes.unique()

sample_median_rent_by_bedtype = clean_df_v3[clean_df_v3['bed']==2]['cleanRent'].median()

# ***************************************************************************


### ***************** Cleaning up amenities *************************
clean_df_v3['allAmenities'] = clean_df_v3['extractedAmenities'].str.split("\n").apply(lambda x: ", ".join(x)).str.split(', ')
clean_df_v3['allUniqueFeatures'] = clean_df_v3['extractedUniqueFeatures'].str.split("\n").apply(lambda x: ", ".join(x)).str.split(', ')

# Merge both lists and convert to a set
clean_df_v3['listedAmenities'] = clean_df_v3.apply(lambda row: list(set(row['allAmenities'] + row['allUniqueFeatures'])), axis=1)

def get_amenities_set(df):
    # Initialize a set to keep track of all unique amenities
    all_amenities = set()

    # Populate the set with amenities from each row
    for amenities in df['listedAmenities']:
        all_amenities.update(amenities)
        
   

    return all_amenities


amenities_set = get_amenities_set(clean_df_v3)

# Define the strings you want to remove
amenities_to_remove = {'', 
                       'Apartment Features', 
                       'Community Amenities', 
                       'Condo Features', 
                       'Could not find', 
                       'Floor Plan Details', 
                       'Kitchen Features & Appliances', 
                       'Mother-in-law Unit', 
                       'Townhome Features', 
                       '**May Vary Between Apartments',
                       '1 & 2 Bedrooms Apartment Homes',
                       '1 - Reserved Parking Space',
                       '1 Gated Reserved Parking Space',
                       '1 Mile To Southwest Tennessee Comm. Coll',
                       '1 Reserved Parking Space',
                       '1 mile from I40 and 2 blocks from trolley',
                       '1/2 mile from Beale Street'
                       '2nd Floor Rooftop Terrace',
                       '3 Acres Of Green Space',
                       'Two and Three bedroom apartment homes',
                       'Two playgrounds',
                       'UTILITIES INCLUDED',
                       'Views Dishwasher',
                       'Views Washer/Dryer',
                       'Walk Score of 92',
                       'Walk To Campus',
                       'Walk To Campus Community-Wide WiFi',
                       'Walk To Campus Controlled Access',
                       'Walk To Campus Highlights'
                       'Walking Distance to Beale Street',
                       'Walking distance of Medical Center/UTCHS/Bio-Med z Laundry Facilities',
                       'and Lounge',
                       '• Maplewood Laminate And Concrete Floors'
                       '• Picnic Station And Gas Grills. Wi-Fi',
                       '• Urban Courtyard With Lounge Area.',
                       'Enjoy the monthly Art Trolley Tour',
                       'Individual Leases Available',
                       'Individual Leases Available Furnished Units Available',
                       'Individual Leases Available Washer/Dryer',
                       'Highlights',
                       'Hop over to the South Main Arts District',
                       'Master bath has stand up shower only',
                       'Mata bus at the corner',
                       'Minutes Away From Retail',
                       'Near Public Transit',
                       'Near Univ. Of Tn Health & Science Center',
                       'New 2 & 3 Bedroom Duplexes',
                       'Play Fetch in our Dog Yard',
                       'Grill Highlights',
                       'Gated Highlights',
                       }

'''
# Using discard() method
for item in amenities_to_remove:
    amenities_set.discard(item)
'''

#clean_df_v4 = get_amenities_dummies(clean_df_v3.copy(), 'listedAmenities', amenities_set, amenity_mapping)


### ************ New Structure *******************************

# Mapping of similar amenities to specified terms

sorted_amenities_list = sorted(list(amenities_set))

amenity_mapping = {
    
    'Wooden Floor Feature': [
                       'Wood Style Flooring Available',
                       'Wood-Finish Plank Flooring', 
                       'Wood-Style Flooring', 
                       'Wood-style Plank Flooring'
                       ],
    
    'Sundeck Features': ['Sundeck', 'Sundeck Car Wash Area', 'Sundeck High Speed Internet Access', 'Sundeck Highlights',
                         'Sundeck Package Service', 'Sundeck Washer/Dryer'],
    
    'Free Wifi': ['• Free Wi-fi Access.', 'Free Wi-Fi' 'Wi-Fi', 'Free Wifi', 'WIFI accessible building', 'Multi Use Room Wi-Fi'],
    
    'Controlled Access': ['Controlled Access',
    'Controlled Access Building',
    'Controlled Access Building & Parking Areas',
    'Controlled Access Building Entry',
    'Controlled Access Buildings Community-Wide WiFi',
    'Controlled Access Entrances',
    'Controlled Access Laundry Facilities',
    'Controlled Access Parking',
    'Controlled Access Washer/Dryer Washer/Dryer',
    'Controlled Access to High-rise Apartments',
    'Gated Controlled Access',
    'Controlled-access gated community'],
    
    
    '24 Hour Facilities' : [
        '24 Hour Access',
        '24 Hour Emergency Maintenance',
        '24 Hour Security',
        '24 Hour South Bluffs Health Club',
        '24 hour emergency maintenance service',
        '24-Hour Emergency Maintenance',
        '24-Hour Fitness Center',
        '24-Hour On-Site Maintenance'
        ],
    
    'Accessibility Features': [
        'ADA',
        'ADA Accessible*',
        'Accent Walls Available*',
        'Accessible'
        ],
    
    'Air Conditioning': [
        'Air Conditioner',
        'Air Conditioning'
        ],
    
    'Bike/Indoor Bike Storage' :[
        'Indoor Bike Storage',
        'Gated Bike Storage',
        'Bicycle Storage',
        'Bike Storage',
        'Bike/Indoor Bike Storage'
        ],
    
    'Ceiling Fan': [
        'Ceiling Fan',
        'Ceiling Fan in Bedroom',
        'Ceiling Fans',
        'Ceiling Fans Throughout',
        'Ceiling Fans in all Bedrooms',
        'Ceiling Fans* (Select Units Only)',
        '8 to 9 foot or vaulted ceilings',
        '9 Foot Ceilings',
        "9' to 12' ceilings",
        '10 Ft Ceilings',
        "10' Ceilings *",
        'Vaulted Ceiling',
        'Vaulted Ceilings*',
        'Vaulted Ceilings* (Select Units Only)',
        'High Ceilings',
        'High Ceilings** Controlled Access',
        'High Ceilings*',
        'High Ceilings**',
        ],
    
    'Closets' : [
        'Walk-In Closets',
        'Walk-In Closets High Speed Internet Access',
        'Walk-In Closets Washer/Dryer',
        'Large Walk In Closet s',
        'Large Walk-In Custom Closets',
        'Large Closets',
        'Modular Closet System *',
        'Oversized Closets',
        'Spacious Closets',
        'Generous Closet',
        'Generous Closet Space',
        'Generous Closet Spaces',
        'Generous Closets',
        ],
    
    'Clubhouse': [
        'Club Room with Outdoor viewing terrace',
        'Clubhouse',
        'Clubhouse with a Coffee Bar',
        'Clubroom',
        'Clubroom with WiFi & TV',
        ],
    
    
    'Energy Efficiency Features': [
        'Energy Efficient light fixtures',
        'Energy Star appliances',
        'Efficiency',
        'Efficient Appliances'
        ],
    
    'EV Charging': [
        'EV Charging',
        'EV Charging Shared Community'
        ],
    
    'Garden' : [
        'Garden',
        'Garden  ',
        ],
    
    'Granite Countertops' : [
        'Granite Counter Tops Throughout',
        'Granite Counters',
        'Granite Counters in Kitchen and Bath',
        'Granite Countertops',
        'Granite Countertops & Smoothtop Stove',
        'Granite Countertops with Tile Backsplash',
        'Granite Countertops*',
        'granite or quartz countertops',
        ],
    
    'Guest Features': [
        'Guest Apartment',
        'Guest Suites Available',
        'Guest bathroom has tub and shower combo',
        'Lounge Fitness & Recreation'
        ],
    
    'Gym' : [
        'Fitness & Recreation',
        'Fitness Center',
        'Fitness Center Controlled Access',
        'Fitness Center Furnished Units Available',
        'Fitness Center Stainless Steel Appliances',
        'Fitness Center Washer/Dryer',
        'Modern Fitness Center',
        'Vintage Building Fitness & Recreation',
        ],
    
    'Heating Features' : [
        'Heated Swimming Pool',
        'Heating',
        ],
    
    'High Speed Internet Features': [
        'High Speed Internet',
        'High Speed Internet Access',
        'High Volume Ceilings',
        'High-Speed Internet and Cable Ready'
        ],
    
    'Historic Feature': [
        'Historic Building',
        'Historic Madison Line Trolly Route Stop',
        'Historical Building'
        ],
    
    'Icemaker': [
        'Ice Maker',
        'Ice Maker in Refrigerator'
        ],
    
    'Keyfob' : [
        'Key Fob Entry',
        'Key Fob Entry Shared Community'
        ],
    
    'Kitchen Island' : [
        'Kitchen',
        'Kitchen Islands',
        'Kitchen Islands* Controlled Access'
        ],
    
    'Large Windows': [
        'Large Windows',
        'Large Windows with Skylights',
        'Large Windows*',
        'Over-Sized Windows**'
        ],
    
    'Laundry Facilities' : [
        'Laundry Facilities',
        'Laundry Facility',
        'Laundry Facility On Site',
        'Laundry Service',
        'Laundry and dry cleaning pickup',
        'On-site laundry facility',
        'Mini- Blinds Laundry Facilities',
        'Gated Laundry Facilities',
        ],
    
    'Media and Game Room' : [
        'Media Center/Movie Theatre',
        'Media Center/Movie Theatre Outdoor Features',
        'Media Room',
        'Game Room',
        'Gameroom',
        'Gameroom Air Conditioning',
        'Gameroom Outdoor Features',
        'Gameroom Package Service',
        ],
    
    'Package Service' : [
        'One Gated Reserved Parking Space',
        'Package Acceptance',
        'Package Concierge',
        'Package Locker System',
        'Package Room',
        'Package Service',
        'Package delivery at office in your absence',
        'Patio/Balcony Package Service',
        ],
    
    'Parking/Garage/Reserved Parking': [
        'Off Street Parking',
        'Reserved Assigned Parking',
        'Reserved Parking',
        'Reserved Parking Space',
        'Resident & Guest Parking',
        'Parking Available',
        'Multi-Level Parking Garage',
        'Gated Parking',
        'Gated Reserved Parking',
        'Gated Reserved Parking Included',
        'Garage',
        ],
    
    'Patio/Balcony': [
        'Patio',
        'Patio  ',
        'Patio High Speed Internet Access',
        'Patio/Balcony',
        'Largest balcony/patio',
        ],
    
    'Pet Facilities': [
        'Pet Area',
        'Pet Care',
        'Pet Care Shared Community',
        'Pet Friendly',
        'Pet Park',
        'Pet Play Area',
        'Pet Spa',
        'Pet Washing Station',
        'Pet walk area'
        ],
    
    'Picnic Area': [
        'Picnic Area',
        'Picnic Area Car Wash Area',
        'Picnic Area Highlights',
        'Picnic Area Laundry Facilities',
        'Picnic Area Package Service',
        'Picnic Area Washer/Dryer',
        'Picnic Area Wi-Fi',
        ],
    
    'Pool Features' : [
        'Pool',
        'Pool Controlled Access',
        'Pool Dishwasher',
        'Pool Fitness Center',
        'Pool Overlook',
        'Pool Pool',
        'Pool Views* (Select Units Only)',
        'Poolside lounge',
        'Swimming Pool',
        'Stunning Priviate Patios On Select Units',
        'Resort-Style Pool & Outdoor Lounge',
        'Resort-style swimming pool',
        'Resort-style swimming pool Laundry Facilities',
        'Five Resort Style Swimming Pools',
        'Heated Swimming Pool',
        'Gorgeous swimming pool',
        ],
    
    'Rooftop Features': [
        'Roof Terrace',
        'Roof Terrace Controlled Access',
        'Roof Terrace High Speed Internet Access',
        'Rooftop Clubroom',
        'Rooftop Terrace with Spectacular Views',
        'Rooftop Viewing Area',
        'Rooftop deck'
        ],
    
    'Security Features': [
        'Security',
        'Security Alarm',
        'Security System'
        ],
    
    'Sewer and Trash': [
        'Sewage And Trash Included! Controlled Access',
        'Sewer & Trash Included',
        'Sewer and Trash'
        ],
    
    'Skylight': [
        'Skylight',
        'Skylights',
        'Skylights*'
        ],
    
    'Stainless Steel Appliances': [
        'Stainless Steel Appliance Package',
        'Stainless Steel Appliances',
        'Stainless Steel Appliances Dishwasher',
        'Stainless Steel Appliances!!',
        'Stainless Steel Appliances*',
        'Gourmet Kitchens Featuring Stainless Appliances',
        ],
    
    'Storage Facilities' :[
        'On-Site Storage',
        'Storage Rooms',
        'Storage Space',
        'Storage Space Fitness & Recreation',
        'Storage Units',
        ],
    
    'Tile Features' : [
        'Tile Floors',
        'Tile Floors Washer/Dryer',
        'Tiled Backsplash & Bath Surrounds',
        'Tiled Entry Foyers'
        ],
    
    'Trash/Garbage Services': [
        'Trash Compactor',
        'Trash Pickup - Curbside',
        'Trash Pickup - Door to Door'
        'Garbage Disposal',
        ],
    
    'Tub/Shower' : [
        'Tub/Shower',
        'Tub/Shower Kitchen Features & Appliances',
        'Tub/shower Combo In Both Bathrooms!',
        'Large garden-style oval tubs*',
        ],
    
    'Views' : [
        'Beautiful river views',
        'Beautiful river views*',
        'River Overlook',
        'River View',
        'River Views* (Select Units Only)',
        'River and City Views Available',
        'River and Downtown Views Package Service',
        'View - Full City',
        'View - PH City',
        'View - PH River',
        'View - Partial City',
        'Views',
        'Spectacular River',
        'Panoramic River Views of Downtown Memphis',
        ],
    
    'Vinyl Flooring' : [
        'Vinyl Flooring',
        'Vinyl Flooring High Speed Internet Access',
        'Vinyl Plank Flooring Throughout',
        ],
    

    'Walking/Biking Trails' : [
        'Walking/Biking Trails',
        'Walking/Biking Trails Outdoor Features'
        ],
    
    'Washer and Dryer' : [
        'Washer & Dryer Included in Each Home Property Services',
        'Washer and Dryer Included Property Services',
        'Washer/Dryer',
        'Washer/Dryer Connections',
        'Washer/Dryer Hookup',
        'Washer/Dryer In Unit',
        'Full Size Washer/Dryer Connections in most units',
        'Full Sized Washer & Dryer',
        'Full size Washer + Dryer',
        'Full-Size Washer/Dryer In-Unit',
        'Full-size Washer/Dryer Included',
        'Full-sized washer/dryers included',
        ],
    
    'Wheelchair Access' : [
        'Wheelchair Access',
        'Wheelchair Accessible (Rooms)',
        'Wheelchair Accessible (Rooms) Air Conditioning',
        'Wheelchair Accessible (Rooms) Kitchen Features & Appliances',
        ]
    
       
}


# Create new columns in clean_df_v5 with initial values set to 0
for key in amenity_mapping:
    clean_df_v3[key] = 0

# Update the relevant columns based on listedAmenities
for index, row in clean_df_v3.iterrows():
    for key, values in amenity_mapping.items():
        if any(value in row['listedAmenities'] for value in values):
            clean_df_v3.loc[index, key] = 1


#print(amenities_df.head())

amenities_df = clean_df_v3[list(amenity_mapping.keys())]

clean_df_v6 = clean_df_v3.copy()

clean_df_v6.insert(3, 'propName', clean_df_v6.pop('propName'))

selected_columns = [
    'url',
     'propType',
     'extractedBuildingName',
     'propName',
     'extractedAddress',
     'extractedAmenities',
     'extractedUniqueFeatures',
     'checkSingleRent',
     'extractedBuildingInfo',
     'extractedScoreCards',
     'extractedReview',
     'bedbathDetailsWithTypes',
     'extractedFloorDetails',
     'bed',
     'bath',
     'rent',
     'area_sqft',
     'cleanRent'
    ]

expanded_df_v0 = clean_df_v6[selected_columns]

expanded_df_v1 = pd.concat([expanded_df_v0, amenities_df], axis=1)



### ************************ Zipcode/Location with my experimental regex *************************

#zipcode_location_regex = re.compile(r'^TN\s(\d{5})', re.MULTILINE)
#zipcode_location_regex = re.compile(r'^[A-Z]{2}\s(\d{5})', re.MULTILINE)

zipcode_location_regex = re.compile(r'\b[A-Z]{2}\s(\d{5})\b')

expanded_df_v1['zipcode'] = expanded_df_v1['extractedBuildingName'].apply(lambda x: zipcode_location_regex.findall(x)[0])

print(expanded_df_v1['zipcode'].iloc[:20])
### ***************************************************************************

### ************************ Year Built with my experimental regex *************************

# Regex pattern
year_built_regex = re.compile(r'Built\sin\s(\d{4})')

# Function to extract year
def extract_year_built(text):
    if text:
        match = year_built_regex.search(text)
        if match:
            return int(match.group(1))
    return None # Conditional check done here

# Apply function to DataFrame column
expanded_df_v1['yearBuilt'] = expanded_df_v1['extractedBuildingInfo'].apply(extract_year_built)

# Print results
print(expanded_df_v1['yearBuilt'].iloc[:20])

# Drop rows where 'yearBuilt' is None
expanded_df_v2 = expanded_df_v1.copy().dropna(subset=['yearBuilt'])

# Create 'builtFactor' column using the apply method
expanded_df_v2['newBuiltFactor'] = expanded_df_v2['yearBuilt'].apply(lambda x: 1 if x >= 2021 else 0)

# Print results
print(expanded_df_v2['newBuiltFactor'].iloc[:20])
expanded_df_v2['newBuiltFactor'].value_counts()
### ***************************************************************************




### ************************ Determining Property Type **********************

def determine_property_type(extractedAmenities):
    apartment_pattern = re.compile(r"\bApartment Features\b", re.IGNORECASE)
    condo_pattern = re.compile(r"\bCondo Features\b", re.IGNORECASE)
    townhome_pattern = re.compile(r"\bTownhome Features\b", re.IGNORECASE)
    #generic_features_pattern = re.compile(r"\bFeatures\b", re.IGNORECASE)

    for line in extractedAmenities.split('\n'):
        if apartment_pattern.search(line):
            return "Apartment"
        elif condo_pattern.search(line):
            return "Condo"
        elif townhome_pattern.search(line):
            return "Townhome"
        # elif generic_features_pattern.search(line):
        #     return "Unknown"
    return "Condo"

def update_prob_prop_type(df):
    df['probPropType'] = df['extractedAmenities'].apply(determine_property_type)
    
update_prob_prop_type(expanded_df_v2)

### ************************************************************

def get_location_type_dummies(df):
    # Create dummy variables for the Zipcode column
    zipcode_dummies = pd.get_dummies(df['zipcode'], prefix='Zipcode', dtype=int, drop_first=True)
    
    # Create dummy variables for the Type column
    type_dummies = pd.get_dummies(df['probPropType'], prefix='propType', dtype=int).drop('propType_Townhome', axis=1)
    
    # Concatenate the original DataFrame with the dummy variables
    df_with_dummies = pd.concat([df, zipcode_dummies, type_dummies], axis=1)
    
    print("Townhome has been removed from Type during one-hot encoding")
    
    return df_with_dummies

expanded_df_v3 = get_location_type_dummies(expanded_df_v2.copy())

zipcode_columns = expanded_df_v3.filter(like='Zipcode_').columns

prop_type_columns = expanded_df_v3.filter(like='propType_').columns



### *********************** Getting ScoreCards *************************

# Your regex patterns
walk_score_regex = re.compile(r'(\d+)\s*Walk Score.*', re.MULTILINE)
transit_score_regex = re.compile(r'(\d+)\s*Transit Score.*', re.MULTILINE)
bike_score_regex = re.compile(r'(\d+)\s*Bike Score.*', re.MULTILINE)

def extract_scores(scorecard_text):
    walk_score_matches = walk_score_regex.findall(scorecard_text)
    transit_score_matches = transit_score_regex.findall(scorecard_text)
    bike_score_matches = bike_score_regex.findall(scorecard_text)
    
    walk_score = float(walk_score_matches[0]) if walk_score_matches else None
    transit_score = float(transit_score_matches[0]) if transit_score_matches else None
    bike_score = float(bike_score_matches[0]) if bike_score_matches else None
    
    return pd.Series([walk_score, transit_score, bike_score])

def add_scores_to_df(df):
    df[['walkScore', 'transitScore', 'bikeScore']] = df['extractedScoreCards'].apply(extract_scores)
    return df


expanded_df_v3 = add_scores_to_df(expanded_df_v3)

expanded_df_v3['rpsf'] = expanded_df_v3['cleanRent'] / expanded_df_v3['area_sqft']


print("\n************* CLEANING ENDED FOR NOW; RETURN TO REGRESSION  **********\n")


##### **********************************************************************

### *** Regression Analysis ****

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

regression_df_v0 = expanded_df_v3.copy()

regression_df_v0.dropna(inplace=True)

independent_num_columns = [
    'bed',
    'bath',
    'area_sqft',
    
    'walkScore',
    'transitScore',
    'bikeScore'
    ]

categorical_columns = [
    'Wooden Floor Feature',
    'Sundeck Features',
    'Free Wifi',
    'Controlled Access',
    '24 Hour Facilities',
    'Accessibility Features',
    'Air Conditioning',
    'Bike/Indoor Bike Storage',
    'Ceiling Fan',
    'Clubhouse',
    'Closets',
    'Energy Efficiency Features',
    'EV Charging',
    'Garden',
    'Granite Countertops',
    'Guest Features',
    'Gym',
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
    'Tile Features',
    'Trash/Garbage Services',
    'Tub/Shower',
    'Views',
    'Vinyl Flooring',
    'Walking/Biking Trails',
    'Washer and Dryer',  
    'Wheelchair Access',
      
    'newBuiltFactor',

    ] + list(zipcode_columns) + list(prop_type_columns)

regression_df_v1 = pd.concat([regression_df_v0['cleanRent'], regression_df_v0[independent_num_columns+categorical_columns]], axis=1)

# Target Varialbe Setup    
Y = regression_df_v1['cleanRent']  

# Independent Variable Setup   
X2 = regression_df_v1[independent_num_columns + categorical_columns]
X2 = sm.add_constant(X2)

# Standardize the features
scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2)

# Create a DataFrame from the scaled features, preserving the column names and indices
X2_scaled_df = pd.DataFrame(X2_scaled, columns=X2.columns, index=X2.index)

# Add constant term to the scaled features
X2_scaled_df = sm.add_constant(X2_scaled_df)

# Fit the OLS model
ks2 = sm.OLS(Y, X2)
ks2_res = ks2.fit()
summary = ks2_res.summary()

# Extract the coefficients table
coef_table = summary.tables[1]

# Convert the coefficients table to a DataFrame
coef_df = pd.DataFrame(coef_table.data[1:], columns=coef_table.data[0])

# Clean the DataFrame
coef_df.columns = coef_df.iloc[0]
coef_df = coef_df[1:]

# Save the DataFrame to a CSV file
output_path = 'OLS_summary_memphis.csv'  # Replace with your desired path
coef_df.to_csv(output_path, index=False)

print(f"OLS summary saved to {output_path}")

### Applying Advanced Regression 

# Getting the statistical summary of dataset
regression_df_columns = regression_df_v1.columns.tolist()
summary_table_org_df = regression_df_v1.describe().T

import seaborn as sns
import matplotlib.pyplot as plt
print("\n**********Plotting Pairplots now******")

'''
sns.pairplot(regression_df_v1[['cleanRent', 
                                'Area in Sq Ft',
                                  'room', 
                                  'bed', 
                                  'bath', 
                                  'Stories', 
                                  'New Built?',
                                  'Gym', 
                                  'Swimming Pool',]])

'''


sns.scatterplot(data=expanded_df_v3, x='newBuiltFactor', y='cleanRent')
plt.title('Scatter plot of Rent vs Newbuilt Factor')
plt.show()


### ******* Applying Machine Learning ************


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from sklearn.svm import SVC

from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures

#regression_df = regression_df_v1.copy() ### needed to avoid constant 


# Drop constant

X3 = X2.iloc[:, 1:]


x_vec = X3

## bed type on hot encoding ## 

y_vec = Y



### ****************************** Saving Scaler for Future Use ********************************

scaler = StandardScaler()

x_vec_standardized = scaler.fit_transform(x_vec[independent_num_columns])
pickle.dump(scaler, open(scaler_filename, 'wb'))
x_vec_standardized = pd.DataFrame(x_vec_standardized, index=x_vec.index, columns=independent_num_columns)

### *******************************************************************************************************




###################### FINAL TRAIN/TEST SPLIT ##################################

x_vec_fnl = pd.concat([x_vec_standardized, x_vec[categorical_columns]], axis=1)

x_vec_fnl_train, x_vec_fnl_test, y_vec_train, y_vec_test = train_test_split(x_vec_fnl, y_vec, test_size=0.2, random_state=42)

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, x_vec_fnl, y_vec, scoring="neg_mean_squared_error", cv=5)).mean()
    return rmse

# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

    

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared




############# Machine Learning Models ##################

models = pd.DataFrame(columns=["Model","MAE","MSE","RMSE","R2 Score","RMSE (Cross-Validation)"])

### Linear Regression ###
ml_method = "Linear OLS Regression"

my_lin_reg = LinearRegression().fit(x_vec_fnl_train, y_vec_train)
predictions = my_lin_reg.predict(x_vec_fnl_test)

mae, mse, rmse, r_squared = evaluation(y_vec_test, predictions)
print("\n***** Current machine-learning model is: ", ml_method, "******")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(my_lin_reg)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "LinearRegression","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
models = pd.concat([models, pd.DataFrame([new_row])], ignore_index=True)

# save the model to disk


pickle.dump(my_lin_reg, open(ols_filename, 'wb'))

### Lasso Regression ###
ml_method = "Lasso Regression"

lasso = Lasso()
lasso.fit(x_vec_fnl_train, y_vec_train)
predictions = lasso.predict(x_vec_fnl_test)

mae, mse, rmse, r_squared = evaluation(y_vec_test, predictions)
print("\n***** Current machine-learning model is: ", ml_method, "******")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(lasso)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "Lasso","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
models = pd.concat([models, pd.DataFrame([new_row])], ignore_index=True)

### Support Vector Machines ###

ml_method = "Support Vector Machines"
my_svr = SVR(C=100000, kernel="rbf").fit(x_vec_fnl_train, y_vec_train)
#my_svc = SVC(kernel='linear', C=1, gamma=0).fit(x_vec_fnl_train, y_vec_train)
predictions = my_svr.predict(x_vec_fnl_test)


# save the model to disk
svr_filename = 'atlanta_svr_model.sav'

pickle.dump(my_svr, open(svr_filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_svr_model = pickle.load(open(svr_filename, 'rb'))


mae, mse, rmse, r_squared = evaluation(y_vec_test, predictions)

print("\n***** Current machine-learning model is: ", ml_method, "******")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(my_svr)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "SVR","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
models = pd.concat([models, pd.DataFrame([new_row])], ignore_index=True)

'''
### Elastic Net ###
ml_method = "Elastic Net"

my_elastic_net = ElasticNet().fit(x_vec_fnl_train, y_vec_train)
predictions = my_elastic_net.predict(x_vec_fnl_test)



mae, mse, rmse, r_squared = evaluation(y_vec_test, predictions)
print("\n***** Current machine-learning model is: ", ml_method, "******")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(my_svr)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "ElasticNet","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
models = pd.concat([models, pd.DataFrame([new_row])], ignore_index=True)
'''
### XGBoost Regressor ###
ml_method = "XGB Boost"
xgb = XGBRegressor(n_estimators=2000, learning_rate=0.015) #max_depth = 4)
xgb.fit(x_vec_fnl_train, y_vec_train)
predictions = xgb.predict(x_vec_fnl_test)

# save the model to disk
xgb_filename = 'atlanta_xgb_model.sav'

pickle.dump(xgb, open(xgb_filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_xgb_model = pickle.load(open(xgb_filename, 'rb'))

mae, mse, rmse, r_squared = evaluation(y_vec_test, predictions)
print("\n***** Current machine-learning model is: ", ml_method, "******")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(xgb)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "XGBRegressor","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
models = pd.concat([models, pd.DataFrame([new_row])], ignore_index=True)

'''
### Polynomial Regression (Degree=2) ###

ml_method = "Polynomial Method"
poly_reg = PolynomialFeatures(degree=2)
X_train_2d = poly_reg.fit_transform(x_vec_fnl_train)
X_test_2d = poly_reg.transform(x_vec_fnl_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_2d, y_vec_train)
predictions = lin_reg.predict(X_test_2d)

mae, mse, rmse, r_squared = evaluation(y_vec_test, predictions)
print("\n***** Current machine-learning model is: ", ml_method, "******")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(lin_reg)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "Polynomial Regression (degree=2)","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
models = pd.concat([models, pd.DataFrame([new_row])], ignore_index=True)
'''


### Final Comparisons ###


models.sort_values(by="RMSE (Cross-Validation)")

plt.figure(figsize=(12,8))
sns.barplot(x=models["Model"], y=models["RMSE (Cross-Validation)"])
plt.title("Models' RMSE Scores (Cross-Validated)", size=15)
plt.xticks(rotation=30, size=12)
plt.show()

print("Sorted model ranked: " , models.sort_values(by="RMSE (Cross-Validation)"))






### *********************** Clean DF Share #############################


#clean_filesharing_path = 'D:\Personal\OneDrive\Pensio Global\My DataScience Projects\Clean Demo/clean_df_demo_20240703.csv'  # Replace with your desired path
# Save the DataFrame to a CSV file
clean_df_demo = expanded_df_v3.drop(['propType',
                                     'rent',
                                    'extractedBuildingName',
                                    'extractedAmenities',
                                    'extractedUniqueFeatures',
                                    'checkSingleRent',
                                    'extractedBuildingInfo',
                                    'extractedScoreCards',
                                    'extractedReview',
                                    'bedbathDetailsWithTypes',
                                    'extractedFloorDetails'
                                    ], 
                                   axis=1).rename(columns={'propName': 'Property Name', 'extractedAddress': "Address", 'cleanRent': 'Rent'})


### ******************************** Clean DF Share *******************************
clean_df_demo['Address'] = clean_df_demo['Address'].apply(lambda x: x.replace(' – ', ', '))
clean_filename = 'clean_df_demo_atlanta_20240923.csv'
clean_filesharing_path = os.path.join(current_directory, clean_filename)

clean_df_demo.to_csv(clean_filesharing_path, index=False)

### ******************************** RAW DF Share *******************************
raw_filename = 'raw_df_demo_atlanta_20240923.csv'
json_filename = 'raw_df_demo_atlanta_20240923.json'  # Specify the JSON file name
raw_filesharing_path = os.path.join(current_directory, raw_filename)
json_filesharing_path = os.path.join(current_directory, json_filename)

merged_df.to_csv(raw_filesharing_path, index=False)
# Save DataFrame to JSON
merged_df.to_json(json_filesharing_path, orient='records', lines=True)


### ******************** Summary Rental *************************************

df3 = expanded_df_v3.copy()

df3.dropna(inplace=True)

### **** Filtering df ************

df3 = df3[(df3['yearBuilt'] >= 2000) & (df3['propType_Condo'] == 1)] #& (df['propType_Condo'] == 1)


# Assuming your DataFrame is named 'df'
df3['RPSF'] = df3['cleanRent'] / df3['area_sqft']



summary_df = df3.groupby('bed').agg({
    'cleanRent': ['mean', 'median'],
    'RPSF': ['mean', 'median']
})

nb_summary = regression_df_v0.groupby('newBuiltFactor').agg({
    'cleanRent': 'mean',
    'rpsf': 'mean'
})

nb_premium_factor = nb_summary.loc[1, 'cleanRent'] / nb_summary.loc[0, 'cleanRent']

import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot 'cleanRent' on the first y-axis
ax1.bar(summary_df.index.astype(str), summary_df['cleanRent']['mean'], label='Average Rent', color='b')
ax1.set_xlabel('Bed Types')
ax1.set_ylabel('Average Rent', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for 'RPSF'
ax2 = ax1.twinx()
ax2.plot(summary_df.index.astype(str), summary_df['RPSF']['mean'], marker='o', color='r', label='Average RPSF')
ax2.set_ylabel('Average RPSF', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add a legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# Set the title
plt.title('Average Rent and RPSF by Bed Types')

# Show the plot
plt.show()


filtered_df_b4 = expanded_df_v3[expanded_df_v3['bed'] == 4]

### ******************** Correlation Matrix Visualization *******************************

summary_output_df = expanded_df_v3[[
     
    'cleanRent',
    
    'bed',
    'bath',
    'area_sqft',
    
    'yearBuilt',
    
    
    'walkScore',
    'transitScore',
    'bikeScore',
    
    'Wooden Floor Feature',
    
    
    
    'Air Conditioning',
    'Bike/Indoor Bike Storage',
    
    'Closets',
    'Clubhouse',
    'Energy Efficiency Features',
    'EV Charging',
    
    
    'Gym',
    
    
    
    'Laundry Facilities',
    
    
    'Parking/Garage/Reserved Parking',
    'Patio/Balcony',
    
    
    'Pool Features',
    
    'Stainless Steel Appliances',
    
    'Views',
    'Vinyl Flooring',
    'Walking/Biking Trails',
    'Washer and Dryer',
    'Granite Countertops',
   
    ] + list(prop_type_columns)]


# Rename columns
summary_output_df = summary_output_df.rename(columns={'cleanRent': 'Rent', 'Pool Features': 'Pool', 'Wooden Floor Feature': 'Hardwood Flooring', 'Closets': 'Large/Walk-in Closets', 'Bike/Indoor Bike Storage': 'Bike Storage'})

plt.figure(figsize=(10,8))
sns.heatmap(summary_output_df.corr(), cmap="RdBu")
plt.title("Correlations Between Rent and Selected Variables", size=15)
plt.show()


### *************** Running Predictions ******************************

final_x_columns = x_vec_fnl.columns.tolist()

all_cols = final_x_columns

# Example new data point



new_data_point = {
    'bed': 1,
    'bath': 1,
    'area_sqft': 790,
    'walkScore': 96,
    'transitScore': 51,
    'bikeScore': 88,
    'Wooden Floor Feature': 1,
    'Granite Countertops': 1,
    'Sundeck Features': 1,
    'Vinyl Flooring': 0,
    'Stainless Steel Appliances': 1,
    'Large/Walk-in Closets':1,
    'Laundry Facilities': 0,
    'Washer and Dryer': 1,
    'Gym': 1,
    'Pool': 1,
    'Views': 0,
    'newBuiltFactor': 0,
    'propType_Condo': 0,
    'propType_Apartment': 1,
    
    'Energy Efficiency Features': 0,
    'EV Charging': 0,
    
    '24 Hour Facilities': 1,
    'Clubhouse': 1,
    
    # Add other provided features...
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
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))
new_data_num_scaled = loaded_scaler.transform(new_data_df[independent_num_columns])
new_data_num_scaled_df = pd.DataFrame(new_data_num_scaled, columns=independent_num_columns)

# Concatenate scaled numerical features and categorical features
new_data_final = pd.concat([new_data_num_scaled_df, new_data_df[categorical_columns]], axis=1)





### ***************************** Applying ANN ***************************************

from sklearn import preprocessing
# from keras.models import Sequential
# from keras.layers import Dense
# # creating a model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam

# evaluation on test data
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler


### ***************************** Applying Random Forests ***************************************

from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators= 400)

clf.fit(x_vec_fnl_train, y_vec_train)

rf_score = clf.score(x_vec_fnl_test, y_vec_test)


# save the model to disk
rf_filename = 'atlanta_rf_model.sav'

pickle.dump(clf, open(rf_filename, 'wb'))

rf_predict = clf.predict(new_data_final)

# XGBoost prediction
loaded_xgb_model = pickle.load(open(xgb_filename, 'rb'))
xgb_prediction = loaded_xgb_model.predict(new_data_final)
print("\nXGBoost Prediction:", xgb_prediction[0])

# SVR prediction
loaded_svr_model = pickle.load(open(svr_filename, 'rb'))
svr_prediction = loaded_svr_model.predict(new_data_final)
print("\nSVR Prediction:", svr_prediction[0])

# RF prediction
loaded_rf_model = pickle.load(open(rf_filename, 'rb'))
rf_prediction = loaded_rf_model.predict(new_data_final)
print("\nRandom Forest Prediction:", rf_prediction[0])

# Linear Regression prediction
loaded_ols_model = pickle.load(open(ols_filename, 'rb'))
lr_prediction = loaded_ols_model.predict(new_data_final)
print("OLS Prediction:", lr_prediction[0])



