import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt     
import seaborn as sns

df = pd.read_csv('ahm_data.csv')

# data cleaning
df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.lower().str.replace(' ','_').str.replace('price_in_cr','price').str.replace('area_type','area')

df['price'] = df['price'].astype(str).str.replace(',','').astype(float)
df['rate_per_sqft'] = df['rate_per_sqft'].astype(str).str.replace(',','').astype(float)

# Additional Data Cleaning
# Remove rows with all missing values
df = df.dropna(how='all')

# Remove rows where price is missing (critical for analysis)
df = df.dropna(subset=['price'])

# Remove rows with missing BHK type
df = df.dropna(subset=['bhk_type'])

# Remove outliers: Rate per sqft > 15000 (likely data entry errors)
df = df[df['rate_per_sqft'] < 15000]

# Remove negative prices or zero prices
df = df[df['price'] > 0]

# Remove rows where area_in_sqft is missing for price calculations
df = df.dropna(subset=['area_in_sqft'])

# Remove rows with area_in_sqft = 0 or very small (<100)
df = df[df['area_in_sqft'] >= 100]

# Fill missing rate_per_sqft by calculating from price/area
df['rate_per_sqft'] = df['rate_per_sqft'].fillna(df['price'] * 10000000 / df['area_in_sqft'])

# Create clean location names (remove extra location details, keep primary location)
df['location'] = df['location'].str.split(',').str[0].str.strip()

# Remove leading/trailing spaces in location
df['location'] = df['location'].str.strip()

# Display data cleaning summary
print(f"Total rows after cleaning: {len(df)}")
print(f"Missing values in price column: {df['price'].isna().sum()}")
print(f"Missing values in rate_per_sqft: {df['rate_per_sqft'].isna().sum()}")
print(f"Data types:\n{df.dtypes}")
print(f"\nDataset shape: {df.shape}")


# quetion:1 Which is the costliest flat in the dataset ?

df.loc[df['price'].idxmax()]
print("and:1  The costliest flat is located at {} with a price of {}.".format(df.loc[df['price'].idxmax()]['location'], df.loc[df['price'].idxmax()]['price']))

# quetion:2 Which is the cheapest flat in the dataset ?

df.loc[df['price'].idxmin()]
print("and:2  The cheapest flat is located at {} with a price of {}.".format(df.loc[df['price'].idxmin()]['location'], df.loc[df['price'].idxmin()]['price']))

# quetion:3 Which is the most expensive area in the dataset ?

area_price = df.groupby('area')['price'].mean()
most_expensive_area = area_price.idxmax()   
print("and:3  The most expensive area is {} with an average price of {} cr.".format(most_expensive_area, area_price[most_expensive_area]))

# qution:4 Which is the least expensive area in the dataset ?
least_expensive_area = area_price.idxmin()
print("and:4  The least expensive area is {} with an average price of {} cr.".format(least_expensive_area, area_price[least_expensive_area]))

# qutioin:5 What is the average price of flats in the dataset ?
average_price = df['price'].mean()
print("and:5  The average price of flats in the dataset is {} cr.".format(average_price))

# qution:6 What is the average price per square foot of flats in the dataset ?
average_price_per_sqft = df['rate_per_sqft'].mean()
print("and:6  The average price per square foot of flats in the dataset is {}.".format(average_price_per_sqft))


# qution:7 Line Chart: Average Price Trend by BHK Type
plt.figure(figsize=(12, 6))
bhk_price_trend = df.groupby('bhk_type')['price'].mean().sort_index()
sns.lineplot(x=bhk_price_trend.index, y=bhk_price_trend.values, marker='o', linewidth=2.5, markersize=10, color='steelblue')
plt.title('Average Price Trend by BHK Type', fontsize=14, fontweight='bold')
plt.xlabel('BHK Type', fontsize=12)
plt.ylabel('Average Price (Cr)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# qution:8 Line Chart: Average Rate per Sqft by BHK Type
plt.figure(figsize=(12, 6))
bhk_rate_trend = df.groupby('bhk_type')['rate_per_sqft'].mean().sort_index()
sns.lineplot(x=bhk_rate_trend.index, y=bhk_rate_trend.values, marker='s', linewidth=2.5, markersize=10, color='coral')
plt.title('Average Rate per Sqft Trend by BHK Type', fontsize=14, fontweight='bold')
plt.xlabel('BHK Type', fontsize=12)
plt.ylabel('Rate per Sqft (₹)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# qution:9 Line Chart: Average Price by Property Type
plt.figure(figsize=(12, 6))
property_price_trend = df.groupby('property_type')['price'].mean()
sns.lineplot(x=property_price_trend.index, y=property_price_trend.values, marker='D', linewidth=2.5, markersize=10, color='green')
plt.title('Average Price Trend by Property Type', fontsize=14, fontweight='bold')
plt.xlabel('Property Type', fontsize=12)
plt.ylabel('Average Price (Cr)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# qution:10 Line Chart: Average Price by Area Type
plt.figure(figsize=(12, 6))
area_price_trend = df.groupby('area')['price'].mean()
sns.lineplot(x=area_price_trend.index, y=area_price_trend.values, marker='^', linewidth=2.5, markersize=10, color='purple')
plt.title('Average Price Trend by Area Type', fontsize=14, fontweight='bold')
plt.xlabel('Area Type', fontsize=12)
plt.ylabel('Average Price (Cr)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()