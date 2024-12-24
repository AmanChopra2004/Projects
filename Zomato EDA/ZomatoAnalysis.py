# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plot style
plt.style.use('dark_background')

# Reading CSV
df = pd.read_csv('C:\Users\amanc\OneDrive\Desktop\ZomatoAnalysis')  # Update the path to your dataset 
df.head()

# Displaying shape and columns
df.shape
df.columns

# Dropping unnecessary columns
df = df.drop(['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list'], axis=1)
df.head()

# Displaying information about the DataFrame
df.info()

# Dropping duplicate rows
df.drop_duplicates(inplace=True)
df.shape

# Cleaning Rate Column
df['rate'].unique()

# Removing "NEW", "-", and "/5" from Rate Column
def handle_rate(value):
    if value == 'NEW' or value == '-':
        return np.nan
    else:
        value = str(value).split('/')
        value = value[0]
        return float(value)

df['rate'] = df['rate'].apply(handle_rate)
df['rate'].head()

# Filling Null Values in Rate Column with Mean
df['rate'].fillna(df['rate'].mean(), inplace=True)
df['rate'].isnull().sum()
df.info()

# Dropping Null Values
df.dropna(inplace=True)
df.head()

# Renaming columns for clarity
df.rename(columns={'approx_cost(for two people)': 'Cost2plates', 'listed_in(type)': 'Type'}, inplace=True)
df.head()

# Cleaning Location Column
df['location'].unique()
df['listed_in(city)'].unique()

# Keeping only one between 'listed_in(city)' and 'location'
df = df.drop(['listed_in(city)'], axis=1)

# Cleaning Cost2Plates Column
df['Cost2plates'].unique()

# Removing commas from Cost2Plates Column
def handle_comma(value):
    value = str(value)
    if ',' in value:
        value = value.replace(',', '')
        return float(value)
    else:
        return float(value)

df['Cost2plates'] = df['Cost2plates'].apply(handle_comma)
df['Cost2plates'].unique()

df.head()

# Cleaning Rest Type Column
rest_types = df['rest_type'].value_counts(ascending=False)
rest_types_lessthan1000 = rest_types[rest_types < 1000]

# Making Rest Types less than 1000 in frequency as 'others'
def handle_rest_type(value):
    if value in rest_types_lessthan1000:
        return 'others'
    else:
        return value

df['rest_type'] = df['rest_type'].apply(handle_rest_type)
df['rest_type'].value_counts()

# Cleaning Location Column
location = df['location'].value_counts(ascending=False)
location_lessthan300 = location[location < 300]

def handle_location(value):
    if value in location_lessthan300:
        return 'others'
    else:
        return value

df['location'] = df['location'].apply(handle_location)
df['location'].value_counts()

# Cleaning Cuisines Column
cuisines = df['cuisines'].value_counts(ascending=False)
cuisines_lessthan100 = cuisines[cuisines < 100]

def handle_cuisines(value):
    if value in cuisines_lessthan100:
        return 'others'
    else:
        return value

df['cuisines'] = df['cuisines'].apply(handle_cuisines)
df['cuisines'].value_counts()

df.head()

# Data is Clean, Let's jump to Visualization

# Count Plot of Various Locations
plt.figure(figsize=(16, 10))
ax = sns.countplot(df['location'])
plt.xticks(rotation=90)

# Visualizing Online Order
plt.figure(figsize=(6, 6))
sns.countplot(df['online_order'], palette='inferno')

# Visualizing Book Table
plt.figure(figsize=(6, 6))
sns.countplot(df['book_table'], palette='rainbow')

# Visualizing Online Order vs Rate
plt.figure(figsize=(6, 6))
sns.boxplot(x='online_order', y='rate', data=df)

# Visualizing Book Table vs Rate
plt.figure(figsize=(6, 6))
sns.boxplot(x='book_table', y='rate', data=df)

# Visualizing Online Order Facility, Location Wise
df1 = df.groupby(['location', 'online_order'])['name'].count()
df1.to_csv('location_online.csv')
df1 = pd.read_csv('location_online.csv')
df1 = pd.pivot_table(df1, values=None, index=['location'], columns=['online_order'], fill_value=0, aggfunc=np.sum)
df1.plot(kind='bar', figsize=(15, 8))

# Visualizing Book Table Facility, Location Wise
df2 = df.groupby(['location', 'book_table'])['name'].count()
df2.to_csv('location_booktable.csv')
df2 = pd.read_csv('location_booktable.csv')
df2 = pd.pivot_table(df2, values=None, index=['location'], columns=['book_table'], fill_value=0, aggfunc=np.sum)
df2.plot(kind='bar', figsize=(15, 8))

# Visualizing Types of Restaurants vs Rate
plt.figure(figsize=(14, 8))
sns.boxplot(x='Type', y='rate', data=df, palette='inferno')

# Grouping Types of Restaurants, Location Wise
df3 = df.groupby(['location', 'Type'])['name'].count()
df3.to_csv('location_Type.csv')
df3 = pd.read_csv('location_Type.csv')
df3 = pd.pivot_table(df3, values=None, index=['location'], columns=['Type'], fill_value=0, aggfunc=np.sum)
df3.plot(kind='bar', figsize=(36, 8))

# No. of Votes, Location Wise
df4 = df[['location', 'votes']]
df4.drop_duplicates()
df5 = df4.groupby(['location'])['votes'].sum()
df5 = df5.to_frame()
df5 = df5.sort_values('votes', ascending=False)
df5.head()

plt.figure(figsize=(15, 8))
sns.barplot(df5.index, df5['votes'])
plt.xticks(rotation=90)

# Visualizing Top Cuisines
df6 = df[['cuisines', 'votes']]
df6.drop_duplicates()
df7 = df6.groupby(['cuisines'])['votes'].sum()
df7 = df7.to_frame()
df7 = df7.sort_values('votes', ascending=False)
df7 = df7.iloc[1:, :]

plt.figure(figsize=(15, 8))
sns.barplot(df7.index, df7['votes'])
plt
