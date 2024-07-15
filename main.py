import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_excel(r'C:\Users\user\Downloads\Fashion_recommendation.xlsx')
print(df)
df.columns = df.columns.str.strip().str.replace('\n', '')
# print(df.columns)

top_wear_formal = ["Shirt", "Blazer", "Suit jacket"]
bottom_wear_formal = ["Chinos", "Formal pant"]
top_wear_casual = ["T-shirt", "Polo Shirt", "Casual Shirt"]
bottom_wear_casual = ["Chinos", "Jeans", "Shorts"]
#
# Creating DataFrames with the specified indices
df1 = pd.DataFrame(top_wear_formal, index=[20, 21, 22], columns=['Top Wear Formal'])
df2 = pd.DataFrame(bottom_wear_formal, index=[30, 31], columns=['Bottom Wear Formal'])
df3 = pd.DataFrame(top_wear_casual, index=[40, 41, 42], columns=['Top Wear Casual'])
df4 = pd.DataFrame(bottom_wear_casual, index=[50, 51, 52], columns=['Bottom Wear Casual'])
#
df['Preferred Top Wear - Formal'] = df['Preferred Top Wear - Formal'].str.extract(r'(\d+)').astype(int)
df['Preferred Bottom Wear - Formal'] = df['Preferred Bottom Wear - Formal'].str.extract(r'(\d+)').astype(int)
df['Preferred Top Wear'] = df['Preferred Top Wear'].str.extract(r'(\d+)').astype(int)
df['Preferred Bottom Wear'] = df['Preferred Bottom Wear'].str.extract(r'(\d+)').astype(int)
#
# df.head()
#
df['Preferred Top Wear - Formal'] = df['Preferred Top Wear - Formal'].replace([1, 2, 3], [20, 21, 22])
df['Preferred Bottom Wear - Formal'] = df['Preferred Bottom Wear - Formal'].replace([1, 2], [30, 31])
df['Preferred Top Wear'] = df['Preferred Top Wear'].replace([1, 2, 3], [40, 41, 42])
df['Preferred Bottom Wear'] = df['Preferred Bottom Wear'].replace([1, 2, 3], [50, 51, 52])
#
colors = [
    'Black', 'White', 'Grey', 'Navy Blue', 'Brown', 'Beige',
    'Red', 'Blue', 'Green', 'Pink', 'Khaki', 'Olive Green', 'Yellow'
]
#
col = pd.DataFrame(colors, columns=['Color'])
col['Number'] = range(100, 100 + len(colors))
color_mapping = col.set_index('Color')['Number'].to_dict()
print(color_mapping)
#
# Replace the colors in the 'Dress Colour' column with their corresponding numbers
df['Dress Colour1'] = df['Dress Colour1'].replace(color_mapping)
df['Dress Colour2'] = df['Dress Colour2'].replace(color_mapping)
df['Dress Colour3'] = df['Dress Colour3'].replace(color_mapping)
df['Dress Colour4'] = df['Dress Colour4'].replace(color_mapping)
#
# df.head()
# print(df.columns)
#
formal_df = df.iloc[:, :7]
casual_df = df.iloc[:, [0, 1, 2, 7, 8, 9, 10]]

# print(formal_df.columns)
# print(casual_df.columns)
#
#
# formal_df.head()
#
# casual_df.head()
#

if 'Skin Color' not in casual_df.columns or 'Body Shape' not in casual_df.columns:
    raise KeyError("'Skin Color' and/or 'Body Shape' column not found in filtered DataFrame")
scaler = StandardScaler()
X = casual_df[['Skin Color', 'Body Shape']]
X_scaled = scaler.fit_transform(X)
#
# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Set n_init explicitly
casual_df.loc[:, 'Cluster'] = kmeans.fit_predict(X_scaled)
# print(casual_df.columns)

top_wear_casual_list = ["T-shirt", "Polo Shirt", "Casual Shirt"]
bottom_wear_casual_list = ["Chinos", "Jeans", "Shorts"]

def recommend_clothing(skin_color, body_shape):
    # Standardize the input
    user_data = scaler.transform([[skin_color, body_shape]])

    # Predict the cluster for the new user
    cluster = kmeans.predict(user_data)[0]

    # Filter the DataFrame to the users in the same cluster
    cluster_df = casual_df[casual_df['Cluster'] == cluster]

    recommendations = {}

    # Check and recommend based on available columns
    if 'Preferred Top Wear' in cluster_df.columns:
        top_wear = cluster_df['Preferred Top Wear'].mode()[0]
        print(top_wear_casual_list[top_wear - 40])
        recommendations['Preferred Top Wear'] = top_wear_casual_list[top_wear - 40]  # Convert to int
    if 'Dress Colour3' in cluster_df.columns:
        dress_color2 = cluster_df['Dress Colour3'].mode()[0]
        print(colors[dress_color2 - 100])
        recommendations['Dress Colour3'] = colors[dress_color2 - 100]  # Convert to int
    if 'Preferred Bottom Wear' in cluster_df.columns:
        bottom_wear = cluster_df['Preferred Bottom Wear'].mode()[0]
        print(bottom_wear_casual_list[bottom_wear - 50])
        recommendations['Preferred Bottom Wear'] = bottom_wear_casual_list[bottom_wear - 50]  # Convert to int
    if 'Dress Colour4' in cluster_df.columns:
        dress_color = cluster_df['Dress Colour4'].mode()[0]
        print(colors[dress_color2 - 100])
        recommendations['Dress Colour4'] = colors[dress_color - 100]  # Convert to int

    print(recommendations)
    return recommendations

#
# Example usage
skin_color = 6
body_shape = 6
recom = recommend_clothing(skin_color, body_shape)
print(recom)
#
scaler1 = StandardScaler()
X1 = formal_df[['Skin Color', 'Body Shape']]
X1_scaled = scaler1.fit_transform(X1)

kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
formal_df.loc[:,'Cluster1'] = kmeans1.fit_predict(X1_scaled)
# print(formal_df.head())
#

top_wear_formal_list = ["Shirt", "Blazer", "Suit jacket"]
bottom_wear_formal_list = ["Chinos", "Formal pant"]
def recommend1_clothing(skin_color, body_shape):
    # Standardize the input
    user_data = scaler1.transform([[skin_color, body_shape]])

    # Predict the cluster for the new user
    cluster1 = kmeans1.predict(user_data)[0]

    # Filter the DataFrame to the users in the same cluster
    cluster_df1 = formal_df[formal_df['Cluster1'] == cluster1]
    # print(cluster_df1.columns)
    recommendations = {}

    # Check and recommend based on available columns
    if 'Preferred Top Wear - Formal' in cluster_df1.columns:
        top_wear_formal = cluster_df1['Preferred Top Wear - Formal'].mode()[0]
        print(top_wear_formal_list[top_wear_formal-20])
        recommendations['Preferred Top Wear - Formal'] = top_wear_formal_list[top_wear_formal-20]  # Convert to int
    if 'Dress Colour1' in cluster_df1.columns:
        dress_color = cluster_df1['Dress Colour1'].mode()[0]
        print(colors[dress_color-100])
        recommendations['Dress Colour'] = colors[dress_color-100]  # Convert to int
    if 'Preferred Bottom Wear - Formal' in cluster_df1.columns:
        bottom_wear_formal = cluster_df1['Preferred Bottom Wear - Formal'].mode()[0]
        print(bottom_wear_formal_list[bottom_wear_formal - 30])
        recommendations['Preferred Bottom Wear - Formal'] = bottom_wear_formal_list[bottom_wear_formal - 30]  # Convert to int
    if 'Dress Colour2' in cluster_df1.columns:
        dress_color1 = cluster_df1['Dress Colour1'].mode()[0]
        print(colors[dress_color1 - 100])
        recommendations['Dress Colour2'] = colors[dress_color1 - 100]  # Convert to int

    print(recommendations)
    return recommendations



# skin_color = 6
# body_shape = 6
# recom1 = recommend1_clothing(skin_color, body_shape)
# print(recom1)
