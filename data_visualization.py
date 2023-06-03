from google.colab import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

uploaded = files.upload()

#Import the dataset into a pandas dataframe. Then report how many rows and columns are present in the dataset.
#----------------------------------------------------------------------------------
filename = "startup_info_.csv"
df = pd.read_csv(filename)

num_rows, num_columns = df.shape

print(f"Dataset \"{filename}\", has {num_rows} rows and {num_columns} columns.\n")
#----------------------------------------------------------------------------------


#Call the describe method to see summary statistics of the numerical attribute columns
#----------------------------------------------------------------------------------
summary_statistics = df.describe()

print(f"Summary Statistics: \n{summary_statistics}\n")
#----------------------------------------------------------------------------------


# List all attribute columns
# ----------------------------------------------------------------------------------
attribute_columns = df.columns.tolist()

print(f"Attribute Columns: {attribute_columns}\n")
# ----------------------------------------------------------------------------------


#The "Unnamed: 0","Unnamed: 6", "state_code.1" and "object_id" feature columns are not useful. Drop them in-place.
#----------------------------------------------------------------------------------
df = df.drop(columns=["Unnamed: 0", "Unnamed: 6", "state_code.1", "object_id"])
#----------------------------------------------------------------------------------


# Show all the numeric columns and save it to a new dataframe.
# ----------------------------------------------------------------------------------
numeric_columns = df.select_dtypes(include=["float64", "int64"])

print(f"Numeric Columns: {numeric_columns.columns.tolist()}\n")
# ----------------------------------------------------------------------------------


# Plot distributions of the numeric columns using histogram and record the skew of each distribution. (Note: positive value = right skewed, negative value = left skewed)
# ----------------------------------------------------------------------------------
numeric_columns.hist()
skew = numeric_columns.skew()

plt.show()

print(f"Skew of Numeric Columns:\n{skew}\n")
# ----------------------------------------------------------------------------------


# Show all the categorical columns and save it to a new dataframe
# ----------------------------------------------------------------------------------
categorical_columns = df.select_dtypes(include=["object"])

print(f"Categorical Columns: {categorical_columns.columns.tolist()}\n")
# ----------------------------------------------------------------------------------


# Show a list with column wise count of missing values and display the list in count wise descending order
# ----------------------------------------------------------------------------------
missing_values = df.isna().sum().sort_values(ascending=False)

print(f"Missing Values in Descending Order:\n{missing_values}\n")
# ----------------------------------------------------------------------------------


# Show columnwise percentage of missing values.
# ----------------------------------------------------------------------------------
missing_values_per = (missing_values / df.shape[0]) * 100
missing_values_per_out = missing_values_per.map("{:.2f}%".format)

print(f"Missing Values in Descending Order:\n{missing_values_per_out}\n")
# ----------------------------------------------------------------------------------


# Display a bar plot to visualize only the columns with missing values and their percentage count.
# ----------------------------------------------------------------------------------
missing_values_per.plot(kind='bar')
plt.title("Columns With Missing Values and Their Percentage Count")
plt.xlabel("Columns With Missing Values")
plt.ylabel("Missing Values (%)")

plt.show()
# ----------------------------------------------------------------------------------


# Copy the dataframe to a new one. Then using scikitlearn's Label Encoder, transform the "status" column to 0-1
# ----------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

new_df = df.copy()
new_df["status"] = LabelEncoder().fit_transform(new_df["status"])
# ----------------------------------------------------------------------------------


# Use seaborn's heatmap to visualize the correlation between numeric features.
# ----------------------------------------------------------------------------------
numeric_df = new_df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

sns.heatmap(corr, annot=True, cmap="Reds")
plt.title("Correlation between Numeric Features")

plt.show()
# ----------------------------------------------------------------------------------


# Use seaborn's countplot to visualize relationship between "state_code" and "labels". Comment on which state produced majority of successful startups
# ----------------------------------------------------------------------------------
#show countplot
sns.countplot(x="state_code", hue="status", data=new_df)
plt.title("Relationship between State Code and Startup Status")

plt.show()

#show state with most successful startups
most_startups = new_df.loc[new_df["status"] == 1].groupby("state_code")["status"].count().sort_values(ascending=False).reset_index(name="count")

print(f"# State Code: {most_startups.iloc[0, 0]} produced majority of successful startups")
# ----------------------------------------------------------------------------------


# Use seaborn's countplot to visualize relationship between "milestones" and "labels". Comment on which milestone made the statistically highest number of successful startups
# ----------------------------------------------------------------------------------
#show countplot
sns.countplot(x="milestones", hue="labels", data=new_df)
plt.title("Relationship between Milestones and Labels")

plt.show()

#show state with most successful startups
highest_count_milestone = new_df.loc[new_df["labels"] == 1].groupby("milestones")["labels"].count().sort_values(ascending=False).reset_index(name="count")

print(f"\t# Companies with {highest_count_milestone.iloc[0, 0]} milestones, made the highest number of successful startups")
# ----------------------------------------------------------------------------------


# Drop features with duplicate values in -place, then show dataframe's new shape.
# ----------------------------------------------------------------------------------
new_df.drop_duplicates(inplace=True)

print(f"Dataframe's new shape: {new_df.shape}\n")
# ----------------------------------------------------------------------------------


# From correlation heatmap above, comment on which feature has the highest correlation with "funding_rounds". Visualize a scatterplot with that and "funding_rounds"
# ----------------------------------------------------------------------------------
highest_corr = corr.loc["funding_rounds"].abs().sort_values(ascending=False).index[1]

print(f"\t# {highest_corr}  has the highest correlation with \"funding_rounds\"\n")

sns.scatterplot(x=highest_corr, y="funding_rounds", data=numeric_df)
plt.title(f"Scatterplot with {highest_corr} and Funding Rounds")

plt.show()
# ----------------------------------------------------------------------------------


# Show boxplots for the numeric features to detect outliers.
# ----------------------------------------------------------------------------------

sns.boxplot(data=numeric_df)
plt.title("Boxplots for Numeric Features to Detect Outliers")

plt.show()
# ----------------------------------------------------------------------------------






































