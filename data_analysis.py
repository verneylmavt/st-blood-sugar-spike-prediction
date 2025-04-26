#!/usr/bin/env python
# coding: utf-8

# ## Data Preprocessing

# In[5]:


import os
import subprocess

import joblib
import zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler


# In[2]:


def impute_df(df, bg_cols, insulin_cols, carbs_cols, hr_cols, steps_cols, cals_cols):
    df = df.copy()
    
    
    # Blood Glucose & Heart Rate: Linear Interpolation & Mean
    # Apply linear interpolation row-wise for bg_cols, then replace any remaining NaNs with the row mean.
    df[bg_cols] = df[bg_cols].apply(
        lambda row: row.interpolate(method='linear', limit_direction='both'), axis=1)
    df[bg_cols] = df[bg_cols].fillna(df[bg_cols].mean(axis=1))

    # Similarly, apply linear interpolation row-wise for hr_cols and fill remaining NaNs with the row mean.
    df[hr_cols] = df[hr_cols].apply(
        lambda row: row.interpolate(method='linear', limit_direction='both'), axis=1)
    df[hr_cols] = df[hr_cols].fillna(df[hr_cols].mean(axis=1))
    
    
    # Insulin & Carbs: Fill NaN w/ 0
    # For insulin_cols and carbs_cols, fill missing values with 0.
    df[insulin_cols + carbs_cols] = df[insulin_cols + carbs_cols].fillna(0)
    
    
    # Steps & Calories: Fill NaN w/ 0
    # For steps_cols and cals_cols, fill missing values with 0.
    df[steps_cols + cals_cols] = df[steps_cols + cals_cols].fillna(0)
    
    
    # Residual: Forward Fill
    # For any leftover missing values, use forward fill then backward fill as a last resort.
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df


# In[3]:


def augment_df(df):
    df = df.copy()
    
    # Define a noise factor to scale the noise to 1% of the feature's standard deviation
    noise_factor = 0.01
    
    # Define columns that should not be augmented (identifiers and target)
    exclude_cols = ['id', 'p_num', 'time', 'bg+1:00']
    
    # Get all numeric columns and exclude the ones specified above
    numeric_aug_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_aug_cols = [col for col in numeric_aug_cols if col not in exclude_cols]
    
    # Create a copy for augmented data
    df_aug = df.copy()
    
    # Add Gaussian noise to each numeric column for augmentation
    for col in numeric_aug_cols:
        std_val = df[col].std()  # Get standard deviation for scaling the noise
        df_aug[col] += np.random.normal(0, noise_factor * std_val, size=len(df))
    
    # Concatenate the original and augmented data, effectively doubling the dataset size
    return pd.concat([df, df_aug], ignore_index=True)


# In[4]:


def select_df(df, bg_cols, insulin_cols, carbs_cols, hr_cols, steps_cols, cals_cols):
    df = df.copy()
    
    # For each sensor type, select only the last 24 time-steps.
    latest_bg_cols     = bg_cols[-24:]
    latest_insulin_cols = insulin_cols[-24:]
    latest_carbs_cols   = carbs_cols[-24:]
    latest_hr_cols      = hr_cols[-24:]
    latest_steps_cols   = steps_cols[-24:]
    latest_cals_cols    = cals_cols[-24:]
    
    # Combine the selected columns for time-series data
    latest_ts_cols = (
        latest_bg_cols +
        latest_insulin_cols +
        latest_carbs_cols +
        latest_hr_cols +
        latest_steps_cols +
        latest_cals_cols
    )

    # Include the 'id' column and if available, the target column 'bg+1:00'
    selected_cols = ["id"] + latest_ts_cols
    if "bg+1:00" in df.columns:
        selected_cols += ["bg+1:00"]
    
    return df[selected_cols]


# In[5]:


def scale_df(df):
    # Create a copy of the dataframe
    df = df.copy()
    
    target_col = 'bg+1:00'
    # Identify all numeric columns except the target
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    
    # Scale features using StandardScaler
    feature_scaler = StandardScaler()
    df[numeric_columns] = feature_scaler.fit_transform(df[numeric_columns])

    # Scale target column separately
    target_scaler = StandardScaler()
    if target_col in df.columns:
        df[[target_col]] = target_scaler.fit_transform(df[[target_col]])
    
    return df, feature_scaler, target_scaler


# In[6]:


def preprocess_df(df):
    
    df = df.copy()
    
    # Identify columns for each type of feature using their prefixes
    bg_cols      = [col for col in df.columns if col.startswith("bg-")]
    insulin_cols = [col for col in df.columns if col.startswith("insulin-")]
    carbs_cols   = [col for col in df.columns if col.startswith("carbs-")]
    hr_cols      = [col for col in df.columns if col.startswith("hr-")]
    steps_cols   = [col for col in df.columns if col.startswith("steps-")]
    cals_cols    = [col for col in df.columns if col.startswith("cals-")]
    # Identify activity columns which are dropped from further analysis
    activity_cols= [col for col in df.columns if col.startswith("activity-")]

    df.drop(columns=activity_cols, inplace=True) # Remove activity features as they are not needed
    

    
    # ================================
    # Data Imputation
    # ================================
    
    # Impute missing values using the impute_df function
    df_imputed = impute_df(df, bg_cols, insulin_cols, carbs_cols, hr_cols, steps_cols, cals_cols)
    
    print("NaN Values (Data Imputation):")
    print(df_imputed.isna().any().any())
    print("\n")
    
    
    
    # ================================
    # Data Augmentation
    # ================================
    
    # Apply augmentation if the target column is available (i.e., training data)
    if "bg+1:00" in df_imputed.columns:
        df_augmented = augment_df(df_imputed)
    else:
        df_augmented = df_imputed
    
    print("NaN Values (Data Augmentation):")
    print(df_augmented.isna().any().any())
    print("\n")
    
    
    
    # ================================
    # Data Selection
    # ================================
    
    # Select only the most recent 24 data points from each feature category
    df_selected = select_df(df_augmented, bg_cols, insulin_cols, carbs_cols, hr_cols, steps_cols, cals_cols)
    
    print("NaN Values (Data Selection):")
    print(df_selected.isna().any().any())
    print("\n")
    
    print("Columns:")
    for i in df_selected.columns:
        print(i)
    print("\n")
    
    
    
    # ================================
    # Data Scaling
    # ================================
    
    # Standardize the features and the target variable
    df_scaled, feature_scaler, target_scaler = scale_df(df_selected)
    
    print("NaN Values (Data Scaling):")
    print(df_scaled.isna().any().any())
    print("\n")
    
    return df_scaled, feature_scaler, target_scaler


# ### Data Importing

# In[7]:


# Define file paths for training and testing data
train_path = "./.data/train.csv.zip"
test_path = "./.data/test.csv"

# If train data doesn't exist, create the .data folder and download the dataset from Kaggle
if not os.path.exists(train_path):
    os.makedirs("./.data", exist_ok=True)
    print("Downloading BrisT1D ...")
    subprocess.run(["kaggle", "competitions", "download", "-c", "brist1d", "-p", "./.data"], check=True)

# Open the zipped training file and load the train.csv as a DataFrame
with zipfile.ZipFile(train_path) as z:
    with z.open("train.csv") as f:
        df_train = pd.read_csv(f)
            
# Load the test CSV directly into a DataFrame
df_test = pd.read_csv("./.data/test.csv")


# ### Data Transformation

# In[8]:


# Print raw training data for initial inspection
print(df_train)


# In[9]:


# Preprocess the training data using the defined pipeline
df_train_preprocessed, df_train_preprocessed_feature_scaler, df_train_preprocessed_target_scaler = preprocess_df(df_train)


# In[10]:


# Print the preprocessed training data to verify transformations
print(df_train_preprocessed)


# In[11]:


# Print raw testing data for inspection
print(df_test)


# In[12]:


# Preprocess the testing data; note that augmentation is skipped if target is absent
df_test_preprocessed, df_test_preprocessed_feature_scaler, df_test_preprocessed_target_scaler = preprocess_df(df_test)


# In[13]:


# Print the preprocessed testing data to verify transformations
print(df_test_preprocessed)


# ### Data Exporting

# In[14]:


# Export the preprocessed training and testing data to CSV files
df_train_preprocessed.to_csv("./.data/train_aug.csv", index=False)
df_test_preprocessed.to_csv("./.data/test_aug.csv", index=False)


# In[15]:


# Save the feature and target scalers for later use (e.g., during model inference)
joblib.dump(df_train_preprocessed_feature_scaler, './.data/feature_scaler_aug.pkl')
joblib.dump(df_train_preprocessed_target_scaler, './.data/target_scaler_aug.pkl')


# ## Exploratory Data Analysis (EDA)

# In[16]:


import os
import subprocess

import joblib
import zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler


# In[17]:


# Reload the preprocessed data and the target scaler from disk
df_train_preprocessed = pd.read_csv("./.data/train_aug.csv")
df_test_preprocessed = pd.read_csv("./.data/test_aug.csv")
target_scaler = joblib.load('./.data/target_scaler_aug.pkl')


# In[18]:


# Inverse transform the scaled target variable to restore it to its original scale for interpretability
df_train_preprocessed["bg+1:00"] = target_scaler.inverse_transform(
    df_train_preprocessed["bg+1:00"].values.reshape(-1, 1)
).ravel()


# In[19]:


# Extract the participant number from the 'id' column using a regex, store it in a new column 'p_num'
df_train_preprocessed['p_num'] = df_train_preprocessed['id'].str.extract(r"(p\d{2})")


# ### Descriptive Analysis

# In[20]:


# Select the target column for analysis
bg_1 = df_train_preprocessed["bg+1:00"]


# In[21]:


# Compute global summary statistics for the target variable
global_stats = {
    "mean": bg_1.mean(),
    "median": bg_1.median(),
    "std_dev": bg_1.std(),
    "min": bg_1.min(),
    "max": bg_1.max(),
    "sum": bg_1.sum(),
    "count": bg_1.count(),
    "q1": bg_1.quantile(0.25),
    "q3": bg_1.quantile(0.75),
}

global_stats["IQR"] = global_stats["q3"] - global_stats["q1"]
global_stats["skewness"] = skew(bg_1, bias=False)
global_stats["kurtosis"] = kurtosis(bg_1, bias=False)


# In[22]:


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

print("Global Summary:")
display(pd.DataFrame([global_stats]).style.hide(axis="index"))

pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
pd.reset_option("display.max_colwidth")
pd.reset_option("display.width")


# In[23]:


# Group the data by participant and aggregate target statistics for each group
group_stats = df_train_preprocessed.groupby("p_num")["bg+1:00"].agg([
    ("mean", "mean"),
    ("median", "median"),
    ("std_dev", "std"),
    ("min", "min"),
    ("max", "max"),
    ("sum", "sum"),
    ("count", "count"),
    ("q1", lambda x: x.quantile(0.25, interpolation='nearest')),
    ("q3", lambda x: x.quantile(0.75, interpolation='nearest')),
])
group_stats["IQR"] = group_stats["q3"] - group_stats["q1"]


# In[24]:


# Define a function to compute skewness and kurtosis for a given participant group
def compute_skew_kurt(group):
    return pd.Series({
        "skewness": skew(group["bg+1:00"], bias=False),
        "kurtosis": kurtosis(group["bg+1:00"], bias=False)
    })

# Apply the function to each participant group and merge with other summary statistics
skew_kurt = df_train_preprocessed.groupby("p_num").apply(compute_skew_kurt).reset_index()


# In[25]:


# Merge group statistics with skewness and kurtosis, then sort by participant number
final_stats = group_stats.reset_index().merge(skew_kurt, on="p_num")
final_stats = final_stats.sort_values("p_num")


# In[26]:


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

print("P_NUM Summary:")
display(final_stats.style.hide(axis="index"))

pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
pd.reset_option("display.max_colwidth")
pd.reset_option("display.width")


# ### Visual Analysis

# In[27]:


# Create a DataFrame with only the participant number and the target value for plotting
df_train_preprocessed_pd = df_train_preprocessed[["p_num", "bg+1:00"]]
stats_pd = final_stats

# Get a sorted list of participants to iterate over in plots
participants = sorted(df_train_preprocessed_pd["p_num"].unique())
n = len(participants)

# Set Seaborn visual style
sns.set(style="whitegrid", palette="muted")


# #### Barplot: Mean of BG+1:00/P_NUM

# In[28]:


plt.figure(figsize=(15, 10))
sns.barplot(data=stats_pd, x="p_num", y="mean")
plt.title("Barplot Mean of BG+1:00/P_NUM")
plt.xlabel("P_NUM")
plt.ylabel("Mean of BG+1:00")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Barplot: Interquartile Range (IQR) of BG+1:00/P_NUM

# In[29]:


plt.figure(figsize=(15, 10))
sns.barplot(data=stats_pd, x="p_num", y="IQR")
plt.title("Barplot Interquartile Range (IQR) of BG+1:00/P_NUM")
plt.xlabel("P_NUM")
plt.ylabel("IQR of BG+1:00")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Barplot: Standard Deviation (SD) of BG+1:00/P_NUM

# In[30]:


plt.figure(figsize=(15, 10))
sns.barplot(data=stats_pd, x="p_num", y="std_dev")
plt.title("Barplot of Standard Deviation (SD) of BG+1:00/P_NUM")
plt.xlabel("P_NUM")
plt.ylabel("SD of BG+1:00")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Scatterplot: Skewness vs. Kurtosis of BG+1:00/P_NUM

# In[31]:


plt.figure(figsize=(15, 10))
sns.scatterplot(data=stats_pd, x="skewness", y="kurtosis", hue="p_num", s=100)
plt.title("Scatterplot of Skewness vs. Kurtosis of BG+1:00/P_NUM")
plt.xlabel("Skewness")
plt.ylabel("Kurtosis")
plt.legend(title="P_NUM")
plt.tight_layout()
plt.show()


# #### Boxplot: BG+1:00/P_NUM

# In[32]:


plt.figure(figsize=(15, 10))
sns.boxplot(data=df_train_preprocessed_pd, x="p_num", y="bg+1:00")
plt.title("Boxplot of BG+1:00/P_NUM")
plt.ylabel("BG+1:00")
plt.xlabel("P_NUM")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Histogram: BG+1:00/P_NUM

# In[33]:


# Determine grid layout for subplots based on the number of participants
ncols = 2
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5), constrained_layout=True)

axes = axes.flatten()

# Plot a histogram with KDE for each participant's target variable
for idx, pid in enumerate(participants):
    subset = df_train_preprocessed_pd[df_train_preprocessed_pd["p_num"] == pid]
    ax = axes[idx]
    sns.histplot(subset["bg+1:00"], bins=50, kde=True, ax=ax)
    ax.set_title(f"P_NUM: {pid}")
    ax.set_xlabel("BG+1:00")
    ax.set_ylabel("Count")

# Hide any unused subplots in the grid
for ax in axes[len(participants):]:
    ax.set_visible(False)

plt.suptitle("Histogram of BG+1:00/P_NUM", fontsize=16, y=1.02)
plt.show()

