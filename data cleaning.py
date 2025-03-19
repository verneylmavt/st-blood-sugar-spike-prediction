import pandas as pd
import numpy as np

# Load the dataset
train_df = pd.read_csv("train.csv")

# Identify column categories
bg_cols = [col for col in train_df.columns if "bg-" in col]
insulin_cols = [col for col in train_df.columns if "insulin-" in col]
carbs_cols = [col for col in train_df.columns if "carbs-" in col]
hr_cols = [col for col in train_df.columns if "hr-" in col]
steps_cols = [col for col in train_df.columns if "steps-" in col]
cals_cols = [col for col in train_df.columns if "cals-" in col]
activity_cols = [col for col in train_df.columns if "activity-" in col]

# Drop activity columns if they cause issues
train_df = train_df.drop(columns=activity_cols, errors='ignore')

# Handle missing values
train_df[bg_cols] = train_df[bg_cols].ffill()  # Forward-fill blood glucose
train_df[insulin_cols] = train_df[insulin_cols].fillna(0)  # Fill insulin with 0
train_df[carbs_cols] = train_df[carbs_cols].fillna(0)  # Fill carbs with 0
train_df[hr_cols] = train_df[hr_cols].fillna(train_df[hr_cols].mean())  # Fill HR with mean
train_df[steps_cols] = train_df[steps_cols].fillna(0)  # Fill steps with 0
train_df[cals_cols] = train_df[cals_cols].fillna(0)  # Fill calories with 0

# Convert time to datetime format and extract hour of the day
train_df['time'] = pd.to_datetime(train_df['time'], format='%H:%M:%S', errors='coerce')
train_df['hour_of_day'] = train_df['time'].dt.hour

print(f"Before dropna: {train_df.shape}")
train_df = train_df.dropna()
print(f"After dropna: {train_df.shape}")


# Print cleaned dataset
print(train_df.head())

print("\nActivity columns removed & NaN values handled! Data is now fully cleaned.")
print(f"Dataset shape: {train_df.shape}")
