import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#add funtion
model = RandomForestClassifier()
dataframe = pd.DataFrame()

# load the dataset to process
#performance_dataset_1 = pd.read_csv('/content/student_habits_performance_part_1_with_errors-1.csv')
# Make a dataframe
#df = pd.DataFrame(performance_dataset_1)

# Preprocessing Function - loads in the dataframe
def process_data(df):
    # This function will clean the csv files that are entered into the app

    # Fill Age with median value
    # However, first need to replace 'unknown' in the data with NaN (Pandas recognises as not a number)
    df['age'] = df['age'].replace('unknown', pd.NA)
    # Calculate median age and fill the unknown values (fill with median to account for outliers)
    # Ensure the column is numberic before calculating the median values
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age'].fillna(df['age'].median(), inplace=True)
    # For gender, the data will need to be encoded: {Female: 0, Male: 1, Other: 2}
    # Clean the gender column: remove whitespace and convert to lowercase
    df['gender'] = df['gender'].str.strip().str.lower()
    # Define the mapping dictionary with lowercase keys
    gender_mapping = {'female': 0, 'male': 1, 'other': 2}
    # Apply the mapping to the cleaned column.
    # The .map() function will automatically turn any values not in the dictionary (like the original NaNs) into NaN.
    df['gender'] = df['gender'].map(gender_mapping)

    # Fill the NaN values in the gender coluumn
    # Calculate the mode (most frequent value) of the gender column, excluding NaNs
    # .mode() can return multiple values if there's a tie, [0] takes the first one
    gender_mode = df['gender'].mode()[0]
    # Fill NaN values in the 'gender' column with the calculated mode
    df['gender'].fillna(gender_mode, inplace=True)
    # Check if there are any remaining NaNs in the gender column
    print(df['gender'].isnull().sum())
    # Print the unique values after mapping to see the result
    print(df['gender'].unique())

    # Study Hours Per Day: replace with mean
    # Replace the non numeric value 'varies'
    df['study_hours_per_day'] = df['study_hours_per_day'].str.strip().str.lower()
    print(df['study_hours_per_day'].unique())
    df['study_hours_per_day'] = df['study_hours_per_day'].replace('varies', pd.NA)
    df['study_hours_per_day'] = pd.to_numeric(df['study_hours_per_day'], errors='coerce')
    df['study_hours_per_day'].fillna(df['study_hours_per_day'].mean(), inplace=True)
    
    # Social Media Hours: replace with mean
    df['social_media_hours'] = pd.to_numeric(df['social_media_hours'], errors='coerce')
    df['social_media_hours'].fillna(df['social_media_hours'].mean(), inplace=True)    

    # Netflix Hours: replace with mean
    df['netflix_hours'] = pd.to_numeric(df['netflix_hours'], errors='coerce')
    df['netflix_hours'].fillna(df['netflix_hours'].mean(), inplace=True)
    
    # Part time job: replace with mode
    # Data will need to be encoded - {No: 0, Yes: 1}
    # I need to clean the part time job column
    df['part_time_job'] = df['part_time_job'].str.strip().str.lower()
    # Map the data points
    part_time_job_mapping = {'no': 0, 'yes': 1}
    df['part_time_job'] = df['part_time_job'].map(part_time_job_mapping)
    
    # Attendance Percentage: replace with mean
    df['attendance_percentage'] = pd.to_numeric(df['attendance_percentage'], errors='coerce')
    df['attendance_percentage'].fillna(df['attendance_percentage'].mean(), inplace=True)

    # Sleep Hours: replace with median to account for outliers
    df['sleep_hours'] = pd.to_numeric(df['sleep_hours'], errors='coerce')
    df['sleep_hours'].fillna(df['sleep_hours'].median(), inplace=True)

    # Diet Quality: replace with mode
    # Map diet quality - {poor: 0, fair: 1, good: 2}

    # Clean the strings
    df['diet_quality'] = df['diet_quality'].str.strip().str.lower()
    # Map to encode
    diet_quality_mapping = {'poor': 0, 'fair': 1, 'good': 2}
    df['diet_quality'] = df['diet_quality'].map(diet_quality_mapping)
    # Fill in missing values with mode
    diet_quality_mode = df['diet_quality'].mode()[0]
    df['diet_quality'].fillna(diet_quality_mode, inplace=True)

    # Excercise Frequency: replace with mode
    # Fill missing values with mode
    exercise_frequency_mode = df['exercise_frequency'].mode()[0]
    df['exercise_frequency'].fillna(exercise_frequency_mode, inplace=True)

    # Parental Education: replace with mode
    # Map the data to encode - {High School: 0, Bachelor: 1, Masters: 2}
    df['parental_education_level'] = df['parental_education_level'].str.strip().str.lower()
    # Map
    parental_education_level_mapping = {'high school': 0, 'bachelor': 1, 'masters': 2}
    df['parental_education_level'] = df['parental_education_level'].map(parental_education_level_mapping)
    # Fill missing values with mode
    parental_education_level_mode = df['parental_education_level'].mode()[0]
    df['parental_education_level'].fillna(parental_education_level_mode, inplace=True)

    # Internet Quality: replace with mode
    # Map data - {poor: 0, average: 1, good: 2}
    df['internet_quality'] = df['internet_quality'].str.strip().str.lower()
    # Map
    internet_quality_mapping = {'poor': 0, 'average': 1, 'good': 2}
    df['internet_quality'] = df['internet_quality'].map(internet_quality_mapping)
    # Fill missing data with mode
    internet_quality_mode = df['internet_quality'].mode()[0]
    df['internet_quality'].fillna(internet_quality_mode, inplace=True)

    # Mental Health Rating: replace with mode
    mental_health_rating_mode = df['mental_health_rating'].mode()[0]
    df['mental_health_rating'].fillna(mental_health_rating_mode, inplace=True)

    # Extracurricular Participation: replace with mode
    # map data - {no: 0, yes: 1}
    df['extracurricular_participation'] = df['extracurricular_participation'].str.strip().str.lower()
    # Map
    extracurricular_participation_mapping = {'no': 0, 'yes': 1}
    df['extracurricular_participation'] = df['extracurricular_participation'].map(extracurricular_participation_mapping)
    # Fill missing data with mode
    extracurricular_participation_mode = df['extracurricular_participation'].mode()[0]
    df['extracurricular_participation'].fillna(extracurricular_participation_mode, inplace=True)

    # Exam Score: replace with median to account for outliers
    df['exam_score'] = pd.to_numeric(df['exam_score'], errors='coerce')
    df['exam_score'].fillna(df['exam_score'].median(), inplace=True)

    # Return the modified dataset
    return df



# Function to load the dataset into tkinter app
def load_dataset():
    global dataframe
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            # call process_data
            df = process_data(df)
            dataframe = df
            
            messagebox.showinfo("Success", "Dataset loaded successfully *but did you check the script!")
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    return None

# New function needed to reliably set the dataframe 
def set_dataframe():
    global dataframe
    df = load_dataset()
    if df is not None:
        dataframe = df

# Trains the model using sklearn 
def train_model(df, features, target):
    global model
    try:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {accuracy:.2f}")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")
    return None

# Function to make a prediction using a dataset provided
def make_predictions(model, df, features):
    try:
        X_new = df[features]
        predictions = model.predict(X_new)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions:\n{predictions}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

# Set the root for tkinter GUI application
root = tk.Tk()
root.title("Student Predictive Grades")

# Making a button in tkinter to provide functionality to load the dataset in
load_button = tk.Button(root, text="Load Dataset", command=lambda: load_dataset())
load_button.pack(pady=10)

# Adds a text box to allow the user to predict a grade on the dataset features provided
tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root)
features_entry.pack(pady=5)

# Adds a textbox ..... 
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

# Adds a button to the tkinter app to allow for the execution of model training via a GUI
train_button = tk.Button(root, text="Train Model", command=lambda: train_model(df, features_entry.get().split(','), target_entry.get()))
train_button.pack(pady=10)

# Adds a button to the tkinter app to allow for the prediction of a grade via a GUI
predict_button = tk.Button(root, text="Make Predictions", command=lambda: make_predictions(model, df, features_entry.get().split(',')))
predict_button.pack(pady=10)

# Provides the result of the prediction 
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# The main loop allowing for the execution of the app and keeping it on screen
root.mainloop()

