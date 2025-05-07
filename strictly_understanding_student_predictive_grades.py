import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Add global variables df & model
df = pd.DataFrame()
model = RandomForestClassifier(random_state=42)  # Initialize with parameters

# Function to load dataset from file
def load_dataset():
    global df
    # Use the existing root window instead of creating a new Tk instance
    file_path = filedialog.askopenfilename(
        parent=root,  # Use the existing root
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")]
    )
    
    # Check if a file was actually selected (not empty string)
    if file_path and len(file_path) > 0:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            
            # Verify the dataframe is not empty
            if df.empty:
                messagebox.showerror("Error", "The selected file contains no data")
                return None
            
            # Display info about the loaded dataset
            info_text = f"Dataset loaded successfully!\n\nShape: {df.shape}\nColumns: {', '.join(df.columns)}"
            messagebox.showinfo("Success", info_text)
            
            # Update the result text widget with dataset preview
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "Dataset Preview:\n\n")
            result_text.insert(tk.END, df.head().to_string())
            
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
            return None
    else:
        # User canceled the file selection
        print("File selection canceled")
        return None

# Function to train the model with selected features and target
def train_model():
    global df, model
    
    # Get feature and target inputs
    features_input = features_entry.get().strip()
    target_input = target_entry.get().strip()
    
    # Validate inputs
    if not features_input or not target_input:
        messagebox.showerror("Error", "Please enter both features and target")
        return None
    
    # Parse features
    features = [f.strip() for f in features_input.split(',')]
    target = target_input.strip()
    
    # Validate dataset
    if df.empty:
        messagebox.showerror("Error", "Please load a dataset first")
        return None
    
    # Validate features and target exist in dataset
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        messagebox.showerror("Error", f"Columns not found in dataset: {', '.join(missing_cols)}")
        return None
    
    try:
        # Prepare data
        X = df[features]
        y = df[target]
        
        # Handle categorical features if needed
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
        
        # Handle categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display results
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Model Training Results:\n\n")
        result_text.insert(tk.END, f"Accuracy: {accuracy:.4f}\n\n")
        
        # Show feature importance
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        result_text.insert(tk.END, "Feature Importance:\n")
        result_text.insert(tk.END, importance_df.to_string(index=False))
        
        messagebox.showinfo("Model Trained", f"Model trained successfully!\nAccuracy: {accuracy:.4f}")
        return model
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {str(e)}")
        import traceback
        traceback.print_exc()  # Print detailed error to console
        return None

# Function to make predictions using the trained model
def make_predictions():
    global df, model
    
    # Validate model exists
    if not hasattr(model, 'predict'):
        messagebox.showerror("Error", "Please train a model first")
        return
    
    # Get features
    features_input = features_entry.get().strip()
    if not features_input:
        messagebox.showerror("Error", "Please enter features")
        return
    
    features = [f.strip() for f in features_input.split(',')]
    
    try:
        # Prepare data
        X_new = df[features]
        
        # Handle categorical features if needed
        categorical_cols = X_new.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                le = LabelEncoder()
                X_new[col] = le.fit_transform(X_new[col])
        
        # Make predictions
        predictions = model.predict(X_new)
        
        # Display results
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Prediction Results:\n\n")
        
        # Create a dataframe with original data and predictions
        result_df = df.copy()
        result_df['Predicted'] = predictions
        
        result_text.insert(tk.END, result_df.head(20).to_string())
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {str(e)}")
        import traceback
        traceback.print_exc()  # Print detailed error to console

# Create the main application window
root = tk.Tk()
root.title("Student Predictive Grades Tool")
root.geometry("800x700")  # Set window size

# Create a frame for controls
control_frame = tk.Frame(root, padx=10, pady=10)
control_frame.pack(fill=tk.X)

# Data loading section
load_frame = tk.LabelFrame(control_frame, text="Data Management", padx=5, pady=5)
load_frame.pack(fill=tk.X, pady=5)

load_button = tk.Button(load_frame, text="Load Dataset", command=load_dataset, width=15)
load_button.pack(pady=5)

# Model configuration section
model_frame = tk.LabelFrame(control_frame, text="Model Configuration", padx=5, pady=5)
model_frame.pack(fill=tk.X, pady=5)

tk.Label(model_frame, text="Features (comma-separated):").pack(anchor=tk.W)
features_entry = tk.Entry(model_frame, width=60)
features_entry.pack(fill=tk.X, pady=2)

tk.Label(model_frame, text="Target:").pack(anchor=tk.W)
target_entry = tk.Entry(model_frame, width=60)
target_entry.pack(fill=tk.X, pady=2)

# Model actions section
action_frame = tk.Frame(control_frame)
action_frame.pack(fill=tk.X, pady=5)

train_button = tk.Button(action_frame, text="Train Model", command=train_model, width=15)
train_button.pack(side=tk.LEFT, padx=5)

predict_button = tk.Button(action_frame, text="Make Predictions", command=make_predictions, width=15)
predict_button.pack(side=tk.LEFT, padx=5)

# Results section
result_frame = tk.LabelFrame(root, text="Results", padx=5, pady=5)
result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

result_text = tk.Text(result_frame, height=20, width=80)
result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Add scrollbar to results
scrollbar = tk.Scrollbar(result_text)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
result_text.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=result_text.yview)

# Initial message
result_text.insert(tk.END, "Welcome to Student Predictive Grades Tool!\n\n")
result_text.insert(tk.END, "Steps to use:\n")
result_text.insert(tk.END, "1. Click 'Load Dataset' to import your data\n")
result_text.insert(tk.END, "2. Enter feature column names (comma-separated)\n")
result_text.insert(tk.END, "3. Enter target column name\n")
result_text.insert(tk.END, "4. Click 'Train Model' to build predictive model\n")
result_text.insert(tk.END, "5. Click 'Make Predictions' to see results\n")

# Start the application
root.mainloop()