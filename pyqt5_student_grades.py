import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QTextEdit, QFileDialog, 
                             QMessageBox, QWidget, QGroupBox, QScrollArea)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class StudentGradePredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize data and model
        self.df = pd.DataFrame()
        self.model = RandomForestClassifier(random_state=42)
        
        # Set up the UI
        self.setWindowTitle("Student Predictive Grades")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Data loading section
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        data_layout.addWidget(self.load_button)
        
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)
        
        # Model configuration section
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        features_layout = QVBoxLayout()
        features_label = QLabel("Features (comma-separated):")
        self.features_entry = QLineEdit()
        features_layout.addWidget(features_label)
        features_layout.addWidget(self.features_entry)
        model_layout.addLayout(features_layout)
        
        target_layout = QVBoxLayout()
        target_label = QLabel("Target:")
        self.target_entry = QLineEdit()
        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_entry)
        model_layout.addLayout(target_layout)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Buttons section
        buttons_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        buttons_layout.addWidget(self.train_button)
        
        self.predict_button = QPushButton("Make Predictions")
        self.predict_button.clicked.connect(self.make_predictions)
        buttons_layout.addWidget(self.predict_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        results_layout.addWidget(self.result_text)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # Add welcome message
        self.result_text.setText("Welcome to Student Predictive Grades Tool!\n\n"
                                "Steps to use:\n"
                                "1. Click 'Load Dataset' to import your data\n"
                                "2. Enter feature column names (comma-separated)\n"
                                "3. Enter target column name\n"
                                "4. Click 'Train Model' to build predictive model\n"
                                "5. Click 'Make Predictions' to see results")
    
    def load_dataset(self):
        """Load dataset from CSV or Excel file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Dataset", 
            "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)", 
            options=options
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                else:
                    self.df = pd.read_excel(file_path, engine='openpyxl')
                
                # Verify the dataframe is not empty
                if self.df.empty:
                    QMessageBox.critical(self, "Error", "The selected file contains no data")
                    return
                
                # Display info about the loaded dataset
                info_text = f"Dataset loaded successfully!\n\nShape: {self.df.shape}\nColumns: {', '.join(self.df.columns)}"
                QMessageBox.information(self, "Success", info_text)
                
                # Update the result text with dataset preview
                self.result_text.clear()
                self.result_text.append("Dataset Preview:\n")
                self.result_text.append(self.df.head().to_string())
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def train_model(self):
        """Train the RandomForest model"""
        # Get feature and target inputs
        features_input = self.features_entry.text().strip()
        target_input = self.target_entry.text().strip()
        
        # Validate inputs
        if not features_input or not target_input:
            QMessageBox.critical(self, "Error", "Please enter both features and target")
            return
        
        # Parse features
        features = [f.strip() for f in features_input.split(',')]
        target = target_input.strip()
        
        # Validate dataset
        if self.df.empty:
            QMessageBox.critical(self, "Error", "Please load a dataset first")
            return
        
        # Validate features and target exist in dataset
        missing_cols = [col for col in features + [target] if col not in self.df.columns]
        if missing_cols:
            QMessageBox.critical(self, "Error", f"Columns not found in dataset: {', '.join(missing_cols)}")
            return
        
        try:
            # Prepare data
            X = self.df[features]
            y = self.df[target]
            
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
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            self.result_text.clear()
            self.result_text.append(f"Model Training Results:\n\n")
            self.result_text.append(f"Accuracy: {accuracy:.4f}\n\n")
            
            # Show feature importance
            feature_importance = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            self.result_text.append("Feature Importance:\n")
            self.result_text.append(importance_df.to_string(index=False))
            
            QMessageBox.information(self, "Model Trained", f"Model trained successfully!\nAccuracy: {accuracy:.4f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def make_predictions(self):
        """Make predictions using the trained model"""
        # Validate model exists
        if not hasattr(self.model, 'predict') or not hasattr(self.model, 'feature_importances_'):
            QMessageBox.critical(self, "Error", "Please train a model first")
            return
        
        # Get features
        features_input = self.features_entry.text().strip()
        if not features_input:
            QMessageBox.critical(self, "Error", "Please enter features")
            return
        
        features = [f.strip() for f in features_input.split(',')]
        
        try:
            # Prepare data
            X_new = self.df[features]
            
            # Handle categorical features if needed
            categorical_cols = X_new.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    le = LabelEncoder()
                    X_new[col] = le.fit_transform(X_new[col])
            
            # Make predictions
            predictions = self.model.predict(X_new)
            
            # Display results
            self.result_text.clear()
            self.result_text.append("Prediction Results:\n\n")
            
            # Create a dataframe with original data and predictions
            result_df = self.df.copy()
            result_df['Predicted'] = predictions
            
            self.result_text.append(result_df.head(20).to_string())
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make predictions: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StudentGradePredictor()
    window.show()
    sys.exit(app.exec_())