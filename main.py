
import tkinter as tk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from tkinter import messagebox
from tkinter import ttk
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")
data = data.drop('condition', axis=1)
X = data.drop('Condition Label', axis=1)  # Features
y = data['Condition Label']  # Target

X = pd.get_dummies(X)
# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Initialize the Decision Tree Classifier
clf = tree.DecisionTreeClassifier()

# Step 6: Train the model using the training sets
clf.fit(X_train, y_train)

# Step 7: Make predictions on the testing set
y_pred = clf.predict(X_test)

# Step 8: Check the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 9: To see the decision tree
tree.plot_tree(clf)
# plt.savefig('tree.png')

# Step 10: Function to predict stress level based on humidity, temperature, and step count
def predict_stress_level(MEAN, MAX, MIN, RANGE, KURT, SKEW, MEAN_1ST_GRAD, STD_1ST_GRAD, MEAN_2ND_GRAD, STD_2ND_GRAD, ALSC, INSC, APSC, RMSC, MIN_PEAKS, MAX_PEAKS, STD_PEAKS, MEAN_PEAKS, MIN_ONSET, MAX_ONSET, STD_ONSET, MEAN_ONSET, subject_id, MEAN_LOG, INSC_LOG, APSC_LOG, RMSC_LOG, RANGE_LOG, ALSC_LOG, MIN_LOG, MEAN_1ST_GRAD_LOG, MEAN_2ND_GRAD_LOG, MIN_LOG_LOG, MEAN_1ST_GRAD_LOG_LOG, MEAN_2ND_GRAD_LOG_LOG, APSC_LOG_LOG, ALSC_LOG_LOG, APSC_BOXCOX, RMSC_BOXCOX, RANGE_BOXCOX, MEAN_YEO_JONSON, SKEW_YEO_JONSON, KURT_YEO_JONSON, APSC_YEO_JONSON, MIN_YEO_JONSON, MAX_YEO_JONSON, MEAN_1ST_GRAD_YEO_JONSON, RMSC_YEO_JONSON, STD_1ST_GRAD_YEO_JONSON, RANGE_SQRT, RMSC_SQUARED, MEAN_2ND_GRAD_CUBE, INSC_APSC, NasaTLX_class, NasaTLX_Label):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'MEAN': [MEAN],
        'MAX': [MAX],
        'MIN': [MIN],
        'RANGE': [RANGE],
        'KURT': [KURT],
        'SKEW': [SKEW],
        'MEAN_1ST_GRAD': [MEAN_1ST_GRAD],
        'STD_1ST_GRAD': [STD_1ST_GRAD],
        'MEAN_2ND_GRAD': [MEAN_2ND_GRAD],
        'STD_2ND_GRAD': [STD_2ND_GRAD],
        'ALSC': [ALSC],
        'INSC': [INSC],
        'APSC': [APSC],
        'RMSC': [RMSC],
        'MIN_PEAKS': [MIN_PEAKS],
        'MAX_PEAKS': [MAX_PEAKS],
        'STD_PEAKS': [STD_PEAKS],
        'MEAN_PEAKS': [MEAN_PEAKS],
        'MIN_ONSET': [MIN_ONSET],
        'MAX_ONSET': [MAX_ONSET],
        'STD_ONSET': [STD_ONSET],
        'MEAN_ONSET': [MEAN_ONSET],
        'subject_id': [subject_id],
        'MEAN_LOG': [MEAN_LOG],
        'INSC_LOG': [INSC_LOG],
        'APSC_LOG': [APSC_LOG],
        'RMSC_LOG': [RMSC_LOG],
        'RANGE_LOG': [RANGE_LOG],
        'ALSC_LOG': [ALSC_LOG],
        'MIN_LOG': [MIN_LOG],
        'MEAN_1ST_GRAD_LOG': [MEAN_1ST_GRAD_LOG],
        'MEAN_2ND_GRAD_LOG': [MEAN_2ND_GRAD_LOG],
        'MIN_LOG_LOG': [MIN_LOG_LOG],
        'MEAN_1ST_GRAD_LOG_LOG': [MEAN_1ST_GRAD_LOG_LOG],
        'MEAN_2ND_GRAD_LOG_LOG': [MEAN_2ND_GRAD_LOG_LOG],
        'APSC_LOG_LOG': [APSC_LOG_LOG],
        'ALSC_LOG_LOG': [ALSC_LOG_LOG],
        'APSC_BOXCOX': [APSC_BOXCOX],
        'RMSC_BOXCOX': [RMSC_BOXCOX],
        'RANGE_BOXCOX': [RANGE_BOXCOX],
        'MEAN_YEO_JONSON': [MEAN_YEO_JONSON],
        'SKEW_YEO_JONSON': [SKEW_YEO_JONSON],
        'KURT_YEO_JONSON': [KURT_YEO_JONSON],
        'APSC_YEO_JONSON': [APSC_YEO_JONSON],
        'MIN_YEO_JONSON': [MIN_YEO_JONSON],
        'MAX_YEO_JONSON': [MAX_YEO_JONSON],
        'MEAN_1ST_GRAD_YEO_JONSON': [MEAN_1ST_GRAD_YEO_JONSON],
        'RMSC_YEO_JONSON': [RMSC_YEO_JONSON],
        'STD_1ST_GRAD_YEO_JONSON': [STD_1ST_GRAD_YEO_JONSON],
        'RANGE_SQRT': [RANGE_SQRT],
        'RMSC_SQUARED': [RMSC_SQUARED],
        'MEAN_2ND_GRAD_CUBE': [MEAN_2ND_GRAD_CUBE],
        'INSC_APSC': [INSC_APSC],
        'NasaTLX class': [NasaTLX_class],
        'NasaTLX Label': [NasaTLX_Label]
    })

    # Predict the stress level using your trained model
    predicted_stress_level = clf.predict(input_data)

    return predicted_stress_level



# Test the function
print("Accuracy:", accuracy_score(y_test, y_pred))
# print(predict_stress_level(16.87,85.87,50))
def predict():
    humidity = float(humidity_entry.get())
    temperature = float(temperature_entry.get())
    step_count = int(step_count_entry.get())
    stress_level = predict_stress_level(humidity, temperature, step_count)
    actual_stress_level = y_test.iloc[0]  # replace this with the actual stress level
    comparison_table.insert("", "end", values=(humidity, temperature, step_count, stress_level[0], actual_stress_level))


root = tk.Tk()
root.title("Stress Level Predictor")

# Create entry fields for the new columns
columns = ["MEAN", "MAX", "MIN", "RANGE", "KURT", "SKEW", "MEAN_1ST_GRAD", "STD_1ST_GRAD", "MEAN_2ND_GRAD", "STD_2ND_GRAD", "ALSC", "INSC", "APSC", "RMSC", "MIN_PEAKS", "MAX_PEAKS", "STD_PEAKS", "MEAN_PEAKS", "MIN_ONSET", "MAX_ONSET", "STD_ONSET", "MEAN_ONSET", "subject_id", "MEAN_LOG", "INSC_LOG", "APSC_LOG", "RMSC_LOG", "RANGE_LOG", "ALSC_LOG", "MIN_LOG", "MEAN_1ST_GRAD_LOG", "MEAN_2ND_GRAD_LOG", "MIN_LOG_LOG", "MEAN_1ST_GRAD_LOG_LOG", "MEAN_2ND_GRAD_LOG_LOG", "APSC_LOG_LOG", "ALSC_LOG_LOG", "APSC_BOXCOX", "RMSC_BOXCOX", "RANGE_BOXCOX", "MEAN_YEO_JONSON", "SKEW_YEO_JONSON", "KURT_YEO_JONSON", "APSC_YEO_JONSON", "MIN_YEO_JONSON", "MAX_YEO_JONSON", "MEAN_1ST_GRAD_YEO_JONSON", "RMSC_YEO_JONSON", "STD_1ST_GRAD_YEO_JONSON", "RANGE_SQRT", "RMSC_SQUARED", "MEAN_2ND_GRAD_CUBE", "INSC_APSC", "NasaTLX class", "NasaTLX Label"]

'''
entries = {}
for i, column in enumerate(columns):
    tk.Label(root, text=column).grid(row=i)
    entries[column] = tk.Entry(root)
    entries[column].grid(row=i, column=1)

# Create a button that will call the predict function when clicked
predict_button = tk.Button(root, text="Predict Stress Level", command=predict)
predict_button.grid(row=len(columns), column=0, columnspan=2)

'''

# Create a table to display the comparison of predicted and actual stress levels
comparison_table = ttk.Treeview(root, columns=columns + ["Predicted", "Actual"], show="headings")

for column in columns[:4]:
    comparison_table.heading(column, text=column)
comparison_table.heading("Predicted", text="Predicted")
comparison_table.heading("Actual", text="Actual")
comparison_table.grid(row=len(columns)+1, column=0, columnspan=2)
# comparison_table.grid(row=1, column=0, columnspan=2)

def print_all_predictions():
    # Make predictions on the entire test set
    y_pred_all = clf.predict(X_test)

    # Insert each test case prediction and actual value into the comparison table
    for i in range(len(y_test)):
        # Determine the color based on the correctness of the prediction
        color = 'green' if y_pred_all[i] == y_test.iloc[i] else 'red'
        
        # Insert the row with the determined color
        values = [X_test.iloc[i][column] for column in columns[:4]]  # Only include the last columns
        comparison_table.insert("", "end", values=values + [y_pred_all[i], y_test.iloc[i]], tags=('correct' if color == 'green' else 'incorrect',))

    # Configure the tag to change the background color
    comparison_table.tag_configure('correct', background='lime')
    comparison_table.tag_configure('incorrect', background='red')


# Create a button that will call the print_all_predictions function when clicked
print_all_button = tk.Button(root, text="Print All Predictions", command=print_all_predictions)
print_all_predictions()
print_all_button.grid(row=len(columns)+2, column=0, columnspan=2)





# Run the event loop
root.mainloop()
