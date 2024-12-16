#importing libaries and packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
import itertools
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tkinter import *
from tkinter import ttk	
from tkinter import PhotoImage
from tkinter import Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import font_manager

root = Tk()
root.title("HIGH PERFORMANCE CONCRETE STRENGTH PREDICTOR")
root.config(bg="#191614")


image = PhotoImage(file="neon.png")
label = Label(root, image=image,anchor='center')
label.grid(row=0, column=0, columnspan = 3, pady = 40)


cement = Entry(root, width=15, font = ("Menlo",18), fg = "#7DC5ED")
cement.grid(row = 1, column=1, columnspan=1)
blast_furnace = Entry(root, width=15, font = (("Menlo"),18), fg = "#7DC5ED")
blast_furnace.grid(row = 2, column=1, columnspan=1)
fly_ash = Entry(root, width=15, font = (("Menlo"),18), fg = "#7DC5ED")
fly_ash.grid(row = 3, column=1, columnspan=1)
water = Entry(root, width=15, font = (("Menlo"),18), fg = "#7DC5ED")
water.grid(row = 4, column=1, columnspan=1)
superplast = Entry(root, width=15, font = (("Menlo"),18), fg = "#7DC5ED")
superplast.grid(row = 5, column=1, columnspan=1)
coarse_agg = Entry(root, width=15, font = (("Menlo"),18), fg = "#7DC5ED")
coarse_agg.grid(row = 6, column=1, columnspan=1)
fine_agg = Entry(root, width=15, font = (("Menlo"),18), fg = "#7DC5ED")
fine_agg.grid(row = 7, column=1, columnspan=1)
age = Entry(root, width=15, font = (("Menlo"),18), fg = "#7DC5ED")
age.grid(row = 8, column=1, columnspan=1)

cement_label = Label(text = "Cement (kg in a m^3 mixture)", font = (("Menlo"),18), bg = "#191614", fg = "#7DC5ED")
cement_label.grid(row = 1, column=0, columnspan=1, stick = W, padx = 40)
blast_furnace_label = Label(text = "Blast Furnace Slag (kg in a m^3 mixture)", font = (("Menlo"),18), bg = "#191614", fg = "#7DC5ED")
blast_furnace_label.grid(row = 2, column=0, columnspan=1, stick = W, padx = 40)
fly_ash_label = Label(text = "Fly Ash (kg in a m^3 mixture)", font = (("Menlo"),18), bg = "#191614", fg = "#7DC5ED")
fly_ash_label.grid(row = 3, column=0, columnspan=1, stick = W, padx = 40)
water_label = Label(text = "Water (kg in a m^3 mixture)", font = (("Menlo"),18), bg = "#191614", fg = "#7DC5ED")
water_label.grid(row = 4, column=0, columnspan=1, stick = W, padx = 40)
superplast_label = Label(text = "Superplasticizer (kg in a m^3 mixture)", font = (("Menlo"),18), bg = "#191614", fg = "#7DC5ED")
superplast_label.grid(row = 5, column=0, columnspan=1, stick = W, padx = 40)
coarse_agg_label = Label(text = "Coarse Aggregate (kg in a m^3 mixture)", font = (("Menlo"),18), bg = "#191614", fg = "#7DC5ED")
coarse_agg_label.grid(row = 6, column=0, columnspan=1, stick = W, padx = 40)
fine_agg_label = Label(text = "Fine Aggregate (kg in a m^3 mixture)", font = (("Menlo"),18), bg = "#191614", fg = "#7DC5ED")
fine_agg_label.grid(row = 7, column=0, columnspan=1, stick = W, padx = 40)
age_label = Label(text = "Age (day)", font = (("Menlo"),18), bg = "#191614", fg = "#7DC5ED")
age_label.grid(row = 8, column=0, columnspan=1, stick = W, padx = 40)

result_label = Label(root, text="", font = (("Menlo"),22), bg = "#191614", fg = "#7DC5ED")
result_label.grid(row=10, column=0, columnspan=3)

cement.insert(0, "0")
blast_furnace.insert(0, "0")
fly_ash.insert(0, "0")
water.insert(0, "0")
superplast.insert(0, "0")
coarse_agg.insert(0, "0")
fine_agg.insert(0, "0")
age.insert(0,"0")

###
# Data preprocessing 
###
#ingesting data from the xls file
df = pd.read_excel("Concrete_Data.xls")
#filtering out the labels
df_without_label = df.iloc[:, :-1]
#https://scikit-learn.org/1.5/modules/preprocessing.html
df_label = df.iloc[:, -1]
label = df_label.to_numpy()

preprocessor = ColumnTransformer(
    transformers=[
        ('scalingAndVectorizer', StandardScaler(), df_without_label.columns.tolist())
    ]
)

###
# Training data on the best model found 
###
transformed_data = preprocessor.fit_transform(df_without_label)
X_train, X_test, y_train, y_test = train_test_split(transformed_data, label, test_size=0.3, random_state=42)
#https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.train_test_split.html

reg = GradientBoostingRegressor(random_state=0, learning_rate=0.31, n_estimators=500)
reg.fit(X_train, y_train)


#Defining function to preprocess training data and train on the Gradient Boost model
def predict_GBR(input):
    test = np.array(input)
    test_df = pd.DataFrame([test], columns=df_without_label.columns)
    test_scaled = preprocessor.transform(test_df)
    prediction = reg.predict(test_scaled)
    return prediction

#Defning function to plot the curve for the prediction of the Compressive Strength 
def create_plot(x, y):
    x = np.array(x)
    y = np.array(y)
    
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.plot(x, y)
    #https://stackoverflow.com/questions/72542210/how-to-plot-a-graph-onto-a-tkinter-canvas

    #Titles and colors are added to the different axis
    ax.set_title('COMPRESSIVE STRENGTH OVER TIME, BASED ON ML MODEL', color = "#7DC5ED")
    ax.set_xlabel('DAYS', color = "#7DC5ED")
    ax.set_ylabel('COMPRESSIVE STRENGTH MPa', color = '#7DC5ED')
    ax.set_facecolor("#191614")
    ax.spines['left'].set_color('#7DC5ED')
    ax.spines['right'].set_color('#7DC5ED')
    ax.spines['top'].set_color('#7DC5ED')
    ax.spines['bottom'].set_color('#7DC5ED')
    fig.patch.set_facecolor('#191614')
    ax.tick_params(axis='x', colors='#7DC5ED')  
    ax.tick_params(axis='y', colors='#7DC5ED')  
    return fig


#defning call back function for returning prediction based on the input from the user 
def callback():
    #saving the input data from the user on entry cells
    val1 = cement.get()
    val2 = blast_furnace.get()
    val3 = fly_ash.get()
    val4 = water.get()
    val5 = superplast.get() 
    val6 = coarse_agg.get()
    val7 = fine_agg.get()
    val8 = age.get()
    #creating list that will turn the values into float type which will then be input as test set for the Gradient boost model
    inputs = [float(val1), float(val2), float(val3), float(val4), float(val5), float(val6), float(val7), float(val8)]
    #calling predict_GBR function earlier to get the prediction from the ML model
    result= predict_GBR([float(val1), float(val2), float(val3), float(val4), float(val5), float(val6), float(val7), float(val8)])
    #defining ranges for the days to predict
    days = [3, 7, 14, 28, 56, 128]
    predictions = []
    #looping over the days to get the prediction to plot the progress of the compressive strength.
    for day in days:
        inputs = [float(val1), float(val2), float(val3), float(val4), float(val5), float(val6), float(val7), float(day)]
        prediction= predict_GBR(inputs)
        predictions.append(prediction[0])

    #creating top level to make sure the plot is shown on new window
    top = Toplevel()
    top.geometry("800x800")
    #https://stackoverflow.com/questions/69038908/in-tkinter-how-to-create-a-new-window-when-a-button-is-clicked-on-the-previous
    #calling create_plot function to plot the graph for the future and past predictions of compressive strength
    plot = create_plot(days, predictions)
    #adding the plot to the new pop up window after pressing the predict button
    canvas = FigureCanvasTkAgg(plot, master=top) 
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, sticky="e")
    
    #changing the result_label to show the predicted value from the ML model
    result_label.config(text=f"Predicted Compressive Strength: {result[0]:.2f} MPa")
    

#styling the ttk Button
style = ttk.Style()
style.configure("TButton", background="#7DC5ED", foreground="#7DC5ED")
button = ttk.Button(root, 
                    text="Predict", 
                    style="TButton", 
                    width=15,
                    command=callback)
button.grid(row=9, column=1, columnspan=2, pady=10)

root.geometry("750x950")
root.mainloop()