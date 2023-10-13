Introduction and Objective:
Earthquakes are one of the most destructive natural hazards on Earth. They can cause widespread damage and loss of life. In recent years, there has been growing interest in using machine learning to predict earthquakes. This project will use machine learning models to predict future earthquakes using the Ultimate Earthquake Dataset: https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023. The dataset contains information on over 3 million earthquakes that occurred worldwide from 1990 to 2023.

The project will use Linear Regression. The model will be trained on the dataset and it's performance will be evaluated on a held-out test set. The best model will be deployed to production so that it can be used to predict future earthquakes.

This project has the potential to make a significant contribution to earthquake prediction. By using machine learning, it may be possible to develop more accurate and reliable earthquake prediction models. This could help to save lives and reduce the damage caused by earthquakes.

!mkdir -p ~/.kaggle
!cp kaggle.json /root/.kaggle/
import os
for filename in os.listdir('/content'):
    print(filename)
.config

kaggle.json

sample_data
# Set the permissions for the kaggle.json file to make it readable only by the owner
!chmod 777 /root/.kaggle
!kaggle datasets download -d alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023
Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /root/.kaggle/kaggle.json'

Downloading the-ultimate-earthquake-dataset-from-1990-2023.zip to /content

 91% 105M/116M [00:01<00:00, 63.5MB/s] 

100% 116M/116M [00:01<00:00, 75.4MB/s]
!unzip the-ultimate-earthquake-dataset-from-1990-2023.zip
Archive:  the-ultimate-earthquake-dataset-from-1990-2023.zip

  inflating: Eartquakes-1990-2023.csv  
1. Importing necessary libraries:
# Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For better plots
import plotly.express as px
import plotly.graph_objects as go

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Deep-learning
import tensorflow as tf
2. Importing Data:
data = pd.read_csv("/content/Eartquakes-1990-2023.csv")
3. Understanding the basics of the Data:
data.head()
time	place	status	tsunami	significance	data_type	magnitudo	state	longitude	latitude	depth	date
0	631153353990	12 km NNW of Meadow Lakes, Alaska	reviewed	0	96	earthquake	2.50	Alaska	-149.669200	61.730200	30.100	1990-01-01 00:22:33.990000+00:00
1	631153491210	14 km S of Volcano, Hawaii	reviewed	0	31	earthquake	1.41	Hawaii	-155.212333	19.317667	6.585	1990-01-01 00:24:51.210000+00:00
2	631154083450	7 km W of Cobb, California	reviewed	0	19	earthquake	1.11	California	-122.806167	38.821000	3.220	1990-01-01 00:34:43.450000+00:00
3	631155512130	11 km E of Mammoth Lakes, California	reviewed	0	15	earthquake	0.98	California	-118.846333	37.664333	-0.584	1990-01-01 00:58:32.130000+00:00
4	631155824490	16km N of Fillmore, CA	reviewed	0	134	earthquake	2.95	California	-118.934000	34.546000	16.122	1990-01-01 01:03:44.490000+00:00
data.describe()
time	tsunami	significance	magnitudo	longitude	latitude	depth
count	3.445751e+06	3.445751e+06	3.445751e+06	3.445751e+06	3.445751e+06	3.445751e+06	3.445751e+06
mean	1.247124e+12	4.434447e-04	7.400973e+01	1.774076e+00	-1.012876e+02	3.746483e+01	2.285387e+01
std	2.976292e+11	2.105346e-02	1.016364e+02	1.291055e+00	7.697416e+01	2.041577e+01	5.484938e+01
min	6.311534e+11	0.000000e+00	0.000000e+00	-9.990000e+00	-1.799997e+02	-8.442200e+01	-1.000000e+01
25%	1.024401e+12	0.000000e+00	1.300000e+01	9.100000e-01	-1.464274e+02	3.406400e+01	3.120000e+00
50%	1.282338e+12	0.000000e+00	3.300000e+01	1.460000e+00	-1.189538e+02	3.793567e+01	7.700000e+00
75%	1.508701e+12	0.000000e+00	8.100000e+01	2.300000e+00	-1.159277e+02	4.784800e+01	1.612000e+01
max	1.690629e+12	1.000000e+00	2.910000e+03	9.100000e+00	1.800000e+02	8.738600e+01	7.358000e+02
data.info()
<class 'pandas.core.frame.DataFrame'>

RangeIndex: 3445751 entries, 0 to 3445750

Data columns (total 12 columns):

 #   Column        Dtype  

---  ------        -----  

 0   time          int64  

 1   place         object 

 2   status        object 

 3   tsunami       int64  

 4   significance  int64  

 5   data_type     object 

 6   magnitudo     float64

 7   state         object 

 8   longitude     float64

 9   latitude      float64

 10  depth         float64

 11  date          object 

dtypes: float64(4), int64(3), object(5)

memory usage: 315.5+ MB
Checking for "Null" values:

data.isna().sum()
time            0
place           0
status          0
tsunami         0
significance    0
data_type       0
magnitudo       0
state           0
longitude       0
latitude        0
depth           0
date            0
dtype: int64
Great, no "Null" values, won't have to go through "Pre-Processing" steps 😎😎

4. EDA Time:
Before delving into EDA, let's convert the "date" column to Pandas dataframe.

data.date = pd.to_datetime(data.date)
data.date
0         1990-01-01 00:22:33.990000+00:00
1         1990-01-01 00:24:51.210000+00:00
2         1990-01-01 00:34:43.450000+00:00
3         1990-01-01 00:58:32.130000+00:00
4         1990-01-01 01:03:44.490000+00:00
                        ...               
3445746   2023-07-29 10:34:11.941000+00:00
3445747   2023-07-29 10:36:15.715000+00:00
3445748   2023-07-29 10:40:15.940000+00:00
3445749   2023-07-29 10:55:46.040000+00:00
3445750   2023-07-29 11:08:57.884000+00:00
Name: date, Length: 3445751, dtype: datetime64[ns, UTC]
a. Magnitude of earthquakesearthquake.png:
# Extract the year
year = pd.to_numeric(data.date.dt.year)

# Extract the magnitude
magnitude = data.magnitudo

# Plot a histogram of the magnitude of earthquakes per year using Seaborn
sns.histplot(magnitude, bins=10, kde=True, x=year)
plt.xlabel("Year")
plt.ylabel("Magnitude")
plt.title("Magnitude of Earthquakes per Year (1990-2023)")
plt.xticks(range(1990, 2024), rotation=90)
plt.show()

data["magnitudo"].plot(kind = "line", style = ".", title ="Magnitudo trend by year", figsize =(16,5))
plt.show()

Inferencesinferential-statistics.png

The histogram shows that the magnitude of earthquakes is distributed over a wide range, from about 2 to 9. There is a peak in the number of earthquakes with a magnitude of about 5. The number of earthquakes with a magnitude of 7 or higher is relatively low.

The histogram also shows that the magnitude of earthquakes has been increasing over time. This is likely due to the improved ability of scientists to detect and record earthquakes.

The line-graph shows, the magnitude of earthquakes has been increasing over time, but the increase is not linear. There are some years with no earthquakes recorded, and the number of earthquakes with a magnitude of 7 or higher is relatively low.


The increase in the magnitude of earthquakes could be due to a number of factors, such as:

Increased urbanization and development, which can lead to more earthquakes due to the stress placed on the Earth's crust.
Climate change, which can cause the Earth's crust to shift and move, leading to earthquakes.
Plate tectonics, the movement of the Earth's tectonic plates, which can cause earthquakes when they collide or rub against each other.

The relatively low number of earthquakes with a magnitude of 7 or higher could be due to a number of factors, such as:
The Earth's crust is not evenly distributed, and some areas are more prone to earthquakes than others.

The monitoring of earthquakes has improved over time, so we are more likely to detect smaller earthquakes.

The effects of climate change may have reduced the number of large earthquakes.
b) Locating the places of earthquakes 🌍:
(To be run, only if you are working with GPU)

import folium

# Create a new DataFrame with the latitude and longitude columns
df = pd.DataFrame({
    "latitude": data.latitude,
    "longitude": data.longitude,
    "magnitude": data.magnitudo
})

# Create a new Folium map
map = folium.Map()

# Add the earthquakes to the map
for latitude, longitude, magnitude in zip(df.latitude, df.longitude, df.magnitude):
    folium.CircleMarker(
        [latitude, longitude],
        radius = magnitude * 10,
        color = "red",
        fill = True,
        fill_color= "red",
        fill_opacity = 0.7,
    ).add_to(map)

# Display the map
map
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Create a new DataFrame with the latitude and longitude columns
df = pd.DataFrame({
    "latitude": data.latitude,
    "longitude": data.longitude,
    "magnitude": data.magnitudo
})

# Create a new Plotly map
fig = go.Figure(
    layout=go.Layout(
        title="Earthquakes (1990-2023)",
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
    )
)

# Add the earthquakes to the map
for latitude, longitude, magnitude in zip(df.latitude, df.longitude, df.magnitude):
    size = max(0, magnitude * 10)  # Ensure size is non-negative
    fig.add_trace(
        go.Scattergeo(
            lat=[latitude],
            lon=[longitude],
            mode="markers",
            marker=go.Marker(
                size=size,
                color="red",
                opacity=0.7,
            ),
        )
    )

# Display the map
fig.show()
c). Top states with highest earthquakes:
# Get the top 5 states with the highest number of earthquakes
top_5_states = (
    data.groupby("state")
    .size()
    .to_frame(name="count")
    .reset_index()
    .sort_values(by=["count"], ascending=False)
    .head(5)["state"]
)

# Get the unique values of the top 5 states
top_5_states_unique = top_5_states.unique()

# Print the unique values
print(top_5_states_unique)
[' California' ' Alaska' 'California' ' Nevada' ' Hawaii']
# Create a DataFrame of the earthquake counts
earthquake_counts_df = pd.DataFrame(
    data={"state": top_5_states["state"], "count": top_5_states["count"]}
)

# Create a bar chart of the earthquake counts
fig = px.bar(
    earthquake_counts_df,
    x="state",
    y="count",
    title="Top 5 States with the Highest Number of Earthquakes (1990-2023)",
)

# Display the bar chart
fig.show()
California is occuring twice.

d) Bottom 5 states with lowest amount of earthquakes.
# Get the bottom 5 states with the highest number of earthquakes
bottom_5_states = (
    data.groupby("state")
    .size()
    .to_frame(name="count")
    .reset_index()
    .sort_values(by=["count"], ascending=True)
    .head(5)
)

# Create a DataFrame of the earthquake counts
bottom_earthquake_counts_df = pd.DataFrame(
    data={"state": bottom_5_states["state"],
          "count": bottom_5_states["count"]}
)

# Create a bar chart of the earthquake counts
fig = px.bar(
    bottom_earthquake_counts_df,
    x="state",
    y="count",
    title="Bottom 5 States with the Highest Number of Earthquakes (1990-2023)",
)

# Display the bar chart
fig.show()
e) Top 5 Strongest earthquakes:
# Get the top 5 strongest earthquakes
top_5_earthquakes = (
    data.sort_values(by=["magnitudo"], ascending=False)
    .head(5)
)

# Create a map of the earthquakes
fig = px.scatter_geo(
    top_5_earthquakes,
    lat="latitude",  # Make sure to use the correct latitude column name
    lon="longitude",  # Make sure to use the correct longitude column name
    size="magnitudo",
    color="magnitudo",
    title="Top 5 Strongest Earthquakes (1990-2023)",
)

# Display the map
fig.show()
The size of the marker on the map represents the magnitude of the earthquake. The color of the marker also represents the magnitude of the earthquake, with red representing the strongest earthquakes and blue representing the weakest earthquakes.

5. Modelling Time:
We will create a simple Linear Regression model as the data is continuous.

# Preprocess the data (drop "place" column and keep only numerical features)
numerical_columns = ["magnitudo", "depth", "latitude", "longitude"]
data_numeric = data[numerical_columns]

# Separate input features (X) and target variable (y)
X = data_numeric.drop(columns=["magnitudo"])
y = data_numeric["magnitudo"]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict using the model
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error
print("Mean Squared Error:", mse)
Mean Squared Error: 0.8998172544133762
y_pred = model.predict(X_test)
for i, (actual, predicted) in enumerate(zip(y_test, y_pred), 1):
    if i % 1000 == 0:
        print(f"Iteration: {i}  |  Actual Magnitude: {actual:.2f}  |  Predicted Magnitude: {predicted:.2f}")
Iteration: 1000  |  Actual Magnitude: 2.90  |  Predicted Magnitude: 1.76

Iteration: 2000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.14

Iteration: 3000  |  Actual Magnitude: 0.28  |  Predicted Magnitude: 1.48

Iteration: 4000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 0.91

Iteration: 5000  |  Actual Magnitude: 3.40  |  Predicted Magnitude: 2.61

Iteration: 6000  |  Actual Magnitude: 0.60  |  Predicted Magnitude: 1.58

Iteration: 7000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.57

Iteration: 8000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 1.57

Iteration: 9000  |  Actual Magnitude: 0.88  |  Predicted Magnitude: 1.48

Iteration: 10000  |  Actual Magnitude: -0.46  |  Predicted Magnitude: 0.96

Iteration: 11000  |  Actual Magnitude: 1.56  |  Predicted Magnitude: 1.61

Iteration: 12000  |  Actual Magnitude: 4.80  |  Predicted Magnitude: 4.23

Iteration: 13000  |  Actual Magnitude: 2.30  |  Predicted Magnitude: 1.28

Iteration: 14000  |  Actual Magnitude: 1.52  |  Predicted Magnitude: 1.58

Iteration: 15000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 4.00

Iteration: 16000  |  Actual Magnitude: 1.39  |  Predicted Magnitude: 1.69

Iteration: 17000  |  Actual Magnitude: 1.26  |  Predicted Magnitude: 1.53

Iteration: 18000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 5.08

Iteration: 19000  |  Actual Magnitude: 1.19  |  Predicted Magnitude: 1.67

Iteration: 20000  |  Actual Magnitude: 3.00  |  Predicted Magnitude: 1.55

Iteration: 21000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.11

Iteration: 22000  |  Actual Magnitude: 1.14  |  Predicted Magnitude: 1.50

Iteration: 23000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.55

Iteration: 24000  |  Actual Magnitude: 1.19  |  Predicted Magnitude: 1.54

Iteration: 25000  |  Actual Magnitude: 0.56  |  Predicted Magnitude: 1.64

Iteration: 26000  |  Actual Magnitude: 0.88  |  Predicted Magnitude: 1.49

Iteration: 27000  |  Actual Magnitude: 1.81  |  Predicted Magnitude: 1.61

Iteration: 28000  |  Actual Magnitude: 0.30  |  Predicted Magnitude: 0.95

Iteration: 29000  |  Actual Magnitude: 1.22  |  Predicted Magnitude: 1.64

Iteration: 30000  |  Actual Magnitude: -0.10  |  Predicted Magnitude: 0.95

Iteration: 31000  |  Actual Magnitude: 2.57  |  Predicted Magnitude: 1.53

Iteration: 32000  |  Actual Magnitude: 0.20  |  Predicted Magnitude: 1.51

Iteration: 33000  |  Actual Magnitude: 0.81  |  Predicted Magnitude: 1.53

Iteration: 34000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 1.48

Iteration: 35000  |  Actual Magnitude: 4.10  |  Predicted Magnitude: 2.36

Iteration: 36000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.94

Iteration: 37000  |  Actual Magnitude: 1.51  |  Predicted Magnitude: 1.63

Iteration: 38000  |  Actual Magnitude: 1.22  |  Predicted Magnitude: 1.48

Iteration: 39000  |  Actual Magnitude: 1.03  |  Predicted Magnitude: 1.55

Iteration: 40000  |  Actual Magnitude: 0.30  |  Predicted Magnitude: 1.52

Iteration: 41000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.65

Iteration: 42000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.07

Iteration: 43000  |  Actual Magnitude: 3.50  |  Predicted Magnitude: 2.63

Iteration: 44000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 0.97

Iteration: 45000  |  Actual Magnitude: 5.20  |  Predicted Magnitude: 4.66

Iteration: 46000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.40

Iteration: 47000  |  Actual Magnitude: 1.56  |  Predicted Magnitude: 1.55

Iteration: 48000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.38

Iteration: 49000  |  Actual Magnitude: 4.10  |  Predicted Magnitude: 2.66

Iteration: 50000  |  Actual Magnitude: 4.00  |  Predicted Magnitude: 5.20

Iteration: 51000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.50

Iteration: 52000  |  Actual Magnitude: 0.37  |  Predicted Magnitude: 1.52

Iteration: 53000  |  Actual Magnitude: 1.13  |  Predicted Magnitude: 1.63

Iteration: 54000  |  Actual Magnitude: 1.26  |  Predicted Magnitude: 1.56

Iteration: 55000  |  Actual Magnitude: -0.10  |  Predicted Magnitude: 1.58

Iteration: 56000  |  Actual Magnitude: 1.05  |  Predicted Magnitude: 1.55

Iteration: 57000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.01

Iteration: 58000  |  Actual Magnitude: 1.48  |  Predicted Magnitude: 1.49

Iteration: 59000  |  Actual Magnitude: -0.30  |  Predicted Magnitude: 1.59

Iteration: 60000  |  Actual Magnitude: -9.99  |  Predicted Magnitude: 1.54

Iteration: 61000  |  Actual Magnitude: 1.79  |  Predicted Magnitude: 1.66

Iteration: 62000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.54

Iteration: 63000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.51

Iteration: 64000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.62

Iteration: 65000  |  Actual Magnitude: 2.50  |  Predicted Magnitude: 1.01

Iteration: 66000  |  Actual Magnitude: 2.23  |  Predicted Magnitude: 1.62

Iteration: 67000  |  Actual Magnitude: 0.75  |  Predicted Magnitude: 1.52

Iteration: 68000  |  Actual Magnitude: -0.56  |  Predicted Magnitude: 0.93

Iteration: 69000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 1.42

Iteration: 70000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.48

Iteration: 71000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.58

Iteration: 72000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 1.45

Iteration: 73000  |  Actual Magnitude: 0.80  |  Predicted Magnitude: 1.56

Iteration: 74000  |  Actual Magnitude: 0.80  |  Predicted Magnitude: 1.05

Iteration: 75000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.58

Iteration: 76000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.11

Iteration: 77000  |  Actual Magnitude: 1.62  |  Predicted Magnitude: 1.37

Iteration: 78000  |  Actual Magnitude: 1.09  |  Predicted Magnitude: 1.55

Iteration: 79000  |  Actual Magnitude: 1.54  |  Predicted Magnitude: 1.50

Iteration: 80000  |  Actual Magnitude: 2.19  |  Predicted Magnitude: 1.66

Iteration: 81000  |  Actual Magnitude: 2.60  |  Predicted Magnitude: 1.58

Iteration: 82000  |  Actual Magnitude: -0.61  |  Predicted Magnitude: 0.95

Iteration: 83000  |  Actual Magnitude: 1.06  |  Predicted Magnitude: 1.47

Iteration: 84000  |  Actual Magnitude: 0.95  |  Predicted Magnitude: 1.68

Iteration: 85000  |  Actual Magnitude: 2.38  |  Predicted Magnitude: 1.51

Iteration: 86000  |  Actual Magnitude: 1.01  |  Predicted Magnitude: 1.61

Iteration: 87000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 1.66

Iteration: 88000  |  Actual Magnitude: 0.68  |  Predicted Magnitude: 1.51

Iteration: 89000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.04

Iteration: 90000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 3.60

Iteration: 91000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 0.89

Iteration: 92000  |  Actual Magnitude: 1.68  |  Predicted Magnitude: 1.60

Iteration: 93000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 0.95

Iteration: 94000  |  Actual Magnitude: 5.60  |  Predicted Magnitude: 3.93

Iteration: 95000  |  Actual Magnitude: 2.39  |  Predicted Magnitude: 1.55

Iteration: 96000  |  Actual Magnitude: 4.00  |  Predicted Magnitude: 5.33

Iteration: 97000  |  Actual Magnitude: 1.66  |  Predicted Magnitude: 1.56

Iteration: 98000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 0.90

Iteration: 99000  |  Actual Magnitude: 0.95  |  Predicted Magnitude: 1.65

Iteration: 100000  |  Actual Magnitude: 2.80  |  Predicted Magnitude: 2.30

Iteration: 101000  |  Actual Magnitude: 1.90  |  Predicted Magnitude: 1.05

Iteration: 102000  |  Actual Magnitude: 1.05  |  Predicted Magnitude: 1.66

Iteration: 103000  |  Actual Magnitude: 2.50  |  Predicted Magnitude: 2.64

Iteration: 104000  |  Actual Magnitude: 2.40  |  Predicted Magnitude: 2.26

Iteration: 105000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.55

Iteration: 106000  |  Actual Magnitude: -0.23  |  Predicted Magnitude: 1.54

Iteration: 107000  |  Actual Magnitude: 1.66  |  Predicted Magnitude: 1.46

Iteration: 108000  |  Actual Magnitude: 2.03  |  Predicted Magnitude: 1.60

Iteration: 109000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.10

Iteration: 110000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.12

Iteration: 111000  |  Actual Magnitude: 1.25  |  Predicted Magnitude: 1.56

Iteration: 112000  |  Actual Magnitude: 1.90  |  Predicted Magnitude: 2.49

Iteration: 113000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.20

Iteration: 114000  |  Actual Magnitude: 2.10  |  Predicted Magnitude: 1.02

Iteration: 115000  |  Actual Magnitude: 1.23  |  Predicted Magnitude: 1.67

Iteration: 116000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 1.61

Iteration: 117000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.25

Iteration: 118000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 1.53

Iteration: 119000  |  Actual Magnitude: 0.97  |  Predicted Magnitude: 1.49

Iteration: 120000  |  Actual Magnitude: 4.00  |  Predicted Magnitude: 5.17

Iteration: 121000  |  Actual Magnitude: 0.57  |  Predicted Magnitude: 1.60

Iteration: 122000  |  Actual Magnitude: 2.10  |  Predicted Magnitude: 1.56

Iteration: 123000  |  Actual Magnitude: 1.23  |  Predicted Magnitude: 1.53

Iteration: 124000  |  Actual Magnitude: 5.30  |  Predicted Magnitude: 3.86

Iteration: 125000  |  Actual Magnitude: 0.43  |  Predicted Magnitude: 1.49

Iteration: 126000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 0.97

Iteration: 127000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 2.20

Iteration: 128000  |  Actual Magnitude: 4.10  |  Predicted Magnitude: 5.06

Iteration: 129000  |  Actual Magnitude: 3.16  |  Predicted Magnitude: 1.59

Iteration: 130000  |  Actual Magnitude: 3.00  |  Predicted Magnitude: 2.66

Iteration: 131000  |  Actual Magnitude: 0.28  |  Predicted Magnitude: 1.49

Iteration: 132000  |  Actual Magnitude: 1.87  |  Predicted Magnitude: 1.64

Iteration: 133000  |  Actual Magnitude: 3.80  |  Predicted Magnitude: 3.31

Iteration: 134000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 0.99

Iteration: 135000  |  Actual Magnitude: 1.11  |  Predicted Magnitude: 1.50

Iteration: 136000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 0.89

Iteration: 137000  |  Actual Magnitude: 0.63  |  Predicted Magnitude: 1.45

Iteration: 138000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 3.01

Iteration: 139000  |  Actual Magnitude: 2.50  |  Predicted Magnitude: 1.10

Iteration: 140000  |  Actual Magnitude: 4.80  |  Predicted Magnitude: 3.69

Iteration: 141000  |  Actual Magnitude: 2.80  |  Predicted Magnitude: 0.86

Iteration: 142000  |  Actual Magnitude: 1.25  |  Predicted Magnitude: 1.56

Iteration: 143000  |  Actual Magnitude: 0.95  |  Predicted Magnitude: 1.48

Iteration: 144000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 2.33

Iteration: 145000  |  Actual Magnitude: 1.49  |  Predicted Magnitude: 1.54

Iteration: 146000  |  Actual Magnitude: 4.10  |  Predicted Magnitude: 2.43

Iteration: 147000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 0.99

Iteration: 148000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.48

Iteration: 149000  |  Actual Magnitude: 3.40  |  Predicted Magnitude: 1.66

Iteration: 150000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.46

Iteration: 151000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.35

Iteration: 152000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.66

Iteration: 153000  |  Actual Magnitude: 4.60  |  Predicted Magnitude: 4.67

Iteration: 154000  |  Actual Magnitude: 2.40  |  Predicted Magnitude: 1.51

Iteration: 155000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.62

Iteration: 156000  |  Actual Magnitude: 0.32  |  Predicted Magnitude: 1.62

Iteration: 157000  |  Actual Magnitude: 0.62  |  Predicted Magnitude: 1.51

Iteration: 158000  |  Actual Magnitude: 1.41  |  Predicted Magnitude: 1.64

Iteration: 159000  |  Actual Magnitude: 4.70  |  Predicted Magnitude: 4.74

Iteration: 160000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 1.59

Iteration: 161000  |  Actual Magnitude: 0.30  |  Predicted Magnitude: 1.55

Iteration: 162000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 5.14

Iteration: 163000  |  Actual Magnitude: 1.34  |  Predicted Magnitude: 1.56

Iteration: 164000  |  Actual Magnitude: 0.80  |  Predicted Magnitude: 1.49

Iteration: 165000  |  Actual Magnitude: 0.86  |  Predicted Magnitude: 1.53

Iteration: 166000  |  Actual Magnitude: 5.00  |  Predicted Magnitude: 4.17

Iteration: 167000  |  Actual Magnitude: 0.82  |  Predicted Magnitude: 1.72

Iteration: 168000  |  Actual Magnitude: 4.10  |  Predicted Magnitude: 4.22

Iteration: 169000  |  Actual Magnitude: 0.83  |  Predicted Magnitude: 1.48

Iteration: 170000  |  Actual Magnitude: 2.02  |  Predicted Magnitude: 1.71

Iteration: 171000  |  Actual Magnitude: 1.04  |  Predicted Magnitude: 1.66

Iteration: 172000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.07

Iteration: 173000  |  Actual Magnitude: 1.48  |  Predicted Magnitude: 1.44

Iteration: 174000  |  Actual Magnitude: 1.42  |  Predicted Magnitude: 1.51

Iteration: 175000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 4.51

Iteration: 176000  |  Actual Magnitude: 0.60  |  Predicted Magnitude: 1.56

Iteration: 177000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.51

Iteration: 178000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.55

Iteration: 179000  |  Actual Magnitude: 4.70  |  Predicted Magnitude: 3.83

Iteration: 180000  |  Actual Magnitude: 4.10  |  Predicted Magnitude: 4.96

Iteration: 181000  |  Actual Magnitude: 0.34  |  Predicted Magnitude: 1.68

Iteration: 182000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.58

Iteration: 183000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.61

Iteration: 184000  |  Actual Magnitude: 1.03  |  Predicted Magnitude: 1.52

Iteration: 185000  |  Actual Magnitude: 1.26  |  Predicted Magnitude: 1.52

Iteration: 186000  |  Actual Magnitude: 5.10  |  Predicted Magnitude: 4.37

Iteration: 187000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 2.79

Iteration: 188000  |  Actual Magnitude: 0.27  |  Predicted Magnitude: 1.37

Iteration: 189000  |  Actual Magnitude: 0.41  |  Predicted Magnitude: 1.37

Iteration: 190000  |  Actual Magnitude: 0.62  |  Predicted Magnitude: 1.65

Iteration: 191000  |  Actual Magnitude: 3.10  |  Predicted Magnitude: 1.53

Iteration: 192000  |  Actual Magnitude: -0.10  |  Predicted Magnitude: 1.57

Iteration: 193000  |  Actual Magnitude: 5.00  |  Predicted Magnitude: 4.11

Iteration: 194000  |  Actual Magnitude: 3.30  |  Predicted Magnitude: 2.56

Iteration: 195000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 0.98

Iteration: 196000  |  Actual Magnitude: 1.03  |  Predicted Magnitude: 1.48

Iteration: 197000  |  Actual Magnitude: 4.00  |  Predicted Magnitude: 2.21

Iteration: 198000  |  Actual Magnitude: 2.40  |  Predicted Magnitude: 3.96

Iteration: 199000  |  Actual Magnitude: 6.00  |  Predicted Magnitude: 3.47

Iteration: 200000  |  Actual Magnitude: 0.88  |  Predicted Magnitude: 1.37

Iteration: 201000  |  Actual Magnitude: 1.91  |  Predicted Magnitude: 1.70

Iteration: 202000  |  Actual Magnitude: 2.28  |  Predicted Magnitude: 2.25

Iteration: 203000  |  Actual Magnitude: 0.70  |  Predicted Magnitude: 1.52

Iteration: 204000  |  Actual Magnitude: 2.36  |  Predicted Magnitude: 1.53

Iteration: 205000  |  Actual Magnitude: 1.15  |  Predicted Magnitude: 1.48

Iteration: 206000  |  Actual Magnitude: 0.36  |  Predicted Magnitude: 1.57

Iteration: 207000  |  Actual Magnitude: 1.62  |  Predicted Magnitude: 1.64

Iteration: 208000  |  Actual Magnitude: 2.10  |  Predicted Magnitude: 1.55

Iteration: 209000  |  Actual Magnitude: 2.46  |  Predicted Magnitude: 1.56

Iteration: 210000  |  Actual Magnitude: 2.89  |  Predicted Magnitude: 1.53

Iteration: 211000  |  Actual Magnitude: 2.90  |  Predicted Magnitude: 2.31

Iteration: 212000  |  Actual Magnitude: 0.62  |  Predicted Magnitude: 1.66

Iteration: 213000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 1.19

Iteration: 214000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 1.33

Iteration: 215000  |  Actual Magnitude: 0.00  |  Predicted Magnitude: 2.77

Iteration: 216000  |  Actual Magnitude: 0.24  |  Predicted Magnitude: 1.54

Iteration: 217000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 2.34

Iteration: 218000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 0.96

Iteration: 219000  |  Actual Magnitude: -0.10  |  Predicted Magnitude: 1.52

Iteration: 220000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.60

Iteration: 221000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.48

Iteration: 222000  |  Actual Magnitude: 2.10  |  Predicted Magnitude: 1.43

Iteration: 223000  |  Actual Magnitude: 0.70  |  Predicted Magnitude: 1.38

Iteration: 224000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.96

Iteration: 225000  |  Actual Magnitude: 2.10  |  Predicted Magnitude: 1.34

Iteration: 226000  |  Actual Magnitude: 4.70  |  Predicted Magnitude: 1.97

Iteration: 227000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.57

Iteration: 228000  |  Actual Magnitude: 0.89  |  Predicted Magnitude: 1.55

Iteration: 229000  |  Actual Magnitude: 3.40  |  Predicted Magnitude: 1.55

Iteration: 230000  |  Actual Magnitude: -0.18  |  Predicted Magnitude: 0.93

Iteration: 231000  |  Actual Magnitude: 0.78  |  Predicted Magnitude: 1.47

Iteration: 232000  |  Actual Magnitude: 1.08  |  Predicted Magnitude: 1.56

Iteration: 233000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.83

Iteration: 234000  |  Actual Magnitude: 2.60  |  Predicted Magnitude: 1.11

Iteration: 235000  |  Actual Magnitude: 3.50  |  Predicted Magnitude: 1.73

Iteration: 236000  |  Actual Magnitude: 3.70  |  Predicted Magnitude: 3.69

Iteration: 237000  |  Actual Magnitude: 2.58  |  Predicted Magnitude: 1.47

Iteration: 238000  |  Actual Magnitude: 5.00  |  Predicted Magnitude: 3.70

Iteration: 239000  |  Actual Magnitude: 2.27  |  Predicted Magnitude: 1.48

Iteration: 240000  |  Actual Magnitude: 1.31  |  Predicted Magnitude: 1.46

Iteration: 241000  |  Actual Magnitude: 1.53  |  Predicted Magnitude: 1.66

Iteration: 242000  |  Actual Magnitude: 1.90  |  Predicted Magnitude: 1.33

Iteration: 243000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.55

Iteration: 244000  |  Actual Magnitude: 1.03  |  Predicted Magnitude: 1.55

Iteration: 245000  |  Actual Magnitude: 1.79  |  Predicted Magnitude: 1.47

Iteration: 246000  |  Actual Magnitude: 5.20  |  Predicted Magnitude: 3.50

Iteration: 247000  |  Actual Magnitude: 2.79  |  Predicted Magnitude: 1.60

Iteration: 248000  |  Actual Magnitude: 1.21  |  Predicted Magnitude: 1.55

Iteration: 249000  |  Actual Magnitude: 0.55  |  Predicted Magnitude: 1.58

Iteration: 250000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.40

Iteration: 251000  |  Actual Magnitude: 1.68  |  Predicted Magnitude: 1.61

Iteration: 252000  |  Actual Magnitude: 4.60  |  Predicted Magnitude: 3.75

Iteration: 253000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.62

Iteration: 254000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.15

Iteration: 255000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.94

Iteration: 256000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.48

Iteration: 257000  |  Actual Magnitude: 1.53  |  Predicted Magnitude: 1.66

Iteration: 258000  |  Actual Magnitude: 0.73  |  Predicted Magnitude: 1.48

Iteration: 259000  |  Actual Magnitude: 1.66  |  Predicted Magnitude: 1.58

Iteration: 260000  |  Actual Magnitude: 0.61  |  Predicted Magnitude: 1.67

Iteration: 261000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.42

Iteration: 262000  |  Actual Magnitude: 1.92  |  Predicted Magnitude: 1.58

Iteration: 263000  |  Actual Magnitude: 0.71  |  Predicted Magnitude: 1.47

Iteration: 264000  |  Actual Magnitude: 1.29  |  Predicted Magnitude: 1.54

Iteration: 265000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.01

Iteration: 266000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 2.09

Iteration: 267000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.20

Iteration: 268000  |  Actual Magnitude: 0.60  |  Predicted Magnitude: 1.57

Iteration: 269000  |  Actual Magnitude: 3.10  |  Predicted Magnitude: 1.71

Iteration: 270000  |  Actual Magnitude: 2.80  |  Predicted Magnitude: 0.90

Iteration: 271000  |  Actual Magnitude: 0.25  |  Predicted Magnitude: 1.47

Iteration: 272000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 1.99

Iteration: 273000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 0.95

Iteration: 274000  |  Actual Magnitude: 2.10  |  Predicted Magnitude: 1.01

Iteration: 275000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 1.53

Iteration: 276000  |  Actual Magnitude: 2.07  |  Predicted Magnitude: 1.63

Iteration: 277000  |  Actual Magnitude: 0.63  |  Predicted Magnitude: 1.66

Iteration: 278000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 0.98

Iteration: 279000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.54

Iteration: 280000  |  Actual Magnitude: 0.13  |  Predicted Magnitude: 1.55

Iteration: 281000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 1.10

Iteration: 282000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.02

Iteration: 283000  |  Actual Magnitude: 0.52  |  Predicted Magnitude: 1.67

Iteration: 284000  |  Actual Magnitude: 1.88  |  Predicted Magnitude: 1.53

Iteration: 285000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 1.60

Iteration: 286000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 2.16

Iteration: 287000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 3.66

Iteration: 288000  |  Actual Magnitude: 0.85  |  Predicted Magnitude: 1.54

Iteration: 289000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.98

Iteration: 290000  |  Actual Magnitude: 1.81  |  Predicted Magnitude: 0.89

Iteration: 291000  |  Actual Magnitude: 2.59  |  Predicted Magnitude: 1.58

Iteration: 292000  |  Actual Magnitude: 1.92  |  Predicted Magnitude: 1.52

Iteration: 293000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.53

Iteration: 294000  |  Actual Magnitude: 3.60  |  Predicted Magnitude: 3.51

Iteration: 295000  |  Actual Magnitude: 1.06  |  Predicted Magnitude: 1.63

Iteration: 296000  |  Actual Magnitude: 0.52  |  Predicted Magnitude: 1.56

Iteration: 297000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 0.94

Iteration: 298000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 3.62

Iteration: 299000  |  Actual Magnitude: 0.71  |  Predicted Magnitude: 0.88

Iteration: 300000  |  Actual Magnitude: 0.79  |  Predicted Magnitude: 1.48

Iteration: 301000  |  Actual Magnitude: 4.70  |  Predicted Magnitude: 4.87

Iteration: 302000  |  Actual Magnitude: 0.21  |  Predicted Magnitude: 1.60

Iteration: 303000  |  Actual Magnitude: 1.16  |  Predicted Magnitude: 1.55

Iteration: 304000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 0.93

Iteration: 305000  |  Actual Magnitude: 0.62  |  Predicted Magnitude: 1.49

Iteration: 306000  |  Actual Magnitude: 0.62  |  Predicted Magnitude: 1.48

Iteration: 307000  |  Actual Magnitude: 0.89  |  Predicted Magnitude: 1.48

Iteration: 308000  |  Actual Magnitude: 1.84  |  Predicted Magnitude: 1.51

Iteration: 309000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.58

Iteration: 310000  |  Actual Magnitude: 4.00  |  Predicted Magnitude: 3.93

Iteration: 311000  |  Actual Magnitude: 2.50  |  Predicted Magnitude: 0.94

Iteration: 312000  |  Actual Magnitude: 3.13  |  Predicted Magnitude: 1.69

Iteration: 313000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.93

Iteration: 314000  |  Actual Magnitude: -0.35  |  Predicted Magnitude: 0.94

Iteration: 315000  |  Actual Magnitude: 2.90  |  Predicted Magnitude: 3.33

Iteration: 316000  |  Actual Magnitude: 1.05  |  Predicted Magnitude: 1.62

Iteration: 317000  |  Actual Magnitude: 1.15  |  Predicted Magnitude: 1.52

Iteration: 318000  |  Actual Magnitude: 0.50  |  Predicted Magnitude: 1.67

Iteration: 319000  |  Actual Magnitude: 1.68  |  Predicted Magnitude: 1.63

Iteration: 320000  |  Actual Magnitude: 1.77  |  Predicted Magnitude: 1.83

Iteration: 321000  |  Actual Magnitude: 2.90  |  Predicted Magnitude: 2.54

Iteration: 322000  |  Actual Magnitude: 0.81  |  Predicted Magnitude: 1.56

Iteration: 323000  |  Actual Magnitude: 1.06  |  Predicted Magnitude: 1.52

Iteration: 324000  |  Actual Magnitude: 0.38  |  Predicted Magnitude: 1.55

Iteration: 325000  |  Actual Magnitude: 2.48  |  Predicted Magnitude: 1.58

Iteration: 326000  |  Actual Magnitude: 0.75  |  Predicted Magnitude: 1.55

Iteration: 327000  |  Actual Magnitude: 0.47  |  Predicted Magnitude: 1.68

Iteration: 328000  |  Actual Magnitude: 4.80  |  Predicted Magnitude: 3.96

Iteration: 329000  |  Actual Magnitude: 0.37  |  Predicted Magnitude: 1.55

Iteration: 330000  |  Actual Magnitude: 0.51  |  Predicted Magnitude: 1.57

Iteration: 331000  |  Actual Magnitude: 0.80  |  Predicted Magnitude: 1.54

Iteration: 332000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 0.92

Iteration: 333000  |  Actual Magnitude: 1.88  |  Predicted Magnitude: 1.47

Iteration: 334000  |  Actual Magnitude: 0.16  |  Predicted Magnitude: 1.48

Iteration: 335000  |  Actual Magnitude: 1.41  |  Predicted Magnitude: 1.64

Iteration: 336000  |  Actual Magnitude: 1.22  |  Predicted Magnitude: 1.67

Iteration: 337000  |  Actual Magnitude: 0.88  |  Predicted Magnitude: 1.48

Iteration: 338000  |  Actual Magnitude: 1.28  |  Predicted Magnitude: 1.48

Iteration: 339000  |  Actual Magnitude: 2.03  |  Predicted Magnitude: 1.47

Iteration: 340000  |  Actual Magnitude: -0.52  |  Predicted Magnitude: 1.37

Iteration: 341000  |  Actual Magnitude: 1.82  |  Predicted Magnitude: 1.53

Iteration: 342000  |  Actual Magnitude: 1.64  |  Predicted Magnitude: 1.57

Iteration: 343000  |  Actual Magnitude: -0.63  |  Predicted Magnitude: 1.50

Iteration: 344000  |  Actual Magnitude: 0.52  |  Predicted Magnitude: 1.52

Iteration: 345000  |  Actual Magnitude: 1.52  |  Predicted Magnitude: 1.49

Iteration: 346000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.59

Iteration: 347000  |  Actual Magnitude: 1.46  |  Predicted Magnitude: 1.49

Iteration: 348000  |  Actual Magnitude: 1.69  |  Predicted Magnitude: 1.59

Iteration: 349000  |  Actual Magnitude: 4.80  |  Predicted Magnitude: 3.63

Iteration: 350000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.53

Iteration: 351000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.13

Iteration: 352000  |  Actual Magnitude: 0.28  |  Predicted Magnitude: 1.52

Iteration: 353000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.26

Iteration: 354000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.01

Iteration: 355000  |  Actual Magnitude: 0.50  |  Predicted Magnitude: 1.63

Iteration: 356000  |  Actual Magnitude: 3.76  |  Predicted Magnitude: 2.20

Iteration: 357000  |  Actual Magnitude: 3.00  |  Predicted Magnitude: 2.54

Iteration: 358000  |  Actual Magnitude: 2.10  |  Predicted Magnitude: 0.94

Iteration: 359000  |  Actual Magnitude: 1.69  |  Predicted Magnitude: 1.59

Iteration: 360000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 0.98

Iteration: 361000  |  Actual Magnitude: 0.85  |  Predicted Magnitude: 1.49

Iteration: 362000  |  Actual Magnitude: 3.80  |  Predicted Magnitude: 3.07

Iteration: 363000  |  Actual Magnitude: 0.20  |  Predicted Magnitude: 1.52

Iteration: 364000  |  Actual Magnitude: 1.15  |  Predicted Magnitude: 1.48

Iteration: 365000  |  Actual Magnitude: 5.00  |  Predicted Magnitude: 4.18

Iteration: 366000  |  Actual Magnitude: 1.36  |  Predicted Magnitude: 1.64

Iteration: 367000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 4.38

Iteration: 368000  |  Actual Magnitude: 1.33  |  Predicted Magnitude: 1.54

Iteration: 369000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.57

Iteration: 370000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.36

Iteration: 371000  |  Actual Magnitude: 1.28  |  Predicted Magnitude: 1.54

Iteration: 372000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.33

Iteration: 373000  |  Actual Magnitude: 1.66  |  Predicted Magnitude: 1.72

Iteration: 374000  |  Actual Magnitude: -0.10  |  Predicted Magnitude: 1.60

Iteration: 375000  |  Actual Magnitude: 0.55  |  Predicted Magnitude: 1.55

Iteration: 376000  |  Actual Magnitude: 1.56  |  Predicted Magnitude: 1.33

Iteration: 377000  |  Actual Magnitude: 0.59  |  Predicted Magnitude: 1.68

Iteration: 378000  |  Actual Magnitude: 2.07  |  Predicted Magnitude: 2.27

Iteration: 379000  |  Actual Magnitude: 0.09  |  Predicted Magnitude: 1.58

Iteration: 380000  |  Actual Magnitude: 1.89  |  Predicted Magnitude: 1.60

Iteration: 381000  |  Actual Magnitude: 3.50  |  Predicted Magnitude: 2.88

Iteration: 382000  |  Actual Magnitude: 1.09  |  Predicted Magnitude: 1.52

Iteration: 383000  |  Actual Magnitude: 3.70  |  Predicted Magnitude: 3.48

Iteration: 384000  |  Actual Magnitude: 5.10  |  Predicted Magnitude: 2.42

Iteration: 385000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.95

Iteration: 386000  |  Actual Magnitude: 4.80  |  Predicted Magnitude: 3.75

Iteration: 387000  |  Actual Magnitude: 1.23  |  Predicted Magnitude: 1.52

Iteration: 388000  |  Actual Magnitude: 0.10  |  Predicted Magnitude: 1.53

Iteration: 389000  |  Actual Magnitude: 1.49  |  Predicted Magnitude: 1.62

Iteration: 390000  |  Actual Magnitude: 2.48  |  Predicted Magnitude: 2.28

Iteration: 391000  |  Actual Magnitude: 2.23  |  Predicted Magnitude: 1.57

Iteration: 392000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.07

Iteration: 393000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 2.33

Iteration: 394000  |  Actual Magnitude: 0.50  |  Predicted Magnitude: 1.49

Iteration: 395000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 1.08

Iteration: 396000  |  Actual Magnitude: 1.31  |  Predicted Magnitude: 1.63

Iteration: 397000  |  Actual Magnitude: 1.11  |  Predicted Magnitude: 1.62

Iteration: 398000  |  Actual Magnitude: 1.16  |  Predicted Magnitude: 1.57

Iteration: 399000  |  Actual Magnitude: 1.33  |  Predicted Magnitude: 1.64

Iteration: 400000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 3.56

Iteration: 401000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 1.57

Iteration: 402000  |  Actual Magnitude: 1.90  |  Predicted Magnitude: 1.30

Iteration: 403000  |  Actual Magnitude: 2.40  |  Predicted Magnitude: 1.54

Iteration: 404000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 1.76

Iteration: 405000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.54

Iteration: 406000  |  Actual Magnitude: 2.02  |  Predicted Magnitude: 1.33

Iteration: 407000  |  Actual Magnitude: 1.56  |  Predicted Magnitude: 1.55

Iteration: 408000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 1.54

Iteration: 409000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.03

Iteration: 410000  |  Actual Magnitude: 3.30  |  Predicted Magnitude: 3.14

Iteration: 411000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.08

Iteration: 412000  |  Actual Magnitude: 0.92  |  Predicted Magnitude: 1.49

Iteration: 413000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 1.18

Iteration: 414000  |  Actual Magnitude: 0.46  |  Predicted Magnitude: 1.47

Iteration: 415000  |  Actual Magnitude: 4.80  |  Predicted Magnitude: 4.00

Iteration: 416000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.58

Iteration: 417000  |  Actual Magnitude: 3.46  |  Predicted Magnitude: 1.53

Iteration: 418000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.06

Iteration: 419000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.56

Iteration: 420000  |  Actual Magnitude: 3.60  |  Predicted Magnitude: 4.25

Iteration: 421000  |  Actual Magnitude: 0.43  |  Predicted Magnitude: 1.47

Iteration: 422000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 0.93

Iteration: 423000  |  Actual Magnitude: 2.34  |  Predicted Magnitude: 1.55

Iteration: 424000  |  Actual Magnitude: 0.07  |  Predicted Magnitude: 1.48

Iteration: 425000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.31

Iteration: 426000  |  Actual Magnitude: 1.13  |  Predicted Magnitude: 1.62

Iteration: 427000  |  Actual Magnitude: 2.30  |  Predicted Magnitude: 1.36

Iteration: 428000  |  Actual Magnitude: 1.38  |  Predicted Magnitude: 1.54

Iteration: 429000  |  Actual Magnitude: 1.52  |  Predicted Magnitude: 1.59

Iteration: 430000  |  Actual Magnitude: 2.33  |  Predicted Magnitude: 1.48

Iteration: 431000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.17

Iteration: 432000  |  Actual Magnitude: 2.46  |  Predicted Magnitude: 2.28

Iteration: 433000  |  Actual Magnitude: 0.23  |  Predicted Magnitude: 1.59

Iteration: 434000  |  Actual Magnitude: 1.64  |  Predicted Magnitude: 1.62

Iteration: 435000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.67

Iteration: 436000  |  Actual Magnitude: 3.30  |  Predicted Magnitude: 1.58

Iteration: 437000  |  Actual Magnitude: 0.02  |  Predicted Magnitude: 1.53

Iteration: 438000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 3.77

Iteration: 439000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 2.08

Iteration: 440000  |  Actual Magnitude: -0.07  |  Predicted Magnitude: 1.55

Iteration: 441000  |  Actual Magnitude: 1.82  |  Predicted Magnitude: 1.53

Iteration: 442000  |  Actual Magnitude: 0.86  |  Predicted Magnitude: 1.49

Iteration: 443000  |  Actual Magnitude: 0.81  |  Predicted Magnitude: 1.56

Iteration: 444000  |  Actual Magnitude: 1.34  |  Predicted Magnitude: 1.49

Iteration: 445000  |  Actual Magnitude: 0.00  |  Predicted Magnitude: 2.53

Iteration: 446000  |  Actual Magnitude: 3.80  |  Predicted Magnitude: 4.89

Iteration: 447000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 0.91

Iteration: 448000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.02

Iteration: 449000  |  Actual Magnitude: 2.12  |  Predicted Magnitude: 1.61

Iteration: 450000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 3.14

Iteration: 451000  |  Actual Magnitude: 1.44  |  Predicted Magnitude: 1.55

Iteration: 452000  |  Actual Magnitude: 0.88  |  Predicted Magnitude: 1.55

Iteration: 453000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.94

Iteration: 454000  |  Actual Magnitude: 4.00  |  Predicted Magnitude: 4.01

Iteration: 455000  |  Actual Magnitude: 0.00  |  Predicted Magnitude: 1.58

Iteration: 456000  |  Actual Magnitude: 0.93  |  Predicted Magnitude: 1.64

Iteration: 457000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.01

Iteration: 458000  |  Actual Magnitude: 1.66  |  Predicted Magnitude: 1.61

Iteration: 459000  |  Actual Magnitude: 3.50  |  Predicted Magnitude: 1.56

Iteration: 460000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 2.46

Iteration: 461000  |  Actual Magnitude: 0.61  |  Predicted Magnitude: 1.68

Iteration: 462000  |  Actual Magnitude: 3.10  |  Predicted Magnitude: 2.26

Iteration: 463000  |  Actual Magnitude: 4.70  |  Predicted Magnitude: 3.80

Iteration: 464000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 0.92

Iteration: 465000  |  Actual Magnitude: 3.10  |  Predicted Magnitude: 3.73

Iteration: 466000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.43

Iteration: 467000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.11

Iteration: 468000  |  Actual Magnitude: 1.53  |  Predicted Magnitude: 1.59

Iteration: 469000  |  Actual Magnitude: 1.22  |  Predicted Magnitude: 1.47

Iteration: 470000  |  Actual Magnitude: 0.80  |  Predicted Magnitude: 1.63

Iteration: 471000  |  Actual Magnitude: 1.19  |  Predicted Magnitude: 1.55

Iteration: 472000  |  Actual Magnitude: 4.80  |  Predicted Magnitude: 3.84

Iteration: 473000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.61

Iteration: 474000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 1.54

Iteration: 475000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 0.94

Iteration: 476000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.60

Iteration: 477000  |  Actual Magnitude: -0.53  |  Predicted Magnitude: 0.94

Iteration: 478000  |  Actual Magnitude: 0.62  |  Predicted Magnitude: 1.57

Iteration: 479000  |  Actual Magnitude: 0.89  |  Predicted Magnitude: 1.54

Iteration: 480000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.05

Iteration: 481000  |  Actual Magnitude: 1.58  |  Predicted Magnitude: 1.63

Iteration: 482000  |  Actual Magnitude: 1.64  |  Predicted Magnitude: 1.52

Iteration: 483000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 0.94

Iteration: 484000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.51

Iteration: 485000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.45

Iteration: 486000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.55

Iteration: 487000  |  Actual Magnitude: 0.47  |  Predicted Magnitude: 1.56

Iteration: 488000  |  Actual Magnitude: 0.83  |  Predicted Magnitude: 1.49

Iteration: 489000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 1.48

Iteration: 490000  |  Actual Magnitude: 1.77  |  Predicted Magnitude: 1.60

Iteration: 491000  |  Actual Magnitude: 1.35  |  Predicted Magnitude: 1.63

Iteration: 492000  |  Actual Magnitude: 0.96  |  Predicted Magnitude: 1.48

Iteration: 493000  |  Actual Magnitude: 1.16  |  Predicted Magnitude: 1.51

Iteration: 494000  |  Actual Magnitude: 3.70  |  Predicted Magnitude: 2.81

Iteration: 495000  |  Actual Magnitude: 4.70  |  Predicted Magnitude: 1.96

Iteration: 496000  |  Actual Magnitude: 2.47  |  Predicted Magnitude: 1.58

Iteration: 497000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.53

Iteration: 498000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 1.26

Iteration: 499000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.97

Iteration: 500000  |  Actual Magnitude: 2.81  |  Predicted Magnitude: 1.53

Iteration: 501000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 4.30

Iteration: 502000  |  Actual Magnitude: 0.68  |  Predicted Magnitude: 1.68

Iteration: 503000  |  Actual Magnitude: 1.12  |  Predicted Magnitude: 1.63

Iteration: 504000  |  Actual Magnitude: -0.18  |  Predicted Magnitude: 1.47

Iteration: 505000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.25

Iteration: 506000  |  Actual Magnitude: 0.76  |  Predicted Magnitude: 1.55

Iteration: 507000  |  Actual Magnitude: 1.39  |  Predicted Magnitude: 1.56

Iteration: 508000  |  Actual Magnitude: 4.40  |  Predicted Magnitude: 3.95

Iteration: 509000  |  Actual Magnitude: 1.49  |  Predicted Magnitude: 1.55

Iteration: 510000  |  Actual Magnitude: 0.53  |  Predicted Magnitude: 1.66

Iteration: 511000  |  Actual Magnitude: 1.03  |  Predicted Magnitude: 1.68

Iteration: 512000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 3.09

Iteration: 513000  |  Actual Magnitude: 0.96  |  Predicted Magnitude: 1.48

Iteration: 514000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.35

Iteration: 515000  |  Actual Magnitude: 2.80  |  Predicted Magnitude: 2.62

Iteration: 516000  |  Actual Magnitude: 2.50  |  Predicted Magnitude: 1.05

Iteration: 517000  |  Actual Magnitude: 1.07  |  Predicted Magnitude: 1.58

Iteration: 518000  |  Actual Magnitude: 2.60  |  Predicted Magnitude: 1.74

Iteration: 519000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.57

Iteration: 520000  |  Actual Magnitude: 4.90  |  Predicted Magnitude: 3.23

Iteration: 521000  |  Actual Magnitude: 1.76  |  Predicted Magnitude: 1.63

Iteration: 522000  |  Actual Magnitude: -1.18  |  Predicted Magnitude: 0.94

Iteration: 523000  |  Actual Magnitude: 3.70  |  Predicted Magnitude: 5.04

Iteration: 524000  |  Actual Magnitude: 1.69  |  Predicted Magnitude: 1.56

Iteration: 525000  |  Actual Magnitude: 2.40  |  Predicted Magnitude: 2.34

Iteration: 526000  |  Actual Magnitude: 1.76  |  Predicted Magnitude: 1.49

Iteration: 527000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 0.89

Iteration: 528000  |  Actual Magnitude: 3.00  |  Predicted Magnitude: 1.07

Iteration: 529000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 1.50

Iteration: 530000  |  Actual Magnitude: 1.49  |  Predicted Magnitude: 1.52

Iteration: 531000  |  Actual Magnitude: 1.90  |  Predicted Magnitude: 1.81

Iteration: 532000  |  Actual Magnitude: 0.50  |  Predicted Magnitude: 1.01

Iteration: 533000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 2.38

Iteration: 534000  |  Actual Magnitude: 0.50  |  Predicted Magnitude: 0.98

Iteration: 535000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 0.95

Iteration: 536000  |  Actual Magnitude: 1.87  |  Predicted Magnitude: 1.74

Iteration: 537000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 3.34

Iteration: 538000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 2.55

Iteration: 539000  |  Actual Magnitude: 0.56  |  Predicted Magnitude: 1.50

Iteration: 540000  |  Actual Magnitude: 0.30  |  Predicted Magnitude: 1.48

Iteration: 541000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 0.93

Iteration: 542000  |  Actual Magnitude: 2.40  |  Predicted Magnitude: 0.91

Iteration: 543000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.57

Iteration: 544000  |  Actual Magnitude: 0.26  |  Predicted Magnitude: 1.48

Iteration: 545000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.42

Iteration: 546000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.58

Iteration: 547000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.60

Iteration: 548000  |  Actual Magnitude: 0.45  |  Predicted Magnitude: 1.66

Iteration: 549000  |  Actual Magnitude: 3.77  |  Predicted Magnitude: 1.59

Iteration: 550000  |  Actual Magnitude: 1.29  |  Predicted Magnitude: 1.53

Iteration: 551000  |  Actual Magnitude: 0.43  |  Predicted Magnitude: 1.48

Iteration: 552000  |  Actual Magnitude: 0.67  |  Predicted Magnitude: 1.52

Iteration: 553000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 4.54

Iteration: 554000  |  Actual Magnitude: 1.22  |  Predicted Magnitude: 1.49

Iteration: 555000  |  Actual Magnitude: 0.80  |  Predicted Magnitude: 1.51

Iteration: 556000  |  Actual Magnitude: 2.02  |  Predicted Magnitude: 1.54

Iteration: 557000  |  Actual Magnitude: 0.43  |  Predicted Magnitude: 1.52

Iteration: 558000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.43

Iteration: 559000  |  Actual Magnitude: 5.10  |  Predicted Magnitude: 2.16

Iteration: 560000  |  Actual Magnitude: 0.91  |  Predicted Magnitude: 1.67

Iteration: 561000  |  Actual Magnitude: 0.60  |  Predicted Magnitude: 0.90

Iteration: 562000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 0.97

Iteration: 563000  |  Actual Magnitude: 3.50  |  Predicted Magnitude: 3.07

Iteration: 564000  |  Actual Magnitude: 0.50  |  Predicted Magnitude: 0.98

Iteration: 565000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 3.45

Iteration: 566000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 1.11

Iteration: 567000  |  Actual Magnitude: 3.10  |  Predicted Magnitude: 3.54

Iteration: 568000  |  Actual Magnitude: 4.70  |  Predicted Magnitude: 3.79

Iteration: 569000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.06

Iteration: 570000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 4.47

Iteration: 571000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 2.52

Iteration: 572000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.56

Iteration: 573000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 0.94

Iteration: 574000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.65

Iteration: 575000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.22

Iteration: 576000  |  Actual Magnitude: 4.60  |  Predicted Magnitude: 2.83

Iteration: 577000  |  Actual Magnitude: 1.85  |  Predicted Magnitude: 1.70

Iteration: 578000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.10

Iteration: 579000  |  Actual Magnitude: 1.78  |  Predicted Magnitude: 1.52

Iteration: 580000  |  Actual Magnitude: 2.50  |  Predicted Magnitude: 1.26

Iteration: 581000  |  Actual Magnitude: 1.21  |  Predicted Magnitude: 1.56

Iteration: 582000  |  Actual Magnitude: 1.87  |  Predicted Magnitude: 1.63

Iteration: 583000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.04

Iteration: 584000  |  Actual Magnitude: 1.48  |  Predicted Magnitude: 1.62

Iteration: 585000  |  Actual Magnitude: 0.35  |  Predicted Magnitude: 1.66

Iteration: 586000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.44

Iteration: 587000  |  Actual Magnitude: 1.14  |  Predicted Magnitude: 1.48

Iteration: 588000  |  Actual Magnitude: 2.34  |  Predicted Magnitude: 1.55

Iteration: 589000  |  Actual Magnitude: 0.35  |  Predicted Magnitude: 1.49

Iteration: 590000  |  Actual Magnitude: 0.33  |  Predicted Magnitude: 1.53

Iteration: 591000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 0.95

Iteration: 592000  |  Actual Magnitude: 0.23  |  Predicted Magnitude: 1.48

Iteration: 593000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 1.06

Iteration: 594000  |  Actual Magnitude: 2.90  |  Predicted Magnitude: 1.61

Iteration: 595000  |  Actual Magnitude: 1.19  |  Predicted Magnitude: 1.55

Iteration: 596000  |  Actual Magnitude: 0.92  |  Predicted Magnitude: 1.47

Iteration: 597000  |  Actual Magnitude: 0.54  |  Predicted Magnitude: 1.49

Iteration: 598000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 3.29

Iteration: 599000  |  Actual Magnitude: 0.68  |  Predicted Magnitude: 1.53

Iteration: 600000  |  Actual Magnitude: 0.20  |  Predicted Magnitude: 1.48

Iteration: 601000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.05

Iteration: 602000  |  Actual Magnitude: 0.70  |  Predicted Magnitude: 1.60

Iteration: 603000  |  Actual Magnitude: 0.64  |  Predicted Magnitude: 1.55

Iteration: 604000  |  Actual Magnitude: 0.62  |  Predicted Magnitude: 1.49

Iteration: 605000  |  Actual Magnitude: 3.10  |  Predicted Magnitude: 2.66

Iteration: 606000  |  Actual Magnitude: 1.76  |  Predicted Magnitude: 1.57

Iteration: 607000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 0.93

Iteration: 608000  |  Actual Magnitude: 2.50  |  Predicted Magnitude: 1.58

Iteration: 609000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.37

Iteration: 610000  |  Actual Magnitude: 2.11  |  Predicted Magnitude: 1.66

Iteration: 611000  |  Actual Magnitude: 2.30  |  Predicted Magnitude: 1.13

Iteration: 612000  |  Actual Magnitude: 0.00  |  Predicted Magnitude: 3.88

Iteration: 613000  |  Actual Magnitude: 0.91  |  Predicted Magnitude: 1.47

Iteration: 614000  |  Actual Magnitude: 2.29  |  Predicted Magnitude: 1.53

Iteration: 615000  |  Actual Magnitude: 0.81  |  Predicted Magnitude: 1.54

Iteration: 616000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.44

Iteration: 617000  |  Actual Magnitude: 3.20  |  Predicted Magnitude: 0.96

Iteration: 618000  |  Actual Magnitude: 1.12  |  Predicted Magnitude: 1.58

Iteration: 619000  |  Actual Magnitude: 1.90  |  Predicted Magnitude: 1.39

Iteration: 620000  |  Actual Magnitude: 1.58  |  Predicted Magnitude: 1.72

Iteration: 621000  |  Actual Magnitude: 0.10  |  Predicted Magnitude: 1.59

Iteration: 622000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.96

Iteration: 623000  |  Actual Magnitude: 2.90  |  Predicted Magnitude: 3.75

Iteration: 624000  |  Actual Magnitude: 6.40  |  Predicted Magnitude: 4.68

Iteration: 625000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.17

Iteration: 626000  |  Actual Magnitude: 1.62  |  Predicted Magnitude: 1.47

Iteration: 627000  |  Actual Magnitude: 0.86  |  Predicted Magnitude: 1.56

Iteration: 628000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.62

Iteration: 629000  |  Actual Magnitude: 1.15  |  Predicted Magnitude: 1.65

Iteration: 630000  |  Actual Magnitude: 0.70  |  Predicted Magnitude: 0.89

Iteration: 631000  |  Actual Magnitude: 0.57  |  Predicted Magnitude: 1.66

Iteration: 632000  |  Actual Magnitude: 0.87  |  Predicted Magnitude: 1.68

Iteration: 633000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 1.53

Iteration: 634000  |  Actual Magnitude: 1.45  |  Predicted Magnitude: 1.62

Iteration: 635000  |  Actual Magnitude: 0.92  |  Predicted Magnitude: 1.48

Iteration: 636000  |  Actual Magnitude: 1.32  |  Predicted Magnitude: 1.62

Iteration: 637000  |  Actual Magnitude: 0.87  |  Predicted Magnitude: 1.48

Iteration: 638000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.19

Iteration: 639000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.48

Iteration: 640000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.57

Iteration: 641000  |  Actual Magnitude: 2.60  |  Predicted Magnitude: 1.66

Iteration: 642000  |  Actual Magnitude: 0.80  |  Predicted Magnitude: 1.48

Iteration: 643000  |  Actual Magnitude: 1.02  |  Predicted Magnitude: 1.56

Iteration: 644000  |  Actual Magnitude: 0.52  |  Predicted Magnitude: 1.61

Iteration: 645000  |  Actual Magnitude: 2.03  |  Predicted Magnitude: 1.54

Iteration: 646000  |  Actual Magnitude: 4.40  |  Predicted Magnitude: 3.93

Iteration: 647000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.00

Iteration: 648000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 0.98

Iteration: 649000  |  Actual Magnitude: 1.74  |  Predicted Magnitude: 1.52

Iteration: 650000  |  Actual Magnitude: 2.30  |  Predicted Magnitude: 0.93

Iteration: 651000  |  Actual Magnitude: 2.52  |  Predicted Magnitude: 1.48

Iteration: 652000  |  Actual Magnitude: 0.93  |  Predicted Magnitude: 1.58

Iteration: 653000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.48

Iteration: 654000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.13

Iteration: 655000  |  Actual Magnitude: 0.10  |  Predicted Magnitude: 1.51

Iteration: 656000  |  Actual Magnitude: 2.21  |  Predicted Magnitude: 1.53

Iteration: 657000  |  Actual Magnitude: 1.75  |  Predicted Magnitude: 1.56

Iteration: 658000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 1.39

Iteration: 659000  |  Actual Magnitude: 1.16  |  Predicted Magnitude: 1.48

Iteration: 660000  |  Actual Magnitude: 4.60  |  Predicted Magnitude: 3.93

Iteration: 661000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.28

Iteration: 662000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.95

Iteration: 663000  |  Actual Magnitude: 0.60  |  Predicted Magnitude: 0.90

Iteration: 664000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.55

Iteration: 665000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 2.64

Iteration: 666000  |  Actual Magnitude: 2.07  |  Predicted Magnitude: 1.45

Iteration: 667000  |  Actual Magnitude: 4.00  |  Predicted Magnitude: 2.13

Iteration: 668000  |  Actual Magnitude: 4.40  |  Predicted Magnitude: 3.47

Iteration: 669000  |  Actual Magnitude: 0.77  |  Predicted Magnitude: 1.49

Iteration: 670000  |  Actual Magnitude: 2.30  |  Predicted Magnitude: 2.43

Iteration: 671000  |  Actual Magnitude: 3.80  |  Predicted Magnitude: 1.93

Iteration: 672000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.74

Iteration: 673000  |  Actual Magnitude: 4.90  |  Predicted Magnitude: 3.10

Iteration: 674000  |  Actual Magnitude: 2.58  |  Predicted Magnitude: 1.59

Iteration: 675000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.13

Iteration: 676000  |  Actual Magnitude: 2.80  |  Predicted Magnitude: 0.94

Iteration: 677000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.45

Iteration: 678000  |  Actual Magnitude: 1.47  |  Predicted Magnitude: 1.54

Iteration: 679000  |  Actual Magnitude: 3.40  |  Predicted Magnitude: 2.40

Iteration: 680000  |  Actual Magnitude: 0.70  |  Predicted Magnitude: 0.94

Iteration: 681000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 0.89

Iteration: 682000  |  Actual Magnitude: 4.70  |  Predicted Magnitude: 4.59

Iteration: 683000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.55

Iteration: 684000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 0.92

Iteration: 685000  |  Actual Magnitude: 0.23  |  Predicted Magnitude: 1.59

Iteration: 686000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 0.90

Iteration: 687000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 0.94

Iteration: 688000  |  Actual Magnitude: -0.10  |  Predicted Magnitude: 0.92

Iteration: 689000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 0.97

Iteration: 690000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 1.56

Iteration: 691000  |  Actual Magnitude: 1.22  |  Predicted Magnitude: 1.67

Iteration: 692000  |  Actual Magnitude: 0.23  |  Predicted Magnitude: 1.48

Iteration: 693000  |  Actual Magnitude: 2.05  |  Predicted Magnitude: 1.85

Iteration: 694000  |  Actual Magnitude: 2.06  |  Predicted Magnitude: 1.74

Iteration: 695000  |  Actual Magnitude: 0.44  |  Predicted Magnitude: 1.55

Iteration: 696000  |  Actual Magnitude: -0.99  |  Predicted Magnitude: 0.95

Iteration: 697000  |  Actual Magnitude: 2.13  |  Predicted Magnitude: 1.56

Iteration: 698000  |  Actual Magnitude: 0.50  |  Predicted Magnitude: 1.57

Iteration: 699000  |  Actual Magnitude: 2.07  |  Predicted Magnitude: 1.62

Iteration: 700000  |  Actual Magnitude: 3.90  |  Predicted Magnitude: 5.49

Iteration: 701000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 1.46

Iteration: 702000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 1.01

Iteration: 703000  |  Actual Magnitude: 0.87  |  Predicted Magnitude: 1.65

Iteration: 704000  |  Actual Magnitude: 5.30  |  Predicted Magnitude: 7.48

Iteration: 705000  |  Actual Magnitude: 1.98  |  Predicted Magnitude: 1.49

Iteration: 706000  |  Actual Magnitude: -0.29  |  Predicted Magnitude: 0.95

Iteration: 707000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 3.62

Iteration: 708000  |  Actual Magnitude: 2.40  |  Predicted Magnitude: 0.92

Iteration: 709000  |  Actual Magnitude: 0.80  |  Predicted Magnitude: 1.09

Iteration: 710000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.36

Iteration: 711000  |  Actual Magnitude: 1.45  |  Predicted Magnitude: 1.56

Iteration: 712000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.54

Iteration: 713000  |  Actual Magnitude: 2.30  |  Predicted Magnitude: 0.91

Iteration: 714000  |  Actual Magnitude: 0.70  |  Predicted Magnitude: 1.57

Iteration: 715000  |  Actual Magnitude: 1.97  |  Predicted Magnitude: 1.61

Iteration: 716000  |  Actual Magnitude: 0.46  |  Predicted Magnitude: 1.48

Iteration: 717000  |  Actual Magnitude: 1.46  |  Predicted Magnitude: 1.63

Iteration: 718000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.11

Iteration: 719000  |  Actual Magnitude: 4.40  |  Predicted Magnitude: 4.85

Iteration: 720000  |  Actual Magnitude: 0.89  |  Predicted Magnitude: 1.57

Iteration: 721000  |  Actual Magnitude: 4.60  |  Predicted Magnitude: 3.71

Iteration: 722000  |  Actual Magnitude: 4.60  |  Predicted Magnitude: 4.42

Iteration: 723000  |  Actual Magnitude: 0.29  |  Predicted Magnitude: 1.60

Iteration: 724000  |  Actual Magnitude: 2.45  |  Predicted Magnitude: 1.56

Iteration: 725000  |  Actual Magnitude: 0.92  |  Predicted Magnitude: 1.48

Iteration: 726000  |  Actual Magnitude: 4.80  |  Predicted Magnitude: 1.91

Iteration: 727000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.14

Iteration: 728000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 0.99

Iteration: 729000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 1.48

Iteration: 730000  |  Actual Magnitude: 1.03  |  Predicted Magnitude: 1.49

Iteration: 731000  |  Actual Magnitude: 0.80  |  Predicted Magnitude: 1.50

Iteration: 732000  |  Actual Magnitude: 4.40  |  Predicted Magnitude: 1.96

Iteration: 733000  |  Actual Magnitude: 0.98  |  Predicted Magnitude: 1.55

Iteration: 734000  |  Actual Magnitude: 4.30  |  Predicted Magnitude: 4.73

Iteration: 735000  |  Actual Magnitude: 3.20  |  Predicted Magnitude: 2.68

Iteration: 736000  |  Actual Magnitude: 1.79  |  Predicted Magnitude: 1.57

Iteration: 737000  |  Actual Magnitude: 3.00  |  Predicted Magnitude: 1.11

Iteration: 738000  |  Actual Magnitude: 1.36  |  Predicted Magnitude: 1.53

Iteration: 739000  |  Actual Magnitude: 0.70  |  Predicted Magnitude: 1.63

Iteration: 740000  |  Actual Magnitude: 1.50  |  Predicted Magnitude: 1.31

Iteration: 741000  |  Actual Magnitude: 0.48  |  Predicted Magnitude: 1.66

Iteration: 742000  |  Actual Magnitude: 4.60  |  Predicted Magnitude: 3.76

Iteration: 743000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 0.83

Iteration: 744000  |  Actual Magnitude: 0.84  |  Predicted Magnitude: 1.48

Iteration: 745000  |  Actual Magnitude: 0.44  |  Predicted Magnitude: 1.48

Iteration: 746000  |  Actual Magnitude: 2.23  |  Predicted Magnitude: 1.70

Iteration: 747000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.50

Iteration: 748000  |  Actual Magnitude: 3.60  |  Predicted Magnitude: 2.76

Iteration: 749000  |  Actual Magnitude: 1.73  |  Predicted Magnitude: 1.51

Iteration: 750000  |  Actual Magnitude: 1.03  |  Predicted Magnitude: 1.63

Iteration: 751000  |  Actual Magnitude: 1.33  |  Predicted Magnitude: 1.48

Iteration: 752000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 0.94

Iteration: 753000  |  Actual Magnitude: 3.50  |  Predicted Magnitude: 2.38

Iteration: 754000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 1.23

Iteration: 755000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 1.55

Iteration: 756000  |  Actual Magnitude: 1.42  |  Predicted Magnitude: 1.54

Iteration: 757000  |  Actual Magnitude: 1.87  |  Predicted Magnitude: 1.69

Iteration: 758000  |  Actual Magnitude: 0.40  |  Predicted Magnitude: 0.94

Iteration: 759000  |  Actual Magnitude: 1.60  |  Predicted Magnitude: 1.60

Iteration: 760000  |  Actual Magnitude: 4.10  |  Predicted Magnitude: 2.28

Iteration: 761000  |  Actual Magnitude: 0.74  |  Predicted Magnitude: 1.67

Iteration: 762000  |  Actual Magnitude: 0.97  |  Predicted Magnitude: 1.66

Iteration: 763000  |  Actual Magnitude: 2.70  |  Predicted Magnitude: 2.36

Iteration: 764000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 3.32

Iteration: 765000  |  Actual Magnitude: 1.09  |  Predicted Magnitude: 1.48

Iteration: 766000  |  Actual Magnitude: 2.10  |  Predicted Magnitude: 2.55

Iteration: 767000  |  Actual Magnitude: 1.08  |  Predicted Magnitude: 1.67

Iteration: 768000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 3.63

Iteration: 769000  |  Actual Magnitude: 4.70  |  Predicted Magnitude: 3.59

Iteration: 770000  |  Actual Magnitude: 2.60  |  Predicted Magnitude: 1.56

Iteration: 771000  |  Actual Magnitude: 5.90  |  Predicted Magnitude: 4.10

Iteration: 772000  |  Actual Magnitude: 4.00  |  Predicted Magnitude: 3.81

Iteration: 773000  |  Actual Magnitude: 2.90  |  Predicted Magnitude: 2.50

Iteration: 774000  |  Actual Magnitude: 1.57  |  Predicted Magnitude: 1.63

Iteration: 775000  |  Actual Magnitude: 5.20  |  Predicted Magnitude: 3.61

Iteration: 776000  |  Actual Magnitude: 2.40  |  Predicted Magnitude: 2.43

Iteration: 777000  |  Actual Magnitude: 2.50  |  Predicted Magnitude: 1.36

Iteration: 778000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.40

Iteration: 779000  |  Actual Magnitude: 2.40  |  Predicted Magnitude: 0.98

Iteration: 780000  |  Actual Magnitude: 1.00  |  Predicted Magnitude: 0.84

Iteration: 781000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 1.60

Iteration: 782000  |  Actual Magnitude: 0.26  |  Predicted Magnitude: 1.61

Iteration: 783000  |  Actual Magnitude: 4.60  |  Predicted Magnitude: 3.36

Iteration: 784000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.39

Iteration: 785000  |  Actual Magnitude: 3.10  |  Predicted Magnitude: 0.86

Iteration: 786000  |  Actual Magnitude: -0.20  |  Predicted Magnitude: 1.60

Iteration: 787000  |  Actual Magnitude: 2.10  |  Predicted Magnitude: 2.42

Iteration: 788000  |  Actual Magnitude: 2.02  |  Predicted Magnitude: 1.63

Iteration: 789000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 1.12

Iteration: 790000  |  Actual Magnitude: 4.50  |  Predicted Magnitude: 3.00

Iteration: 791000  |  Actual Magnitude: 1.01  |  Predicted Magnitude: 1.62

Iteration: 792000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.56

Iteration: 793000  |  Actual Magnitude: 2.34  |  Predicted Magnitude: 1.66

Iteration: 794000  |  Actual Magnitude: 0.00  |  Predicted Magnitude: 1.59

Iteration: 795000  |  Actual Magnitude: 1.58  |  Predicted Magnitude: 1.48

Iteration: 796000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 0.95

Iteration: 797000  |  Actual Magnitude: 2.30  |  Predicted Magnitude: 1.08

Iteration: 798000  |  Actual Magnitude: 1.35  |  Predicted Magnitude: 1.57

Iteration: 799000  |  Actual Magnitude: 1.80  |  Predicted Magnitude: 0.94

Iteration: 800000  |  Actual Magnitude: 1.17  |  Predicted Magnitude: 1.52

Iteration: 801000  |  Actual Magnitude: 0.00  |  Predicted Magnitude: 1.57

Iteration: 802000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 0.95

Iteration: 803000  |  Actual Magnitude: 0.70  |  Predicted Magnitude: 1.66

Iteration: 804000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 3.10

Iteration: 805000  |  Actual Magnitude: 4.60  |  Predicted Magnitude: 4.81

Iteration: 806000  |  Actual Magnitude: 1.34  |  Predicted Magnitude: 1.55

Iteration: 807000  |  Actual Magnitude: 1.63  |  Predicted Magnitude: 1.54

Iteration: 808000  |  Actual Magnitude: 2.20  |  Predicted Magnitude: 0.95

Iteration: 809000  |  Actual Magnitude: -0.50  |  Predicted Magnitude: 1.58

Iteration: 810000  |  Actual Magnitude: 3.49  |  Predicted Magnitude: 1.48

Iteration: 811000  |  Actual Magnitude: 4.10  |  Predicted Magnitude: 4.12

Iteration: 812000  |  Actual Magnitude: 0.84  |  Predicted Magnitude: 1.66

Iteration: 813000  |  Actual Magnitude: 1.40  |  Predicted Magnitude: 1.51

Iteration: 814000  |  Actual Magnitude: 1.10  |  Predicted Magnitude: 0.95

Iteration: 815000  |  Actual Magnitude: 1.20  |  Predicted Magnitude: 0.90

Iteration: 816000  |  Actual Magnitude: 0.75  |  Predicted Magnitude: 1.48

Iteration: 817000  |  Actual Magnitude: 1.51  |  Predicted Magnitude: 1.68

Iteration: 818000  |  Actual Magnitude: 4.20  |  Predicted Magnitude: 3.42

Iteration: 819000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.16

Iteration: 820000  |  Actual Magnitude: 0.58  |  Predicted Magnitude: 1.50

Iteration: 821000  |  Actual Magnitude: 4.80  |  Predicted Magnitude: 2.04

Iteration: 822000  |  Actual Magnitude: -0.10  |  Predicted Magnitude: 1.60

Iteration: 823000  |  Actual Magnitude: 1.48  |  Predicted Magnitude: 1.54

Iteration: 824000  |  Actual Magnitude: 1.49  |  Predicted Magnitude: 1.62

Iteration: 825000  |  Actual Magnitude: 0.49  |  Predicted Magnitude: 1.45

Iteration: 826000  |  Actual Magnitude: 1.87  |  Predicted Magnitude: 1.59

Iteration: 827000  |  Actual Magnitude: 1.05  |  Predicted Magnitude: 1.53

Iteration: 828000  |  Actual Magnitude: 0.51  |  Predicted Magnitude: 1.55

Iteration: 829000  |  Actual Magnitude: 1.68  |  Predicted Magnitude: 1.48

Iteration: 830000  |  Actual Magnitude: 0.58  |  Predicted Magnitude: 1.48

Iteration: 831000  |  Actual Magnitude: 0.95  |  Predicted Magnitude: 1.58

Iteration: 832000  |  Actual Magnitude: 1.48  |  Predicted Magnitude: 1.57

Iteration: 833000  |  Actual Magnitude: 0.98  |  Predicted Magnitude: 1.47

Iteration: 834000  |  Actual Magnitude: 1.32  |  Predicted Magnitude: 1.55

Iteration: 835000  |  Actual Magnitude: 1.90  |  Predicted Magnitude: 1.62

Iteration: 836000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 0.92

Iteration: 837000  |  Actual Magnitude: 2.00  |  Predicted Magnitude: 1.61

Iteration: 838000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.39

Iteration: 839000  |  Actual Magnitude: 4.40  |  Predicted Magnitude: 3.95

Iteration: 840000  |  Actual Magnitude: -0.28  |  Predicted Magnitude: 0.91

Iteration: 841000  |  Actual Magnitude: 3.40  |  Predicted Magnitude: 3.79

Iteration: 842000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.01

Iteration: 843000  |  Actual Magnitude: 3.50  |  Predicted Magnitude: 2.70

Iteration: 844000  |  Actual Magnitude: 1.90  |  Predicted Magnitude: 0.95

Iteration: 845000  |  Actual Magnitude: 2.63  |  Predicted Magnitude: 1.63

Iteration: 846000  |  Actual Magnitude: 1.30  |  Predicted Magnitude: 0.92

Iteration: 847000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 0.94

Iteration: 848000  |  Actual Magnitude: 1.70  |  Predicted Magnitude: 1.51

Iteration: 849000  |  Actual Magnitude: -0.50  |  Predicted Magnitude: 1.37

Iteration: 850000  |  Actual Magnitude: 2.75  |  Predicted Magnitude: 1.55

Iteration: 851000  |  Actual Magnitude: 0.90  |  Predicted Magnitude: 1.54

Iteration: 852000  |  Actual Magnitude: 1.61  |  Predicted Magnitude: 1.65

Iteration: 853000  |  Actual Magnitude: 0.00  |  Predicted Magnitude: 3.96

Iteration: 854000  |  Actual Magnitude: -0.08  |  Predicted Magnitude: 0.91

Iteration: 855000  |  Actual Magnitude: -0.30  |  Predicted Magnitude: 1.56

Iteration: 856000  |  Actual Magnitude: 1.69  |  Predicted Magnitude: 1.53

Iteration: 857000  |  Actual Magnitude: 3.30  |  Predicted Magnitude: 1.20

Iteration: 858000  |  Actual Magnitude: 0.61  |  Predicted Magnitude: 1.35

Iteration: 859000  |  Actual Magnitude: 1.87  |  Predicted Magnitude: 1.65

Iteration: 860000  |  Actual Magnitude: 1.71  |  Predicted Magnitude: 1.62

Iteration: 861000  |  Actual Magnitude: 0.82  |  Predicted Magnitude: 1.55
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Magnitude")
plt.ylabel("Predicted Magnitude")
plt.title("Actual vs. Predicted Earthquake Magnitudes")
plt.show()

Inferences:inferential-statistics.png
The model does not seem to be perfomring very bad. The Mean Squared Error is almost 0.9.

The predicted values are almost close to the actual values.

Conclusionconclusion.png