import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import altair as alt
from sklearn.metrics import mean_absolute_error




# Load the preprocessed BoT IoT dataset
df = pd.read_csv(r'C:\Sai Nikhitha\Data Science\ML Real Estate Price Prediction\Updated_hp.csv')


  
# df4=df3.copy()
# df4.total_sqft=df4.total_sqft.apply(convert_sqft_to_num)
# df4=df4[df4.total_sqft.notnull()]

# df5=df4.copy()
# df5['Pricepersqft']=df5['price']*100000/df5['total_sqft']

# df5.location=df5.location.apply(lambda x:x.strip()) #Strip removes any leadig and trailing charecters
# locationcount=df5['location'].value_counts(ascending=False)

# uniqueloc=len(locationcount)

# loc_lessthan10=locationcount[locationcount<=10]

# uniqueloc=len(locationcount)

# df5.location=df5['location'].apply(lambda x: 'other' if x in loc_lessthan10 else x)
# len(df5.location.unique())

# df6=df5[~(df5.total_sqft/df5.bhk<300)]

# from pandas import Grouper
# df6grp=df6.groupby('location')
# df6grpdf=pd.DataFrame(df6grp)

# import numpy as np
# from pandas import concat
# def remove_pricepersqft_outliers(df):
#   df_out=pd.DataFrame()
#   for key,subdf in df.groupby('location'):
#     mean1=np.mean(subdf.Pricepersqft)
#     stdev=np.std(subdf.Pricepersqft)
#     reduced_df=subdf[(subdf.Pricepersqft>(mean1-stdev)) & (subdf.Pricepersqft<(mean1+stdev))]
#     df_out=pd.concat([df_out,reduced_df],ignore_index=True)
#   return df_out

# df7=remove_pricepersqft_outliers(df6)

# def remove_bhk_outliers(df):
#     exclude_indices = np.array([])
#     for location, location_df in df.groupby('location'):
#         bhk_stats = {}
#         for bhk, bhk_df in location_df.groupby('bhk'):
#             bhk_stats[bhk] = {
#                 'mean': np.mean(bhk_df.Pricepersqft),
#                 'std': np.std(bhk_df.Pricepersqft),
#                 'count': bhk_df.shape[0]
#             }
#         for bhk, bhk_df in location_df.groupby('bhk'):
#             stats = bhk_stats.get(bhk-1)
#             if stats and stats['count']>5:
#                 exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.Pricepersqft<(stats['mean'])].index.values)
#     return df.drop(exclude_indices,axis='index')
    
# df8 = remove_bhk_outliers(df7)

# df9=df8[df8.bath<df8.bhk+2]

# df10=df9.drop(['size','Pricepersqft'],axis=1)

# X=df10.drop('price',axis=1)
# y=df10.price


# Define the sidebar
st.sidebar.header("Dashboard Options")
feature_choices = st.sidebar.multiselect("Select Features", ['location','total_sqft','bath','bhk'])

# Define the main content
st.title("House Price Prediction Dashboard")
st.header("Predicting price of a house")

if len(feature_choices) > 0:
    st.header("Selected Features")
    st.write(feature_choices)
    selected_data = df[feature_choices + ["price"]]

    # Convert categorical features to numerical using LabelEncoder
    le = LabelEncoder()
    selected_data["location"] = le.fit_transform(selected_data["location"])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(selected_data[feature_choices], selected_data["price"], test_size=0.2, random_state=42)

    # Train and test a random forest classifier
    from sklearn.linear_model import LinearRegression
    regclf=LinearRegression()
    regclf.fit(X_train,y_train)
    y_pred = regclf.predict(X_test)

    chart = alt.Chart(df).mark_circle().encode(
    x='location',
    y='price'
   )

    st.altair_chart(chart, use_container_width=True)


    mae = mean_absolute_error(y_test, y_pred)

    # display the MAE
    st.write("Mean Absolute Error:", mae)

    # # Display the model's accuracy and classification report
    # accuracy = accuracy_score(y_test, y_pred)
    # st.header("Model Performance")
    # st.write("Accuracy: {:.2f}%".format(accuracy * 100))
    # st.write("Classification Report:")
    # st.write(classification_report(y_test, y_pred))

    # # Display the predictions for the full dataset
    selected_data["predicted_proce"] = regclf.predict(selected_data[feature_choices])
    st.header("Predictions")
    st.write(selected_data)
# # Display the breakdown of categories for predicted attacks
#     st.header("Breakdown by Category")
#     categories = selected_data.groupby(["predicted_attack", "category"]).size().unstack(fill_value=0)
#     st.write(categories)

else:
    st.warning("Please select at least one feature.")