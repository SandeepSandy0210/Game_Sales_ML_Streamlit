import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (replace 'videogame_sales.csv' with your dataset file path)
@st.cache_data
def load_data():
    data = pd.read_csv('vgsales.csv')
    return data

# Main function to create the Streamlit app
def main():
    st.title("Video Game Sales Analysis\n Sales in millions")
    st.sidebar.title("Select Options")

    data = load_data()
    data.drop(columns=['Name', 'Rank'], inplace=True)

    # Fill null/Nan values in 'Year' with the mean of existing years (rounded to nearest year)
    mean_year = round(data['Year'].mean())
    data['Year'].fillna(mean_year, inplace=True)

    # Categorize publishers with less than 50 games as "Small Publishers"
    publisher_counts = data['Publisher'].value_counts()
    small_publishers = publisher_counts[publisher_counts < 50].index
    data.loc[data['Publisher'].isin(small_publishers), 'Publisher'] = 'Small Publisher'

    # Sidebar: Select columns for analysis
    selected_x_column = st.sidebar.selectbox(
        "Select X-Axis Column",
        ["Publisher", "Platform", "Genre", 'Year']
    )

    selected_y_column = st.sidebar.selectbox(
        "Select Y-Axis Column",
        ["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    )

    # Filter data based on selected columns
    filtered_data = data[[selected_x_column, selected_y_column]]

    # Group the data by the selected x-axis column and sum the y-axis column
    grouped_data = filtered_data.groupby(selected_x_column)[selected_y_column].sum().reset_index()

    # Display the bar graph
    st.subheader(f"Bar Graph: {selected_y_column} by {selected_x_column}")
    plt.figure(figsize=(12, 6))
    plt.bar(grouped_data[selected_x_column], grouped_data[selected_y_column])
    plt.xlabel(selected_x_column)
    plt.ylabel(selected_y_column)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(plt)

    # Display the data as a table
    st.subheader("Data Table")
    st.dataframe(grouped_data)

if __name__ == "__main__":
    main()
