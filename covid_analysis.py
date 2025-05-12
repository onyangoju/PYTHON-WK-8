import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Data Collection & Loading
# Load the dataset
df = pd.read_csv(r'C:\Users\Junior\Desktop\COVID 19 TRACKER\owid-covid-data.csv')

# 2️⃣ Data Exploration
# Display the first few rows to understand the structure of the data
print("First 5 rows of the data:")
print(df.head())

# Display the column names
print("\nColumn names:")
print(df.columns)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3️⃣ Data Cleaning
# Filter the dataset for countries of interest (e.g., Kenya, USA, India)
countries_of_interest = ['Kenya', 'USA', 'India']
df_filtered = df[df['location'].isin(countries_of_interest)]

# Drop rows with missing critical values (like dates or cases)
df_filtered = df_filtered.dropna(subset=['date', 'total_cases'])

# Convert the 'date' column to datetime
df_filtered['date'] = pd.to_datetime(df_filtered['date'])

# Handle missing numeric values (optional: fill with zeros or interpolate)
df_filtered['total_cases'] = df_filtered['total_cases'].fillna(0)
df_filtered['total_deaths'] = df_filtered['total_deaths'].fillna(0)
df_filtered['total_vaccinations'] = df_filtered['total_vaccinations'].fillna(0)

# Preview the cleaned data
print("\nCleaned Data (First 5 rows):")
print(df_filtered.head())

# 4️⃣ Exploratory Data Analysis (EDA)

# Plot total cases over time for India
india_data = df_filtered[df_filtered['location'] == 'India']

plt.figure(figsize=(10, 6))
plt.plot(india_data['date'], india_data['total_cases'], label='Total Cases', color='blue')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.title('COVID-19 Total Cases in India over Time')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot total deaths over time for India
plt.figure(figsize=(10, 6))
plt.plot(india_data['date'], india_data['total_deaths'], label='Total Deaths', color='red')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.title('COVID-19 Total Deaths in India over Time')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 5️⃣ Visualizing Vaccination Progress

# Plot cumulative vaccinations over time for India
plt.figure(figsize=(10, 6))
plt.plot(india_data['date'], india_data['total_vaccinations'], label='Total Vaccinations', color='green')
plt.xlabel('Date')
plt.ylabel('Total Vaccinations')
plt.title('COVID-19 Vaccinations in India over Time')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Compare % of vaccinated population between selected countries
df_vaccinations = df_filtered[df_filtered['location'].isin(countries_of_interest)]

# Calculate vaccination percentage for each country
df_vaccinations['vaccination_percentage'] = df_vaccinations['total_vaccinations'] / df_vaccinations['population'] * 100

# Plot the vaccination percentage for each country over time
plt.figure(figsize=(10, 6))
for country in countries_of_interest:
    country_data = df_vaccinations[df_vaccinations['location'] == country]
    plt.plot(country_data['date'], country_data['vaccination_percentage'], label=country)

plt.xlabel('Date')
plt.ylabel('Vaccination Percentage (%)')
plt.title('Vaccination Progress in Selected Countries')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6️⃣ Compare Daily New Cases Between Countries
plt.figure(figsize=(10, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['location'] == country]
    plt.plot(country_data['date'], country_data['new_cases'], label=country)

plt.xlabel('Date')
plt.ylabel('Daily New Cases')
plt.title('Daily New COVID-19 Cases in Selected Countries')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7️⃣ Compare Total Cases and Deaths in Each Country
latest_data = df_filtered.sort_values('date').groupby('location').tail(1)

countries = latest_data['location']
total_cases = latest_data['total_cases']
total_deaths = latest_data['total_deaths']

x = range(len(countries))
width = 0.4

plt.figure(figsize=(10, 6))
plt.bar(x, total_cases, width=width, label='Total Cases', color='orange')
plt.bar([i + width for i in x], total_deaths, width=width, label='Total Deaths', color='brown')

plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Total Cases vs Total Deaths per Country')
plt.xticks([i + width / 2 for i in x], countries)
plt.legend()
plt.tight_layout()
plt.show()

# 8️⃣ Compare New Daily Cases Over Time
plt.figure(figsize=(10, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['location'] == country]
    plt.plot(country_data['date'], country_data['new_cases'], label=country)

plt.xlabel('Date')
plt.ylabel('New Daily Cases')
plt.title('New Daily COVID-19 Cases in Selected Countries')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9️⃣ Compare Total Deaths per Million (Population-adjusted)
plt.figure(figsize=(10, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['location'] == country]
    plt.plot(country_data['date'], country_data['total_deaths_per_million'], label=country)

plt.xlabel('Date')
plt.ylabel('Total Deaths per Million')
plt.title('COVID-19 Total Deaths per Million People')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import seaborn as sns  # Make sure seaborn is installed

# Select relevant numerical columns for correlation
corr_columns = ['total_cases', 'total_deaths', 'total_vaccinations',
                'stringency_index', 'population_density', 'gdp_per_capita',
                'diabetes_prevalence', 'life_expectancy']

# Drop rows with NaNs in selected columns
df_corr = df_filtered[corr_columns].dropna()

# Compute correlation matrix
correlation_matrix = df_corr.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix of COVID-19 Indicators')
plt.tight_layout()
plt.show()

# 6️⃣ Exporting Cleaned Data (Optional Final Step)
output_path = r'C:\Users\Junior\Desktop\COVID 19 TRACKER\cleaned_covid_data.csv'
df_filtered.to_csv(output_path, index=False)
print(f"\n✅ Cleaned data saved to: {output_path}")
