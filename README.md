Here's a concise `README.md` file based on the provided code:

```markdown
# Restaurant Rating Prediction

This project involves building a machine learning model to predict the aggregate rating of a restaurant based on various features.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset

The dataset contains information about various restaurants, including features such as restaurant name, location, cuisines, price range, and ratings.

## Data Preprocessing

1. **Remove Null Values**:
    ```python
    df = df.dropna()
    ```

2. **Drop Irrelevant Columns**:
    ```python
    df = df.drop(['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address', 'Locality', 'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines', 'Currency'], axis=1)
    ```

3. **Encode Categorical Features**:
    ```python
    from sklearn.preprocessing import LabelEncoder
    lbe = LabelEncoder()
    categorical_features = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Rating color', 'Rating text']
    for feature in categorical_features:
        df[feature] = lbe.fit_transform(df[feature])
    ```

## Exploratory Data Analysis

1. **Aggregate Rating Distribution**:
    ```python
    df['Aggregate rating'].value_counts().plot(kind='pie', autopct='%.3f')
    ```

2. **Rating Distribution**:
    ```python
    sns.distplot(df['Aggregate rating'])
    ```

3. **Scatter Plot**:
    ```python
    sns.scatterplot(x=df['Aggregate rating'], y=df['Votes'], hue=df['Price range'])
    ```

4. **Correlation Heatmap**:
    ```python
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation between attributes")
    plt.show()
    ```

## Model Training and Evaluation

1. **Split the Data**:
    ```python
    from sklearn.model_selection import train_test_split
    x = df.drop('Aggregate rating', axis=1)
    y = df['Aggregate rating']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=250)
    ```

2. **Linear Regression**:
    ```python
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_prediction = lr.predict(x_test)
    ```

3. **Decision Tree Regressor**:
    ```python
    from sklearn.tree import DecisionTreeRegressor
    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train)
    dt_prediction = dt.predict(x_test)
    ```

4. **Performance Metrics**:
    ```python
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    lr_mae = mean_absolute_error(y_test, lr_prediction)
    lr_mse = mean_squared_error(y_test, lr_prediction)
    lr_r2 = r2_score(y_test, lr_prediction)
    dt_mae = mean_absolute_error(y_test, dt_prediction)
    dt_mse = mean_squared_error(y_test, dt_prediction)
    dt_r2 = r2_score(y_test, dt_prediction)
    ```

## Results

- **Linear Regression**:
  - Mean Absolute Error: 0.36
  - Mean Squared Error: 0.23
  - R² Score: 0.57

- **Decision Tree Regressor**:
  - Mean Absolute Error: 0.15
  - Mean Squared Error: 0.05
  - R² Score: 0.98

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This `README.md` file provides a clear overview of the project, including installation instructions, dataset details, preprocessing steps, exploratory data analysis, model training and evaluation, and the results.
