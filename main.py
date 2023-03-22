import pandas as pd
import numpy as np

# Load Titanic dataset into a Pandas DataFrame
titanic_df = pd.read_csv('train.csv')
# Reorder columns
titanic_df = titanic_df[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 
                         'Fare', 'Embarked', 'Cabin', 'SibSp', 'Parch', 'Ticket']]
titanic_df = titanic_df.dropna(how='any', subset=['Age', 'Embarked'])
titanic_df = titanic_df.drop(columns=['Name', 'Ticket', 'Cabin'])
titanic_df['is_male'] = (titanic_df['Sex'] == 'male').astype(int)

titanic_df['age_n'] = titanic_df['Age']/titanic_df['Age'].max()

#Create a log10 column based on fare price
titanic_df['fare_log'] = np.log10(titanic_df['Fare'] + 1)

# Create two new columns 'pclass_1' and 'pclass_2'
titanic_df['pclass_1'] = titanic_df['Pclass'].apply(lambda x: 1 if x==1 else 0)
titanic_df['pclass_2'] = titanic_df['Pclass'].apply(lambda x: 1 if x==2 else 0)

# Create two new columns 'embark_s' and 'embark_c'
titanic_df['embark_s'] = titanic_df['Embarked'].apply(lambda x: 1 if x=='S' else 0)
titanic_df['embark_c'] = titanic_df['Embarked'].apply(lambda x: 1 if x=='C' else 0)

# Create new column with ones
titanic_df['ones'] = 1

# Create a random matrix
matrix = np.random.rand(10) - 0.5

columns = titanic_df.loc[:, 'SibSp':'ones'].to_numpy()

# Extract 'Age' and 'SibSp' columns to form matrix2
#matrix2 = titanic_df[['Age', 'SibSp']].to_numpy()

# Perform matrix multiplication
result = np.dot(matrix.reshape((1,-1)), columns.T)

# Extract the 'Survived' column from titanic_df and store it in a numpy array
survived = titanic_df.loc[:, 'Survived'].to_numpy()

# Calculate the squared difference between the result and the 'Survived' array
squared_diff = (result.flatten() - survived) ** 2

# Print the squared difference array
print(squared_diff)

# Calculate the average loss
avg_loss = np.mean(squared_diff)

# Print the average loss
print("The average loss is:", avg_loss)
# Print the result
#print(titanic_df)
#print(result)
#print(result)