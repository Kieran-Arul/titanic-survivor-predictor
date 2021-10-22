from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("titanic.csv")

# ------------------------- Exploratory Analysis -------------------------

# Check for nulls
# A fair number of passenger age values missing
# Almost all cabin values missing
# 2 missing values in the embarked column
print(df.isnull().sum())

sns.set_style("whitegrid")

# Mostly men died
sns.countplot(x="Survived", data=df, hue="Sex")
# plt.show()

# Mostly people in lower class cabins died
sns.countplot(x="Survived", data=df, hue="Pclass")
# plt.show()

# Most people were single
plt.figure(figsize=(10, 4))
sns.countplot(x="SibSp", data=df)
# plt.show()

# Obtain mean ages to fill in missing ages
# 1st class mean age -> about 37
# 2nd class mean age -> about 29
# 3rd class mean age -> about 24
sns.boxplot(x="Pclass", y="Age", data=df)
# plt.show()

# ------------------------- Data cleaning -------------------------

# Function to fill in nulls
def impute_age(cols):

    age = cols[0]
    p_class = cols[1]

    if pd.isnull(age):

        if p_class == 1:
            return 37

        elif p_class == 2:
            return 29

        else:
            return 24

    else:
        return age


# Fill in nulls
df["Age"] = df[["Age", "Pclass"]].apply(impute_age, axis=1)

# Too many missing cabin values
df.drop("Cabin", axis=1, inplace=True)

# Drop the 2 people with missing embarked value
df.dropna(inplace=True)

# One-hot encode sex column
sex = pd.get_dummies(df["Sex"], drop_first=True)

# One-hot encode embarked column
embark = pd.get_dummies(df["Embarked"], drop_first=True)

# Join one-hot encoded dfs with main df
df = pd.concat([df, sex, embark], axis=1)

# Drop redundant columns
df.drop(["Sex", "Embarked", "Name", "Ticket", "PassengerId"], axis=1, inplace=True)

# ------------------------- Model Building -------------------------

# Features
X = df.drop("Survived", axis=1)

# Target
y = df["Survived"]

# TTS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------- Model Evaluation -------------------------

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
