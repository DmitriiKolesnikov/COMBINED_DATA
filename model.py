import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Загрузка данных
df = pd.read_excel('/Users/jimsgood/Desktop/маргу.xlsx')

# Обработка пропущенных данных
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Преобразование целевой переменной в числовой формат
df['Отчислен'] = df['Отчислен'].apply(lambda x: 1 if x == 'да' else 0)

# Выделение признаков и целевой переменной
X = df.drop(['ФИО', 'Отчислен', 'Адрес по прописке'], axis=1)
y = df['Отчислен']

# Обработка числовых и категориальных данных
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Уравновешивание классов с помощью SMOTE
smote = SMOTE(random_state=42)

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Применение SMOTE к тренировочным данным
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Создание и настройка модели с PCA
pca = PCA(n_components=0.95)  # Сохранить 95% дисперсии
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', pca),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Настройка гиперпараметров
param_grid = {
    'classifier__n_estimators': [100, 200, 300, 400, 500],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Поиск лучших гиперпараметров с использованием StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

# Вывод лучших параметров
print(f'Лучшие параметры: {grid_search.best_params_}')

# Прогнозирование и оценка модели на тестовых данных
y_pred = grid_search.predict(X_test)

# Оценка качества модели
print(f'Точность модели: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis")
plt.title("Матрица ошибок")
plt.xlabel("Предсказанные метки")
plt.ylabel("Истинные метки")
plt.show()