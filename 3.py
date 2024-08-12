import pandas as pd

file1 = '/Users/jimsgood/Downloads/оценки.xlsx'
df = pd.read_excel(file1)

df['Оценка (успеваемость)'] = df['Оценка (успеваемость)'].replace({
    'Хорошо': 4,
    'Удовлетворительно': 3,
    'зачтено': 5,
    'Отлично': 5,
    'Неудовлетворительно': 2,
    'Неявка': 1,
    'не зачтено': 2,
    'Неявка по ув.причине': 1,
    'Не допущен': 1
})
df['Оценка (успеваемость)'] = df['Оценка (успеваемость)'].fillna(0)
df['Key'] = (df['Полугодие'] != df['Полугодие'].shift(1)) | (df['ФИО'] != df['ФИО'].shift(1))
df['Номер'] = df.groupby('ФИО')['Key'].cumsum()
df.drop(columns=['Key'], inplace=True)
column_to_drop = ['Учебный год', 'Полугодие']
df = df.drop(columns=column_to_drop, errors='ignore')
df = df.rename(columns={'Номер': 'Полугодие'})
df['Средний балл'] = df.groupby(['ФИО', 'Полугодие'])['Оценка (успеваемость)'].transform('mean') 
df = df.drop(columns='Оценка (успеваемость)', errors='ignore')
df = df.loc[(df['Полугодие'] != df['Полугодие'].shift()) | (df['Средний балл'] != df['Средний балл'].shift())]
df = df.pivot(index='ФИО', columns='Полугодие', values='Средний балл').fillna(0)

# Переименование столбцов для более удобного отображения полугодий
df.columns = ['Полугодие_' + str(col) for col in df.columns]

# Сброс индекса для сохранения ФИО как столбца
df.reset_index(inplace=True)
df.to_excel('оценки_мисис.xlsx', index=False)
