from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix

#Se lee el csv y se almacena en un dataframe
df = pd.read_csv('data.csv')

'''Se debe tomar en cuenta que la columna "class" contendrá "P" = Positivo para alzheimer
"H" = negativo para alzheimer '''

# Se observan los nombres de las columnas
#print(df.columns)

''' Se puede observar en la terminal que ID es solo la identificacion del paciente, por lo que no agrega
nada al analisis. Sin embargo, considero que los demas features pueden ser muy utiles ya que corresponden
al tiempo de escritura, promedio de su presion, tiempo en el que estuvo el lapiz abajo, etc '''

# Se eliminan las columnas que no voy a utilizar y las restantes se almacenan en X, que seran mis features

X = df[['gmrt_on_paper1', 'paper_time2', 'mean_gmrt4',
        'air_time5', 'air_time6', 'paper_time8', 'mean_gmrt6', 'mean_gmrt7', 'mean_speed_on_paper7', 'total_time10', 'mean_gmrt10', 'gmrt_on_paper10',
        'mean_gmrt11', 'pressure_mean10', 'air_time18', 'total_time18', 'total_time13', 'gmrt_on_paper14', 'mean_speed_on_paper14', 'mean_speed_on_paper15',
        'air_time16', 'num_of_pendown16', 'mean_gmrt16', 'mean_acc_in_air17', 'max_x_extension18', 'mean_speed_on_paper18', 'gmrt_on_paper18', 'mean_speed_in_air19',
        'gmrt_in_air19', 'gmrt_in_air25', 'mean_speed_in_air25', 'air_time21', 'total_time21', 'gmrt_on_paper22', 'mean_speed_on_paper22', 'gmrt_in_air23', 'mean_gmrt23',
        'paper_time9', 'gmrt_on_paper15']]

y = (df['class'] == 'P').astype(int)
y = y.to_numpy()

''' Se divide el dataset, 90% para entrenar, 10% para probar, stratify= y se usa para mantener
    las mismas proporciones de clases en los conjuntos divididos.'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10, stratify=y)


''' Se crea una instancia del Random Forest Classifier, utilizando gini, un número de estimadores de 5
    y se utilizan dos núcleos para mejorar la eficiencia'''
forest = RandomForestClassifier(criterion='gini', n_estimators=5, random_state=10, n_jobs=2)

# Se entrena el modelo
forest.fit(X_train, y_train)
train_scores = []
cv_scores    = []
# Se mide el performance del modelo teniendo el accuracy para train y para test
y_pred = forest.predict(X_test)
y_train_pred = forest.predict(X_train)
print('Train accuracy: %.3f' % accuracy_score(y_train, y_train_pred))
print('Test accuracy: %.3f' % accuracy_score(y_test, y_pred))

''' Se prueba el modelo pero ahora contando con un set de validación.
    Esto se hará mediante el uso de cross validation. Se utilizará K-Folds, que iterará 10 veces.
    Se apartan 1/10 de las muestras, y se entrena al modelo con el restante de muestras.
    Se mide el accuracy obtenido sobre las muestras apartadas.
    Esto quiere decir que se hacen 10 entrenamientos independientes y el accuracy final será
    el promedio de las 10 accuracies anteriores.
    Cross validation permite observar qué tan bueno es el modelo en la práctica. '''

kf = KFold(n_splits=10)
scores = cross_val_score(forest, X_train, y_train, cv=kf, scoring="accuracy")


''' Se observa que al usar el cross validation, se obtiene un accuracy de 0.775806 que es mucho menor que
    sin haberlo utilizado'''

print("Cross validation:", scores.mean())

precision = precision_score(y_test, y_pred, average='macro')
print(f'Precision: {precision * 100:.2f}%')

recall = recall_score(y_test, y_pred, average='macro')
print(f'Recall: {recall * 100:.2f}%')


'''Se busca generar una gráfica para observar el bias y la varianza  que puede tener el modelo,
   además de ver si existe overfitting o underfitting'''

# Generar la curva de aprendizaje
#El eje X representa el validation set, fracciones del conjunto de datos total, desde el 10% hasta el 100%.
train_sizes, train_scores, test_scores = learning_curve(
    forest, X_train, y_train, cv=kf, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

# Calcular las medias y desviaciones estándar de los puntajes
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)


# Crear una figura para la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.title("Curva de Aprendizaje")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Accuracy")
plt.grid()

# Dibujar las líneas de entrenamiento y prueba
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación Cruzada")

# Agregar una leyenda
plt.legend(loc="best")

# Mostrar la curva de aprendizaje
plt.show()

''' Se aplica el Grid Search para ajustar los hiperparámetros del modelo. Este genera todas las
posibles combinaciones de valores de hiperparámetros a partir del espacio de búsqueda definido en grid_space.
Por ejemplo, para max_depth, se probarán los valores 3, 5, 10 y None, y para n_estimators, se probarán 10, 100 y 200, y
así sucesivamente para los demás hiperparámetros.

 Para cada combinación de hiperparámetros, se entrena un modelo de Random Forest utilizando esos hiperparámetros en un conjunto
 de datos de entrenamiento y se evalúa su rendimiento mediante cross validation con Kfolds, para ver el desempeño del modelo con datos
 que no ha aprendido'''



grid_space = {
    'max_depth': [30, 40, 50, None],
    'n_estimators': [200, 300, 400],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [5, 10, 15]
}

grid = GridSearchCV(forest,param_grid=grid_space,cv=kf,scoring='accuracy')
model_grid = grid.fit(X,y)

print('Best hyperparameters are: '+str(model_grid.best_params_))
print('Best score is: '+str(model_grid.best_score_))

''' Se crea una nueva instancia del Random Forest Classifier (con los nuevos hiperparámetros), utilizando gini, un número de estimadores de 300, un max_depth de 30, un min_samples_leaf de 1,
    y se utilizan dos núcleos para mejorar la eficiencia'''
forest1 = RandomForestClassifier(criterion='gini', n_estimators=300, min_samples_split=5, min_samples_leaf=1, max_depth=30, random_state=10, n_jobs=2)

# Se entrena el nuevo modelo forest1
forest1.fit(X_train, y_train)
train_scores1 = []
cv_scores1    = []
# Se mide el performance del nuevo modelo teniendo el accuracy para train y para test
y_pred1 = forest1.predict(X_test)
y_train_pred1 = forest1.predict(X_train)
print('Train accuracy (modelo con nuevos hiperparametros): %.3f' % accuracy_score(y_train, y_train_pred1))
print('Test accuracy (modelo con nuevos hiperparametros): %.3f' % accuracy_score(y_test, y_pred1))

''' Se prueba el modelo con nuevos hiperparámetros pero ahora contando con un set de validación.
    Esto se hará mediante el uso de cross validation. Se utilizará K-Folds, que iterará 10 veces.
    Se apartan 1/10 de las muestras, y se entrena al modelo con el restante de muestras.
    Se mide el accuracy obtenido sobre las muestras apartadas.
    Esto quiere decir que se hacen 10 entrenamientos independientes y el accuracy final será
    el promedio de las 10 accuracies anteriores.
    Cross validation permite observar qué tan bueno es el modelo en la práctica. '''

scores = cross_val_score(forest1, X_train, y_train, cv=kf, scoring="accuracy")


''' Se observa que al usar el cross validation, se obtiene un accuracy de 0.775806 que es mucho menor que
    sin haberlo utilizado'''

print("Cross validation:", scores.mean())

precision = precision_score(y_test, y_pred1, average='macro')
print(f'Precision: {precision * 100:.2f}%')

recall = recall_score(y_test, y_pred1, average='macro')
print(f'Recall: {recall * 100:.2f}%')


'''Se busca generar una gráfica para observar el bias y la varianza  que puede tener el modelo,
   además de ver si existe overfitting o underfitting'''

# Generar la curva de aprendizaje
#El eje X representa el validation set, fracciones del conjunto de datos total, desde el 10% hasta el 100%.
train_sizes, train_scores1, test_scores1= learning_curve(
    forest1, X_train, y_train, cv=kf, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

# Calcular las medias y desviaciones estándar de los puntajes para el modelo con nuevos hiperparámetros
train_scores_mean1 = np.mean(train_scores1, axis=1)
test_scores_mean1 = np.mean(test_scores1, axis=1)


# Crear una figura para la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.title("Curva de Aprendizaje")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Accuracy")
plt.grid()

# Dibujar las líneas de entrenamiento y prueba
plt.plot(train_sizes, train_scores_mean1, 'o-', color="r", label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean1, 'o-', color="g", label="Validación Cruzada")

# Agregar una leyenda
plt.legend(loc="best")

# Mostrar la curva de aprendizaje del modelo con nuevos hiperparámetros
plt.show()