from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

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
# Como para mis features no quiero la clase, tambien sera eliminada 
X = df.drop (['ID', "class"], axis = 1)


# Se establece el target en y, que en este caso quiero predecir la clase (H o P)
y = df[['class']]
y = (y["class"] == 'P').astype(int) # regresa 1 si el paciente tiene alzheimer y 0 si no



# Se imprime para visualizar las dimensiones de "X" y "y"
#print('X size: {}'.format(X.shape))
# se imprime ('Dimensions of X: {}'.format(X.ndim))
#print('X ({}): \n{} ...'.format(type(X), X[0:5]))
#print('\ny size: {}'.format(y.shape))
# se imprime ('Dimensions of y: {}'.format(y.ndim))
#print('y ({}): \n{}'.format(type(y), y))

''' Se escalan los valores de los features con el fin de acelerar la convergencia, 
minimizar la función de pérdida,  y mejorar el performance del modelo minimizando la diferencia 
entre la predicción y los valores reales pero sin memorizar datos. '''

sc = StandardScaler()
X = sc.fit_transform(X)


# Crear la matriz para theta con los valores iniciales
theta = np.random.randn(len(X[0]) + 1, 1) * 0.01 # Se inicia con valores cercanos a 0


# Se agrega x0 (que es una columna de 1s) a X para multiplicar theta y obtener el bias
X_vect = np.c_[np.ones((len(X), 1)), X]
#print(X_vect[:5])
#print(X_vect.shape)


# ------------- Regresión logística -----------------

'''Se genera la función sigmoide, que transforma valores de un rango amplio a valores de 0 a 1, 
transformando las salidas lineales en probabilidades de pertenecer a una clase. '''

def sigmoid_function(X):
  return 1/(1+math.e**(-X))


''' Se crea la función log_regression, que implementa la optimización 
de regresión logística utilizando el descenso por gradiente. Esta función parará de entrenar al modelo
al alcanzar un error entre la función de pérdida anterior y la actual menor a 0.000001'''

def log_regression(X, y, theta, alpha, epochs):
  y_ = np.reshape(y, (len(y), 1)) # shape (150,1)
  N = len(X)
  avg_loss_list = []
  loss_last_epoch = 9999999 #Se inicializa la pérdida anterior
  for epoch in range(epochs):
    sigmoid_x_theta = sigmoid_function(X_vect.dot(theta)) # shape: (150,5).(5,1) = (150,1)
    grad = (1/N) * X_vect.T.dot(sigmoid_x_theta - y_) # shapes: (5,150).(150,1) = (5, 1)
    best_params = theta
    theta = theta - (alpha * grad)
    hyp = sigmoid_function(X_vect.dot(theta)) # shape (150,5).(5,1) = (150,1)
    avg_loss = -np.sum(np.dot(y_.T, np.log(hyp) + np.dot((1-y_).T, np.log(1-hyp)))) / len(hyp)
    if epoch % 10 == 0:
      print('epoch: {} | avg_loss: {}'.format(epoch, avg_loss))
    avg_loss_list.append(avg_loss)
    loss_step = abs(loss_last_epoch - avg_loss) # Se obtendrá el valor absoluto de la resta entre la función de pérdida anterior y la actual, para así ir viendo si el error es de 0.001
    loss_last_epoch = avg_loss # Se actualiza el valor de la función de pérdida anterior
    if loss_step < 0.000001: # Si el error es menor a 0.001 entonces se habrá llegado a lo que se pide
      print('\nStopping training on epoch {}/{}, as (last epoch loss - current epoch loss) is less than 0.001 [{}]'.format(epoch, epochs, loss_step)) #*
      break # Por lo tanto se deja de entrenar el modelo o hacer los epochs
  return best_params
epochs = 100000
alpha = 0.1
best_params = log_regression(X, y, theta, alpha, epochs)
print('Best parameters:\n', best_params)

# ------------- Hacer predicciones ------------------------

''' Establecí tres casos de prueba, uno donde todos los valores son como los entrenados para un paciente con alzheimer, uno donde le modifiqué dos valores al primero
(air_time1 y max_x_extension1), y me debía dar que aún así tiene probabilidad de tener alzheimer, y por último donde todos los valores son como los entrenados para 
un paciente sin alzheimer'''
x_to_predict = [[2600,1.03E-05,229.9339973,172.7618583,2333,5802,0.387019883,0.181342283,201.3479278,0.064220393,0.020126445,3.378343433,3.30886561,10,6080,1520.253289,120845.8717,8680,560,1.05E-05,122.8095843,92.98327881,322,5059,0.211138474,0.075103317,107.8964316,0.030854063,0.009264755,2.736669283,1.856534397,1,6545,1948.3178,48577.01894,7105,420,8.63E-06,82.56239194,146.1162954,4645,266,0.202898789,0.102299118,114.3393437,0.026745257,0.011406152,1.834658305,2.906881307,1,5110,1814.446184,76556.47998,5530,1030,2.33E-05,107.3481862,94.27085815,3960,4013,0.256849969,0.090973199,100.8095222,0.034442096,0.009971824,0.847393203,1.88839189,1,17770,1842.092009,101793.3677,18800,680,1.43E-05,101.6284101,77.01411536,2556,2245,0.240662044,0.155546806,89.32126274,0.022617012,0.013758778,1.067686358,1.540443509,1,11630,1992.745916,20179.47241,12310,3055,4.51E-06,156.5122888,300.4798888,2111,4385,0.160120214,0.177293294,228.4960888,0.022402315,0.018881405,3.221477455,4.946831569,3,2290,1226.744541,129706.0636,5345,3300,5.35E-0,273.9085787,366.3688056,12620,677,1.091498128,0.162193546,320.1386921,0.202532795,0.018332491,5.71028505,3.885387743,5,2665,1587.56848,203046.5718,5965,3940,6.90E-06,936.4647205,117.0799873,1598,2323,6.121763754,0.10904222,526.7723539,1.218504869,0.013047462,17.15932273,2.32781741,1,4055,1834.902589,42796.46523,7995,625,9.56E-06,299.5461063,96.96948047,1736,4187,0.229538406,0.120400177,198.2577934,0.030777835,0.013190617,1.238624473,1.972156487,1,6345,1811.951143,35086.33489,6970,505,5.48E-06,201.9967387,123.4186679,1779,2096,0.246590211,0.122487145,162.7077033,0.02819765,0.016122404,1.905944811,2.494940393,2,3135,1550.052632,97088.4741,3640,670,5.00E-06,210.4977709,131.0796439,1752,1868,0.234543796,0.151772546,170.7887074,0.022957305,0.017452625,3.256622648,2.635389069,3,2975,1599.717647,189210.1522,3645,1190,6.17E-06,209.4570251,56.71852097,481,3380,0.340840453,0.126547318,133.0877731,0.057870264,0.015056083,1.773553365,1.233790695,1,5170,1560.658607,34583.73548,6360,241,6.27E-06,314.0143573,65.64667391,806,3730,0.542508569,0.184141247,189.8305156,0.088879102,0.018890679,5.46192676,1.381828584,5,4550,1853.78022,180637.9319,6960,4675,9.78E-06,229.4767189,172.6577686,2109,6584,0.324243203,0.157076312,201.0672438,0.04787333,0.018128388,3.065252077,3.214526841,4,5250,1407.159048,115674.7433,9925,22345,1.42E-05,219.2763071,85.9567351, 3106, 4276, 0.747249797, 0.127279156, 152.6165211, 0.133921582, 0.016585865, 3.92253238, 1.519800703, 11, 11350, 1717.3163, 173772.7572, 33695, 3265, 0, 255.3670511, 0, 0, 0, 1.693373004, 0, 127.6835256, 0.322505017, 0, 5.380795045, 0, 0, 0, 0, 0, 3265, 27640, 2.88E-05, 355.4280376, 166.6643259, 12885, 13413, 2.11670939, 0.137683162, 261.0461817, 0.410326299, 0.01733735, 6.94766303, 1.956684153, 26, 23350, 1627.26424, 205307.2008, 50990, 525, 4.40E-06, 167.8123043, 147.7869756, 1250, 2906, 0.245819244, 0.140660122, 157.7996399, 0.027656135, 0.01730222, 1.470191698, 2.933225221, 4, 2100, 1840.95, 113069.7618, 2625, 65635, 2.98E-05, 160.4685989, 121.1830506, 7143, 10158, 0.928650027, 0.145060653, 140.8258248, 0.173414448, 0.017981206, 3.232291575, 1.605500634, 42, 28785, 1611.954316, 193490.8166, 94420, 4655, 1.43E-05, 124.7198502, 155.8039447, 1574, 11819, 0.19144191, 0.141611328, 140.2618974, 0.024976884, 0.017397351, 2.25819997, 2.686954585, 16, 8105, 1564.901912, 153318.4759, 12760, 1135, 4.89E-05, 703.1554978, 98.13585736, 9702, 16771, 0.526515116, 0.123402105, 400.6456776, 0.090781953, 0.012181298, 3.020595477, 1.965451136, 2, 32890, 1954.565521, 54854.49989, 34025, 7425, 9.52E-06, 147.9532255, 151.3062394, 1198, 5111, 0.310100579, 0.15207765, 149.6297325, 0.047833242, 0.019265661, 2.960016856, 2.388018394, 12, 5505, 1680.804723, 222939.0745, 12930, 7330, 1.09E-05, 117.7653038, 189.6821559, 1338, 8208, 0.229315569, 0.138531569, 153.7237299, 0.031520668, 0.018083461, 2.415460388, 2.730515247, 12, 6535, 1599.237184, 264722.8435, 13865, 68360, 2.21E-05, 110.209716, 85.44896921, 3798, 4213, 0.367091271, 0.126616426, 97.82934262, 0.060498739, 0.016243104, 2.256226282, 1.14333186, 41, 34735, 1269.215777, 343051.6317, 103095, 33545, 5.60E-05, 215.3795419, 171.9544941, 7688, 14127, 0.941396258, 0.134279603, 193.667018, 0.178194063, 0.017174178, 4.000780602, 2.392521369, 74, 45480, 1431.443492, 144411.7055, 79025]] # tiene alzheimer (p)
#x_to_predict = [[1760,1.03E-05,229.9339973,172.7618583,1674,5802,0.387019883,0.181342283,201.3479278,0.064220393,0.020126445,3.378343433,3.30886561,10,6080,1520.253289,120845.8717,8680,560,1.05E-05,122.8095843,92.98327881,322,5059,0.211138474,0.075103317,107.8964316,0.030854063,0.009264755,2.736669283,1.856534397,1,6545,1948.3178,48577.01894,7105,420,8.63E-06,82.56239194,146.1162954,4645,266,0.202898789,0.102299118,114.3393437,0.026745257,0.011406152,1.834658305,2.906881307,1,5110,1814.446184,76556.47998,5530,1030,2.33E-05,107.3481862,94.27085815,3960,4013,0.256849969,0.090973199,100.8095222,0.034442096,0.009971824,0.847393203,1.88839189,1,17770,1842.092009,101793.3677,18800,680,1.43E-05,101.6284101,77.01411536,2556,2245,0.240662044,0.155546806,89.32126274,0.022617012,0.013758778,1.067686358,1.540443509,1,11630,1992.745916,20179.47241,12310,3055,4.51E-06,156.5122888,300.4798888,2111,4385,0.160120214,0.177293294,228.4960888,0.022402315,0.018881405,3.221477455,4.946831569,3,2290,1226.744541,129706.0636,5345,3300,5.35E-0,273.9085787,366.3688056,12620,677,1.091498128,0.162193546,320.1386921,0.202532795,0.018332491,5.71028505,3.885387743,5,2665,1587.56848,203046.5718,5965,3940,6.90E-06,936.4647205,117.0799873,1598,2323,6.121763754,0.10904222,526.7723539,1.218504869,0.013047462,17.15932273,2.32781741,1,4055,1834.902589,42796.46523,7995,625,9.56E-06,299.5461063,96.96948047,1736,4187,0.229538406,0.120400177,198.2577934,0.030777835,0.013190617,1.238624473,1.972156487,1,6345,1811.951143,35086.33489,6970,505,5.48E-06,201.9967387,123.4186679,1779,2096,0.246590211,0.122487145,162.7077033,0.02819765,0.016122404,1.905944811,2.494940393,2,3135,1550.052632,97088.4741,3640,670,5.00E-06,210.4977709,131.0796439,1752,1868,0.234543796,0.151772546,170.7887074,0.022957305,0.017452625,3.256622648,2.635389069,3,2975,1599.717647,189210.1522,3645,1190,6.17E-06,209.4570251,56.71852097,481,3380,0.340840453,0.126547318,133.0877731,0.057870264,0.015056083,1.773553365,1.233790695,1,5170,1560.658607,34583.73548,6360,241,6.27E-06,314.0143573,65.64667391,806,3730,0.542508569,0.184141247,189.8305156,0.088879102,0.018890679,5.46192676,1.381828584,5,4550,1853.78022,180637.9319,6960,4675,9.78E-06,229.4767189,172.6577686,2109,6584,0.324243203,0.157076312,201.0672438,0.04787333,0.018128388,3.065252077,3.214526841,4,5250,1407.159048,115674.7433,9925,22345,1.42E-05,219.2763071,85.9567351, 3106, 4276, 0.747249797, 0.127279156, 152.6165211, 0.133921582, 0.016585865, 3.92253238, 1.519800703, 11, 11350, 1717.3163, 173772.7572, 33695, 3265, 0, 255.3670511, 0, 0, 0, 1.693373004, 0, 127.6835256, 0.322505017, 0, 5.380795045, 0, 0, 0, 0, 0, 3265, 27640, 2.88E-05, 355.4280376, 166.6643259, 12885, 13413, 2.11670939, 0.137683162, 261.0461817, 0.410326299, 0.01733735, 6.94766303, 1.956684153, 26, 23350, 1627.26424, 205307.2008, 50990, 525, 4.40E-06, 167.8123043, 147.7869756, 1250, 2906, 0.245819244, 0.140660122, 157.7996399, 0.027656135, 0.01730222, 1.470191698, 2.933225221, 4, 2100, 1840.95, 113069.7618, 2625, 65635, 2.98E-05, 160.4685989, 121.1830506, 7143, 10158, 0.928650027, 0.145060653, 140.8258248, 0.173414448, 0.017981206, 3.232291575, 1.605500634, 42, 28785, 1611.954316, 193490.8166, 94420, 4655, 1.43E-05, 124.7198502, 155.8039447, 1574, 11819, 0.19144191, 0.141611328, 140.2618974, 0.024976884, 0.017397351, 2.25819997, 2.686954585, 16, 8105, 1564.901912, 153318.4759, 12760, 1135, 4.89E-05, 703.1554978, 98.13585736, 9702, 16771, 0.526515116, 0.123402105, 400.6456776, 0.090781953, 0.012181298, 3.020595477, 1.965451136, 2, 32890, 1954.565521, 54854.49989, 34025, 7425, 9.52E-06, 147.9532255, 151.3062394, 1198, 5111, 0.310100579, 0.15207765, 149.6297325, 0.047833242, 0.019265661, 2.960016856, 2.388018394, 12, 5505, 1680.804723, 222939.0745, 12930, 7330, 1.09E-05, 117.7653038, 189.6821559, 1338, 8208, 0.229315569, 0.138531569, 153.7237299, 0.031520668, 0.018083461, 2.415460388, 2.730515247, 12, 6535, 1599.237184, 264722.8435, 13865, 68360, 2.21E-05, 110.209716, 85.44896921, 3798, 4213, 0.367091271, 0.126616426, 97.82934262, 0.060498739, 0.016243104, 2.256226282, 1.14333186, 41, 34735, 1269.215777, 343051.6317, 103095, 33545, 5.60E-05, 215.3795419, 171.9544941, 7688, 14127, 0.941396258, 0.134279603, 193.667018, 0.178194063, 0.017174178, 4.000780602, 2.392521369, 74, 45480, 1431.443492, 144411.7055, 79025]] #Se hace una alteracion a primera y 5ta columna
#x_to_predict = [[3830,8.36E-06,151.536989,171.1046932,1287,7352,0.165479773,0.161058267,161.3208411,0.022704575,0.022440786,2.041488772,3.507108391,14,4060,1800.671182,212575.802,7890,4150,1.15E-05,303.5220015,97.72068783,954,5131,0.463247415,0.078434067,200.6213447,0.074039976,0.008904834,3.338464682,1.519675396,6,7820,1933.470588,112520.267,11970,2485,1.06E-05,428.0202677,179.132823,4092,722,0.282043992,0.109258914,303.5765453,0.029754269,0.014427334,4.902278957,2.190053855,5,6520,1664.501534,239053.7776,9005,610,2.68E-05,63.10565635,50.76686386,4389,4303,0.390870531,0.145515952,56.9362601,0.05419415,0.011431387,2.093381112,1.017244236,2,34435,2036.033251,14700.83162,35045,680,1.47E-05,102.3082762,57.99571066,2405,2410,0.229593477,0.184362162,80.15199342,0.023604712,0.013463397,1.526709807,1.162445071,1,15230,2034.027249,15189.28521,15910,3250,4.45E-06,87.32463977,104.2886278,1044,3367,0.184760416,0.089999026,95.80663379,0.018446684,0.013860736,1.779184341,1.326151413,4,3180,1803.550314,202297.8041,6430,295415,6.76E-06,139.2887168,271.6948768,13268,1261,0.73599636,0.116642814,205.4917968,0.128033109,0.01471791,2.831126199,2.811845585,5,3685,1858.453189,184381.952,299100,1135,7.07E-06,193.2028767,127.7007263,1450,2811,0.247221161,0.106817412,160.4518015,0.030829824,0.012315573,2.641846847,2.510853744,3,3850,1872.737662,210090.4455,4985,2070,1.07E-05,178.7008354,96.1091485,1624,4171,0.396100695,0.125820487,137.4049919,0.055846637,0.012725848,2.60425714,1.856158382,4,7465,1981.525787,73509.07184,9535,1310,6.64E-06,196.2248078,151.0707008,1951,2899,0.540621023,0.102930427,173.6477543,0.092495408,0.014157366,3.423044761,2.953243286,3,3310,1937.311178,90946.52855,4620,3595,7.03E-06,91.21056224,150.014276,2067,2527,0.261240758,0.119339543,120.6124191,0.031303998,0.015453852,1.723164169,2.913168863,5,4250,1836.072941,209789.7029,7845,1079,8.83E-06,377.196177,67.3666814,918,4655,0.214671473,0.107930806,222.2814292,0.027066534,0.013432169,4.971417751,1.322522424,3,6950,1978.122302,85074.68576,8029,785,7.98E-06,238.4922906,100.6088789,779,3948,0.221027341,0.1221214,169.5505848,0.027698701,0.016457486,2.243011478,2.020157202,3,5055,1904.738872,159306.8141,5840,8895,1.78E-05,153.1026085,169.0955248,5651,4550,0.345886141,0.10727977,161.0990667,0.051254418,0.014850175,2.467744323,2.681298384,13,11235,1852.963952,186080.0392,20130,9125,9.35E-06,149.9048439,126.4144608,1713,4654,0.490123435,0.126933553,138.1596523,0.08023609,0.018555595,2.959504647,2.125312005,13,5520,1768.248188,198123.2609,14645,915,4.33E-06,213.8482379,121.0437303,877,2586,0.223864636,0.119674921,167.4459841,0.035432142,0.016164623,3.614685356,2.104061881,4,2380,1892.544118,115321.4791,3295,13250,3.09E-05,360.7439971,228.4644494,12177,13644,1.616910166,0.125822693,294.6042232,0.310986274,0.016942579,6.575365789,2.972026561,31,22145,1847.216979,184288.3076,35395,940,5.39E-06,356.5485112,171.6530556,1363,3269,0.27984123,0.124619072,264.1007834,0.027031704,0.016446496,4.593471067,3.347513547,4,2755,1857.774955,171931.0637,3695,9170100,3.39E-05,95.34681849,126.8355671,7457,9793,0.444348921,0.121603208,111.0911928,0.074218602,0.01672931,2.025521414,1.823022393,52,36065,1868.644808,166125.7663,9206165,2705,1.82E-05,243.5004353,153.326098,1677,12645,0.179317711,0.114995953,198.4132667,0.023918297,0.014566782,3.372516476,2.640351704,14,10350,1902.345894,137523.863,13055,4225,5.34E-05,305.3679988,105.8024578,10470,16889,0.570933304,0.101307556,205.5852283,0.098393801,0.011276069,3.666651689,2.080084685,3,33140,1988.397556,75569.47336,37365,8265,1.06E-05,97.4733407,135.6602534,1196,5315,0.180115946,0.133256612,116.566797,0.021829871,0.016017466,1.882687062,2.07553168,12,6885,1862.310821,143599.3965,15150,1850,1.02E-05,249.9770867,166.7921134,1127,6014,0.229854428,0.125804021,208.3846,0.033345487,0.016877479,4.92718861,2.492175594,12,5910,1930.645516,128484.0156,7760,31380,2.48E-05,141.3250047,121.0237467,6102,6337,0.442462734,0.132393435,131.1743757,0.072291011,0.016190553,2.855702169,1.840004926,26,23030,1878.400782,188004.6023,54410,31770,5.86E-05,183.2610913,172.6100404,6618,15141,0.659078771,0.128780324,177.9355658,0.114472226,0.017194113,3.493814654,2.510601187,88,51675,1915.573488,128727.1241,83445]] # No tiene alzheimer (H)
# Convertir x_to_predict a un array de Numpy

print(" Se quiere predecir si el paciente tiene alzheimer, con las siguientes caracteristicas:")
print(x_to_predict)

# Scaling
x_to_predict = sc.transform(x_to_predict)
print("Se hizo un escalamiento, x:")
print(x_to_predict)

# Add x0 (1)
x_to_predict = np.c_[np.ones((len(x_to_predict), 1)), x_to_predict] # Se crea la matriz x_to_predict para predecir utilizando el modelo previamente entrenado
print(x_to_predict)

theta = best_params
y_pred = sigmoid_function(x_to_predict.dot(theta)) #Se aplica la función sigmoide a la x que se predecirá usando los parámetros que se encontraron en el entrenamiento
y_pred_w_threshold = 1 if y_pred[0][0] >= 0.5 else 0 #Se toma el valor predicho, si es mayor que 0.5 se devolverá 1 de predicción, si no, se devuelve 0.
print('Prediction Value: {}'.format(y_pred))
print('Probabilidad de que el paciente tenga alzheimer: {}%'.format(round(y_pred[0][0], 2)*100))
print('Prediccion con etiqueta de 0 o 1: {}'.format(y_pred_w_threshold))

''' 

* Conclusiones y explicación *

El Alzheimer es una enfermedad neurodegenerativa crónica, que afecta la memoria y el comportamiento de
una persona, usualmente de edad avanzada. Suele ser una causa de demencia para los adultos mayores. 
Entre sus síntomas se encuentran: problemas de memoria, desafíos en la resolución de problemas, confusión
con tiempo y lugar, y la pérdida de habilidades cognitivas. Con base en esta enfermedad, se decidió utilizar el dataset 
"Alzheimer's Disease Detection from Handwriting" obtenido de Kaggle. Este dataset cuenta con features o características 
que presenta un adulto mayor al momento de escribir, con características como cuántas veces el bolígrafo tocó la hoja,
promedio de la presión, el tiempo que le llevó escribir, entre otras. Este proyecto se centró en determinar si un paciente tiene 
o no tiene Alzheimer, basándose con los features mencionados, entendiendo que este es un problema de clasificación binaria.

En el presente  código se implementó un algoritmo de regresión logística como el visto
en las clases de machine learning sin utilizar ninguna biblioteca o framework de aprendizaje máquina
ni de estadística avanzada. Este algoritmo es de gran utilidad para predicciones binarias, ya que escala los 
valores y permite modelar la probabilidad de que algo sea verdadero o falso. Este utiliza la función sigmoidal 
para que la salida sea un valor entre 0 o 1, y permite establecer límites para determinar cuando es un resultado
positivo y un resultado negativo. Además, se seleccionó este algoritmo porque considero que es relativamente rápido
de entrenar y es realmente eficiente, esto lo observé en las clases, donde fue el que mejor entendí como funciona y 
cómo se aplica.

Para desarrollar este algoritmo primeramente se creó un dataframe "X" que contiene todos los features o características que 
suele tener un paciente con Alzheimer, y otro dataframe  "y" que contiene las posibles salidad o labels, como "P" para "positivo"
y "H" para negativo a padecer la enfermedad. Sin embargo, este dataframe "x" se le asignó valores de "1" en vez de "P" y de 0 en vez de "H" 
para tener solo probabilidades entre 0 y 1 al ser un clasificador binario. Posteriormente,  se escalaron los valores de los features con el 
fin de acelerar la convergencia, minimizar la función de pérdida,  y mejorar el performance del modelo minimizando la diferencia entre la
predicción y los valores reales pero sin memorizar datos, y se creó la matriz para theta con los valores iniciales cercanos a 0 (al inicio 
había inicializado theta con ceros, pero esto me generaba valores de pérdida muy bajos que producían NA al entrenar el modelo). Posteriormente,
se agregó x0 (que es una columna de valores de 1) a X para multiplicar theta y obtener el bias. Se crearon distintas funciones como: la función 
sigmoide, que transforma valores de un rango amplio a valores de 0 a 1, transformando las salidas lineales en probabilidades de pertenecer a una 
clase; la función log_regression (con un alfa de 0.1 que fue el que mejor me funcionó) que implementa la optimización de regresión logística 
utilizando el descenso por gradiente y parará de entrenar al modelo al alcanzar un error entre la función de pérdida anterior y la actual menor a 0.000001, 
y se almacenen los mejores parámetros en una variable, ya que se estarían repitiendo los valores de pérdida y no habría un cambio significativo al seguir 
ejecutando epochs. Por último, se establecieron los casos de prueba: el primero donde todos los valores son como los entrenados para un paciente con alzheimer, 
otro donde le modifiqué dos valores al primero (air_time1 y max_x_extension1), y por último donde todos los valores son como los entrenados para un paciente sin alzheimer. 
Por último, se aplicó la función sigmoide a la x que se predecirá usando los parámetros que se encontraron en el entrenamiento. Si el valor predicho es mayor que 0.5 se devolverá 
1 de predicción, si no, se devuelve 0.

'''