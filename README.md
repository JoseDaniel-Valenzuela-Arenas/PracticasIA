-JOSE DANIEL VALENZUELA ARENAS 7-8PM

-------------------------------------------------------------------------------------------------------------------------------------
Proyecto de Forecasting

Dataset de Entrenamiento
El archivo train.csv no está incluido en este repositorio por su tamaño.
dejo el enlace para descargarlo: https://drive.google.com/file/d/1uhxvA6_57-bg-3Rt7uNFCYCsD-aaSTxS/view?usp=sharing


Cómo ejecutar el código:

Para que el modelo corra correctamente, descarga el dataset y coloca el archivo train.csv en la carpeta principal afuera de 
la carpeta de forecasting

asi mas o menos debe de quedar la estructura:
```text
IA_practica3/ 
├── forecasting/ 
│   └── main.py 
└── train.csv

Para esta práctica utilicé el algoritmo HistGradientBoostingRegressor de la librería `scikit-learn` para predecir las ventas futuras de los productos. 

El programa toma el dataset original y genera variables nuevas. Básicamente se fija en cuánto se vendió hace 1, 7 y 14 días, y también extrae datos de la fecha qué día de la semana y mes es para que el modelo pueda encontrar patrones.
Para evaluar el modelo de forma realista, los datos no se dividieron al azar. se agarro el primer 80% del historial  para entrenar el modelo, y el 20% más reciente lo dejé para ponerlo a prueba.

Resultados obtenidos:

se  compararon los resultados contra un pronóstico base muy sencillo básicamente suponer que hoy se va a vender exactamente lo mismo que el mismo día de la semana pasada

Los resultados fueron los siguientes:
Mi modelo:Logró un RMSE de 330.25 y un R² de 0.9415
Pronóstico base de hace 7 días: Tuvo un RMSE de 451.58 y un R² de 0.8906.

Conclusión:
El modelo logró mejorar el margen de error en un 26.87% frente al método básico. Al terminar de correr el script guarda automáticamente todas las predicciones en un archivo `resultados_forecasting.csv` y genera una gráfica donde se pueden comparar visualmente las ventas reales contra las que predijo el programa.


------------------------------------------------------------------------------------------------------------------------------------
Practica de algoritmos de busqueda BFS Y A*
Este programa no ocupa archivos externos solo se ejecuta y ya.
------------------------------------------------------------------------------------------------------------------------------------

