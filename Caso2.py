import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt

# Lee los datos desde Excel
df = pd.read_excel('g3Caso2/Super Store.xlsx')

####Segmentación de Clientes
#Grafica segmentacion de clientes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Calcula el valor del cliente (ventas totales por cliente) y la frecuencia de compra (número de compras por cliente)
customer_data = df.groupby('Customer ID').agg({'Sales': 'sum', 'Order ID': 'count'}).reset_index()
customer_data.rename(columns={'Sales': 'Total Sales', 'Order ID': 'Frequency'}, inplace=True)

# Selecciona las características relevantes para la segmentación
X = customer_data[['Total Sales', 'Frequency']]

# Estandariza las características para que tengan media cero y varianza unitaria
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Utiliza el algoritmo K-Means para segmentar a los clientes en 4 grupos (puedes ajustar el número de grupos según tus necesidades)
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Crea un gráfico de dispersión que muestra los segmentos en función del valor del cliente y la frecuencia de compra
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_data, x='Total Sales', y='Frequency', hue='Cluster', palette='viridis')
plt.xlabel('Valor del Cliente (Total Sales)')
plt.ylabel('Frecuencia de Compra (Frequency)')
plt.title('Segmentación de Clientes por Valor y Frecuencia de Compra')
plt.legend(title='Segmento')
plt.show()
# Guarda el gráfico en formato JPG



# Analiza los segmentos resultantes
segment_summary = customer_data.groupby('Cluster').agg({
    'Total Sales': 'mean',
    'Frequency': 'mean',
    'Customer ID': 'count'
}).reset_index()

print("Resumen de Segmentos:")
print(segment_summary)


####Análisis de Productos
#Grafico de 5 productos mas vendidos y los menos vendidos

# Agrupa los datos por el nombre del producto y suma las ventas
product_sales = df.groupby('Product Name')['Sales'].sum().reset_index()

# Ordena los productos por ventas en orden descendente (de mayor a menor)
productos_mas_vendidos = product_sales.nlargest(5, 'Sales')
productos_menos_vendidos = product_sales.nsmallest(5, 'Sales')

# Combina los productos más vendidos y menos vendidos en un solo DataFrame
top_and_bottom_products = pd.concat([productos_mas_vendidos, productos_menos_vendidos])

# Crea un gráfico de barras para mostrar los productos más y menos vendidos
plt.figure(figsize=(10, 6))
sns.barplot(data=top_and_bottom_products, x='Sales', y='Product Name', palette='viridis')
plt.xlabel('Ventas')
plt.ylabel('Producto')
plt.title('Los 5 Productos Más y Menos Vendidos')
plt.show()

#Lista de los productos más vendidos y menos vendidos
product_sales = df.groupby('Product Name')['Sales'].sum().reset_index()
top_products = product_sales.nlargest(5, 'Sales')  # Los 5 productos más vendidos
bottom_products = product_sales.nsmallest(5, 'Sales')  # Los 5 productos menos vendidos

print("Productos más vendidos:")
print(top_products)

print("\nProductos menos vendidos:")
print(bottom_products)


#Grafico de las categorias mas vendidas

# Agrupa los datos por la columna 'Category' y suma las ventas
category_sales = df.groupby('Category')['Sales'].sum().reset_index()

# Ordena las categorías por ventas en orden descendente (de mayor a menor)
categorias_mas_vendidas = category_sales.nlargest(5, 'Sales')

# Crea un gráfico de barras para mostrar las 5 categorías más vendidas
plt.figure(figsize=(10, 6))
sns.barplot(data=categorias_mas_vendidas, x='Sales', y='Category', palette='viridis')
plt.xlabel('Ventas')
plt.ylabel('Categoría')
plt.title('Las Categorías Más Vendidas')
plt.show()


# Lista categorias mas vendidas 
# Agrupa los datos por la columna 'Category' y suma las ventas
category_sales = df.groupby('Category')['Sales'].sum().reset_index()

# Ordena las categorías por ventas en orden descendente (de mayor a menor)
categorias_mas_vendidas = category_sales.sort_values(by='Sales', ascending=False).head(5)


# Imprime las categorías más vendidas
print("Categorías más vendidas:")
print(categorias_mas_vendidas)

####Análisis de Tiempo
# Ordena los datos por fecha
# Convierte la columna "Order Date" en un tipo de dato de fecha
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extrae el año y el mes de la columna "Order Date"
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# Agrupa los datos por año y mes y suma las ventas
sales_by_month = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

# Crea un gráfico de líneas para mostrar las tendencias estacionales en las ventas
plt.figure(figsize=(12, 6))  # Tamaño de la figura
sns.lineplot(data=sales_by_month, x='Month', y='Sales', hue='Year')
plt.xlabel('Mes')
plt.ylabel('Ventas')
plt.title('Tendencias Estacionales en las Ventas')
plt.legend(title='Año', loc='upper right')
plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
plt.show()








