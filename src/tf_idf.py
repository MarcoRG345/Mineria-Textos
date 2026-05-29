import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Bibliotecas para Procesamiento de Lenguaje Natural (NLP) y Machine Learning
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Configuración visual
sns.set_theme(style="whitegrid")

# 1. Obtención del Corpus
# Seleccionamos 4 categorías distintas para que el contraste en TF-IDF sea evidente
categorias = ['sci.space', 'comp.graphics', 'sci.med', 'talk.politics.mideast']

print("Descargando el corpus de noticias...")
corpus_raw = fetch_20newsgroups(subset='train', categories=categorias, remove=('headers', 'footers', 'quotes'))

df = pd.DataFrame({
    'texto': corpus_raw.data,
    'categoria_id': corpus_raw.target,
    'categoria_nombre': [corpus_raw.target_names[i] for i in corpus_raw.target]
})

# Eliminamos documentos vacíos que pudieron quedar tras remover encabezados
df = df[df['texto'].str.strip().astype(bool)].reset_index(drop=True)
print(f"Total de documentos recuperados: {len(df)}")

# 2. Procesamiento de Datos (Limpieza)
def limpiar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar caracteres especiales, números y signos de puntuación (dejar solo letras)
    texto = re.sub(r'[^a-záéíóúñ]', ' ', texto)
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

df['texto_limpio'] = df['texto'].apply(limpiar_texto)

# Mostramos un ejemplo del antes y después
print("\nEjemplo de texto original:")
print(df['texto'].iloc[0][:150], "...")
print("\nEjemplo de texto limpio:")
print(df['texto_limpio'].iloc[0][:150], "...")

# 3. Aplicación de TF-IDF
vectorizador_tfidf = TfidfVectorizer(
    stop_words='english', 
    max_df=0.8,        # Ignorar palabras que aparecen en más del 80% de los documentos
    min_df=5,          # Ignorar palabras que aparecen en menos de 5 documentos
    max_features=2000  # Limitar el vocabulario a las 2000 palabras más importantes
)

# Ajustamos y transformamos el texto en la matriz dispersa TF-IDF
matriz_tfidf = vectorizador_tfidf.fit_transform(df['texto_limpio'])
nombres_caracteristicas = vectorizador_tfidf.get_feature_names_out()

print(f"\nDimensiones de la matriz TF-IDF: {matriz_tfidf.shape}")
print(f"(Documentos: {matriz_tfidf.shape[0]}, Palabras en el Vocabulario: {matriz_tfidf.shape[1]})")
print(matriz_tfidf)
print(nombres_caracteristicas)

# 4.1 Identificación de las palabras clave por categoría
def top_palabras_por_categoria_mejorado(df_original, matriz_tfidf, nombres_features, top_n=10):
    df_tfidf = pd.DataFrame(matriz_tfidf.toarray(), columns=nombres_features)
    df_tfidf['categoria'] = df_original['categoria_nombre']
    
    # Agrupamos por categoría y calculamos el promedio
    tfidf_promedio = df_tfidf.groupby('categoria').mean()
    
    # Aumentamos el tamaño total de la figura para dar más aire
    fig, axes = plt.subplots(2, 2, figsize=(18, 12)) 
    axes = axes.flatten()
    
    for i, categoria in enumerate(categorias):
        top_terminos = tfidf_promedio.loc[categoria].sort_values(ascending=False).head(top_n)
        
        sns.barplot(x=top_terminos.values, y=top_terminos.index, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Tópico: {categoria}", fontsize=14, fontweight='bold')
        axes[i].set_xlabel("Peso TF-IDF Promedio")
    
    # Ajuste fino para evitar que los títulos y etiquetas se solapen
    plt.tight_layout(pad=4.0) 
    plt.show()

# Llamada a la función optimizada
top_palabras_por_categoria_mejorado(df, matriz_tfidf, nombres_caracteristicas)

# 4.2 Reducción de Dimensionalidad para Visualización 2D
# Reducimos las 2000 dimensiones de TF-IDF a solo 2 componentes principales
pca = PCA(n_components=2, random_state=42)
coordenadas_2d = pca.fit_transform(matriz_tfidf.toarray())

# Creamos un DataFrame para graficar
df_pca = pd.DataFrame(coordenadas_2d, columns=['Componente Principal 1', 'Componente Principal 2'])
df_pca['Categoría'] = df['categoria_nombre']

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_pca, 
    x='Componente Principal 1', 
    y='Componente Principal 2', 
    hue='Categoría', 
    palette='Set1', 
    alpha=0.7,
    s=60
)

plt.title("Proyección PCA de Vectores TF-IDF de Documentos", fontsize=15)
plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza explicada)")
plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza explicada)")
plt.legend(title='Sección de Noticias')
plt.grid(True)
plt.show()
