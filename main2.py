import pandas as pd
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# -----------------------------
# 1. Função de normalização
# -----------------------------
def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
    return text

# -----------------------------
# 2. Carregar planilha
# -----------------------------
df = pd.read_excel("items.xlsx")

# Criar coluna normalizada
df["descricao_norm"] = df["descricao"].apply(normalize)

# -----------------------------
# 3. Vetorização TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    min_df=2,                # ignora termos muito raros
    analyzer='word',         # analisa por palavras
    ngram_range=(1, 2)       # considera 1 ou 2 palavras juntas
)
X = vectorizer.fit_transform(df["descricao_norm"])

# -----------------------------
# 4. Clustering com DBSCAN
# -----------------------------
clustering = DBSCAN(
    eps=0.3,                 # raio de similaridade (ajustável)
    min_samples=2,           # mínimo de itens no cluster
    metric="cosine"          # similaridade baseada em cosseno
).fit(X)

# -----------------------------
# 5. Salvar resultado
# -----------------------------
df["cluster"] = clustering.labels_

df.to_excel("itens_clusterizados.xlsx", index=False)

print("✅ Arquivo 'itens_clusterizados.xlsx' gerado com clusters de similares!")
