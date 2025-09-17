# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import DBSCAN

# # 1. Carregar planilha
# df = pd.read_excel("items2.xlsx")
# descricoes = df["Item"].astype(str).tolist()

# # 2. Gerar embeddings
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# embeddings = model.encode(descricoes, show_progress_bar=True)

# # 3. Clustering
# clustering = DBSCAN(eps=0.35, min_samples=2, metric="cosine").fit(embeddings)
# df["grupo"] = clustering.labels_

# # 4. Para cada grupo, manter só o item mais detalhado
# resultados = []
# for grupo in df["grupo"].unique():
#     if grupo == -1:  
#         # outliers ficam como estão
#         itens = df[df["grupo"] == grupo]["Item"].tolist()
#         resultados.extend(itens)
#     else:
#         itens = df[df["grupo"] == grupo]["Item"].tolist()
#         # escolher o mais detalhado (maior número de palavras)
#         escolhido = max(itens, key=lambda x: len(x.split()))
#         resultados.append(escolhido)

# # 5. Criar nova planilha com os únicos saneados
# df_final = pd.DataFrame({"Item_saneado": sorted(set(resultados))})
# df_final.to_excel("produtos_saneados.xlsx", index=False)




# import dedupe
# import pandas as pd

# # 1. Carregar dados
# df = pd.read_excel("items2.xlsx")

# # Transformar no formato que o dedupe espera
# data = {i: {"Item": str(row["Item"])} for i, row in df.iterrows()}

# # 2. Configurar campos
# fields = [dedupe.variables.String("Item")]

# # 3. Criar deduper
# deduper = dedupe.Dedupe(fields)

# # 4. Preparar treinamento
# deduper.prepare_training(data)

# # 5. Treinamento interativo
# dedupe.console_label(deduper)  # você já fez isso e digitou 'f'
# deduper.train()

# # 6. Deduplicar
# clustered_dupes = deduper.partition(data, 0.5)

# # 7. Criar lista de itens saneados
# saneados = []
# for cluster, score in clustered_dupes:
#     # Manter o primeiro item de cada cluster
#     # (ou você pode escolher outro critério)
#     record_id = cluster[0]
#     saneados.append(data[record_id]["Item"])

# # 8. Salvar em uma nova planilha
# df_saneado = pd.DataFrame({"Item": saneados})
# df_saneado.to_excel("items_saneados.xlsx", index=False)

# print("Planilha com itens saneados criada: items_saneados.xlsx")




# # FICOUBAO
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # 1. Carregar planilha
# df = pd.read_excel("items.xlsx")  # ou CSV
# produtos = df["Item"].astype(str).tolist()

# # 2. Gerar embeddings
# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(produtos, show_progress_bar=True)

# # 3. Calcular similaridade
# similarity_matrix = cosine_similarity(embeddings)

# # 4. Agrupar produtos semelhantes
# threshold = 0.88  # ajuste conforme necessidade
# visited = set()
# groups = []

# for i in range(len(produtos)):
#     if i in visited:
#         continue
#     group = [i]
#     visited.add(i)
#     for j in range(i + 1, len(produtos)):
#         if similarity_matrix[i, j] >= threshold:
#             group.append(j)
#             visited.add(j)
#     groups.append(group)

# # 5. Criar lista de produtos saneados
# saneados = []
# for group in groups:
#     # escolhe qualquer item do grupo (ou o primeiro)
#     saneados.append(produtos[group[0]])

# # 6. Salvar em Excel
# df_saneado = pd.DataFrame({"Item": saneados})
# df_saneado.to_excel("items_saneados.xlsx", index=False)

# print(f"{len(produtos)} itens originais -> {len(saneados)} itens saneados")




# # DEU MUITO BAO
# import pandas as pd
# import re
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------
# # 1️⃣ Carregar dados
# # -----------------------------
# df = pd.read_excel("items2.xlsx")  # coluna "Item"
# produtos = df["Item"].astype(str).tolist()

# # -----------------------------
# # 2️⃣ Funções auxiliares
# # -----------------------------
# def extrair_medidas(texto):
#     """Extrai todos os números do texto e normaliza para float."""
#     numeros = re.findall(r'\d+(?:[\.,]\d+)?', texto)
#     numeros = [float(n.replace(',', '.')) for n in numeros]
#     return numeros

# def medidas_iguais(med1, med2, tol=0.1):
#     """Compara duas listas de medidas com tolerância."""
#     if len(med1) != len(med2):
#         return False
#     return all(abs(a - b) <= tol for a, b in zip(med1, med2))

# # -----------------------------
# # 3️⃣ Gerar embeddings
# # -----------------------------
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# embeddings = model.encode(produtos, show_progress_bar=True)

# # -----------------------------
# # 4️⃣ Matriz de similaridade
# # -----------------------------
# similarity_matrix = cosine_similarity(embeddings)

# # -----------------------------
# # 5️⃣ Agrupamento considerando medidas
# # -----------------------------
# threshold = 0.75  # ajuste fino conforme necessário
# visited = set()
# grupos = []

# for i in range(len(produtos)):
#     if i in visited:
#         continue
#     group = [i]
#     visited.add(i)
#     med_i = extrair_medidas(produtos[i])

#     for j in range(i + 1, len(produtos)):
#         if j in visited:
#             continue
#         if similarity_matrix[i, j] >= threshold:
#             med_j = extrair_medidas(produtos[j])
#             if medidas_iguais(med_i, med_j):
#                 group.append(j)
#                 visited.add(j)
#     grupos.append(group)

# # -----------------------------
# # 6️⃣ Escolher produto representativo
# # -----------------------------
# # Vamos pegar o mais detalhado (maior comprimento)
# resultado = []
# for group in grupos:
#     itens = [produtos[i] for i in group]
#     escolhido = max(itens, key=len)
#     resultado.append(escolhido)

# # -----------------------------
# # 7️⃣ Salvar planilha limpa
# # -----------------------------
# df_limpo = pd.DataFrame({"Item": resultado})
# df_limpo.to_excel("produtos_saneados.xlsx", index=False)
# print(f"Saneamento concluído! {len(produtos)} itens → {len(resultado)} itens.")

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =====================
# Função auxiliar para comparar medidas
# =====================
def medidas_iguais(med1, med2):
    """
    Retorna True se as medidas são iguais ou muito próximas.
    Ajuste a lógica conforme sua necessidade.
    """
    if pd.isna(med1) or pd.isna(med2):
        return True  # considerar iguais se não tiver medida
    return med1.strip() == med2.strip()

# =====================
# Carregar dados
# =====================
df = pd.read_excel("items2.xlsx")
df["Item"] = df["Item"].astype(str)
# Se tiver coluna de medidas, use:
# df["Medidas"] = df["Medidas"].astype(str)

# =====================
# Gerar embeddings
# =====================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(df["Item"].tolist(), show_progress_bar=True, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

# =====================
# Criar índice FAISS
# =====================
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # produto interno = cosseno, embeddings já normalizados
index.add(embeddings)

# =====================
# Configurar busca de vizinhos
# =====================
k = 10  # número de vizinhos mais próximos a verificar
threshold = 0.8  # similaridade mínima

# =====================
# Encontrar vizinhos e agrupar
# =====================
n = len(df)
usados = set()
agrupados = []

distances, indices = index.search(embeddings, k)

for i in range(n):
    if i in usados:
        continue

    grupo = [i]

    for j_idx, sim in zip(indices[i], distances[i]):
        if j_idx == i or j_idx in usados:
            continue
        if sim >= threshold:
            # Aqui você pode comparar medidas se tiver
            # Ex: medidas_iguais(df.at[i,"Medidas"], df.at[j_idx,"Medidas"])
            grupo.append(j_idx)
            usados.add(j_idx)

    usados.add(i)
    agrupados.append(grupo)

# =====================
# Criar planilha final
# =====================
resultado = []
for grupo in agrupados:
    # Seleciona o item “mais longo” do grupo (ou outra regra)
    itens = [df.at[i, "Item"] for i in grupo]
    melhor_item = max(itens, key=len)
    resultado.append(melhor_item)

df_result = pd.DataFrame({"Item": resultado})
df_result.to_excel("produtos_saneados.xlsx", index=False)
print("Planilha de produtos saneados gerada!")
