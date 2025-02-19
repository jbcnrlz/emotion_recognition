import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 🎭 Simulando dados de estados emocionais
np.random.seed(42)

# Definição dos estados emocionais com médias e covariâncias diferentes
classesDist = np.array([
    [0,0,0,0],
    [0.81,0.21,0.51,0.26], #happy
    [-0.63,0.23,-0.27,0.34], #sad
    [0.4,0.3,0.67,0.27], #surprised
    [-0.64,0.2,0.6,0.32],#fear
    [-0.6,0.2,0.35,0.41],#disgust
    [-0.51,0.2,0.59,0.33],#angry
    [-0.23,0.39,0.31,0.33]#contempt
])

emotions = {"neutral" : [],"happy" : [] ,"sad" : [],"surprised" : [],"fear" : [],"disgust":[],"angry":[],"contempt": []}
idx = -1
covm = []
means = []
for k in emotions:
    idx += 1
    emotions[k] = [[classesDist[idx][0],classesDist[idx][2]], [[classesDist[idx][1]**2,0],[0,classesDist[idx][3]**2]]]
    covm.append([[classesDist[idx][1]**2,0],[0,classesDist[idx][3]**2]])
    means.append([classesDist[idx][0],classesDist[idx][2]])
'''
emotions = {
    "Neutro": [[0, 0], [[0.00, 0], [0, 0.00]]],
    "Feliz": [[0.81, 0.751], [[0.02, 0.01], [0.01, 0.02]]],
    "Triste": [[0.2, 0.3], [[0.02, -0.01], [-0.01, 0.02]]],
    "Estressado": [[0.5, 0.1], [[0.03, 0], [0, 0.02]]],
    "Calmo": [[0.6, 0.8], [[0.02, 0], [0, 0.02]]]
}
'''
# Gerando amostras para cada estado emocional
X = []
labels = []
for i, (emotion, (mean, cov)) in enumerate(emotions.items()):
    samples = np.random.multivariate_normal(mean, cov, 100)
    X.append(samples)
    labels.extend([i] * len(samples))

X = np.vstack(X)
labels = np.array(labels)

# 📊 Treinando o GMM
n_components = len(emotions)  # Número de estados emocionais
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(X)
#gmm.covariances_ = np.array(covm)
#gmm.means_ = np.array(means)

# 🔍 Testando um novo ponto (valence, arousal)
new_point = np.array([[0.55, 0.75]])  # Exemplo de um ponto para classificação
probs = gmm.predict_proba(new_point)[0]
predicted_label = np.argmax(probs)
predicted_emotion = list(emotions.keys())[predicted_label]
emos = np.array(list(emotions.keys()))
print(f"Probabilidades por estado emocional: {dict(zip(emotions.keys(), probs))}")
print(f"Estado emocional mais provável: {predicted_emotion}")
print(f"Rank emotions: {emos[(-probs).argsort()]}")
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
# 📊 Visualizando os clusters e o novo ponto
for idx, l in enumerate(labels):
    plt.scatter(X[idx, 0], X[idx, 1], c=colors[labels[idx]], cmap='viridis', alpha=0.5)
plt.scatter(classesDist[:,0],classesDist[:,2],color='purple',marker='x',s=200,label="Distribution center")
plt.scatter(new_point[:, 0], new_point[:, 1], color='red', marker='x', s=200, label="New point")
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.title("Clusters de Emoções via GMM")
plt.legend()
plt.show()