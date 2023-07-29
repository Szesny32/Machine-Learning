from sklearn.cluster import KMeans
from sklearn.preprocessing import scale # to normalize the data
from sklearn.datasets import load_digits #handwritten digits from 0 to 9 scanned


digits = load_digits()
data = scale(digits.data)


model = KMeans(m_clusters=10, init = 'random', n_init=10)
model.fit(data)

# model.predict([...])