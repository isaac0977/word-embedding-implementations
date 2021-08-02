import multiprocessing
from gensim.models import Word2Vec

cores = multiprocessing.cpu_count()

sentences = []
with open("data/brown.txt", "r") as f:
  for line in f:
    sentences.append(line.rstrip().lower().split(' '))

window_size = [2, 5, 10, 20, 30]
dim_size = [50, 100, 300, 400, 500, 600, 700, 800]
negative_samples = [1, 5, 15, 20, 30, 40]

for i in range(10000):
	for window in window_size:
		for dim in dim_size:
			for negative in negative_samples:
				model = Word2Vec(sentences=sentences, size=dim, window=window, min_count=1, negative=negative, workers=cores)
				model_name = "w2v_window" + str(window) + "negative" + str(negative) + "dim" + str(dim) + ".model"
				print("saving ", model_name) 
				model.wv.save(model_name)
