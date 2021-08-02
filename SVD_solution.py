import numpy as np
import pandas
import math
import scipy.sparse as sp
from scipy import linalg

DATA_FN = "data/brown.txt"

# hyperparameters:
#WINDOWS = (2,5,7) # context window size
#for k in WINDOWS:
k = 2 
#DIMENSIONS = (50,100,300)
#for dim in DIMENSIONS:
dim = 50

word_counts = {}
index = {}
n = 0

# count words and create index for each unique word 
with open(DATA_FN, "r") as f:
  for line in f:
    splits = line.rstrip().lower().split()
    for word in splits:
      if not word in word_counts:
        word_counts[word] = 0
        word_counts[word] += 1
      if not word in index:
        index[word] = n
        n += 1


# co-occurence matrix
com = sp.lil_matrix((n,n))
with open(DATA_FN, "r") as f:
  for line in f:
    words = line.strip().lower().split()
    for w in range(len(words)):
      for c in range(max(0,w-k),min(w+k+1,len(words))):
        if not c == w:
          com[index[words[w]],index[words[c]]] += 1
# (for testing)
z = 2
while com[1,z] == 0:
  z += 1
print(com[1,z])
for key in index:
  if index[key] == 1:
    print(key)
  if index[key] == z:
    print(key)

# m = max(0, pmi(val in co-occurence matrix))
# multiply each count by |D| - size of collection of word-context pairs, 
  #|D| = sum of values in matrix/2
d = com.sum() / 2
print("sum")
m = com*d
print("com")
d = None
# divide each cell by count(w_i)*count(w_j) (Levy,4)
  # count(w) = sum(w,c') over c (Levy, 2)
m = m/com.sum(0)
print("m")
m = m.transpose()/com.transpose().sum(0)
m = m.transpose()


# take log of each cell 
    # !!!!! log 0 = -infinity; if cell_val >= 1, take the log, else cell_val = 0 
for r in range(n):
  for c in range(n):
    if m[r,c] >= 1:
      m[r,c] = math.log(m[r,c])
    else:
      m[r,c] = 0

# SVD
U, s, VT = linalg.svd(m)
Uk = U.transpose()[:dim].transpose()
sk = s[:dim]
sk_sqrt = linalg.sqrtm(sk*np.identity(dim))
W = np.matmul(Uk,sk_sqrt)

