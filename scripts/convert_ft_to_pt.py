from gensim.models import FastText

model = FastText.load("kubhist2_increm_1880/kubhist2_tokens_1880.ft")

# obtain the full list of words
words = list(model.wv.index_to_key)
dim = model.wv.vector_size

with open("sv.vectors", "w", encoding="utf8") as f:
    f.write(f"{len(words)} {dim}\n")
    for w in words:
        vec = model.wv[w]
        f.write(w + " " + " ".join(map(str, vec)) + "\n")