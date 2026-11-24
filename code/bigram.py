import numpy as np

# 1. Read text from a file
with open("../data/training_data.txt", "r", encoding="utf-8") as f:
    text = f.read()
text = text.replace('\n', '') # getting rid of newline symbols

# 2. Create a set of all unique characters
chars = sorted(set(text)) # sorting to making mapping reproducible across executions

# 3. Create a map of all characters and their indexes
char_indexes = {}
for ind, char in enumerate(chars):
    char_indexes[char] = ind

# 4. Create a map of element transitions (transitions[a][b] = count of "a followed by b" ocasions)
transition_counts = np.zeros((len(chars), len(chars)), dtype=np.int64)

# 5. Count transitions
for i in range(len(text) - 1):
    cur, nxt = text[i], text[i + 1]
    transition_counts[char_indexes[cur], char_indexes[nxt]] += 1
    
print("Character mapping:", char_indexes)
print("Transition counts:")
print(transition_counts)
print("Total transitions:", transition_counts.sum())

# 6. Now we have to find probabilitis for each transition
row_sums = transition_counts.sum(axis = 1, keepdims=True)
transition_probabilities = np.zeros_like(transition_counts, dtype=float)
np.divide(
    transition_counts,
    row_sums,
    out=transition_probabilities,
    where=row_sums != 0
)
print(transition_probabilities)
print("Probability row sums:", transition_probabilities.sum(axis=1, keepdims=True))

# Saving Transition probability matrix and char - id map    
np.savez("../artifacts/weights.npz", 
         charArray=np.array(chars), 
         transitionProbabilitiesMatrix=transition_probabilities
         )



