import numpy as np

# 1. Read text from a file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

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