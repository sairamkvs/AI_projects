import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Sample text data
corpus = [
    "This is the first sentence.","And this is the second sentence.","Finally, the third sentence."
]

# Tokenizing and creating sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding sequences
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Creating predictors and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.array([[1 if i == word_index else 0 for i in range(total_words)] for word_index in y])

# Build the model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_length-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_length, tokenizer, temperature=1.0):
    generated_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list)[0]
        
        # Apply temperature to adjust the randomness
        predicted_probs = np.log(predicted_probs) / temperature
        exp_probs = np.exp(predicted_probs)
        predicted_probs = exp_probs / np.sum(exp_probs)
        
        # Choose the word with the highest probability
        predicted = np.argmax(np.random.multinomial(1, predicted_probs, 1))
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        generated_text += " " + output_word
    
    return generated_text
# Generate text using the trained model
generated_text = generate_text("This is ", 10, model, max_sequence_length, tokenizer)
print(generated_text)
