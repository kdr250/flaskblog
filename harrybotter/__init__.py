# テスト
import tensorflow as tf
# import keras
from keras.backend import tensorflow_backend as backend
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
# import sys
import io
from janome.tokenizer import Tokenizer
import pickle


graph = tf.get_default_graph()
model=load_model('flaskblog/harrybotter/harry_wakati.h5')

with open("flaskblog/harrybotter/wakati_harry.picle", mode="rb") as f:
    wakati_data = pickle.load(f)

# from flaskblog import model, wakati_data, graph

# keras.backend.clear_session()


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def bot_run(input_words=""):
    # graph = tf.get_default_graph()

    chars = sorted(list(set(wakati_data)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))


    # cut the words in semi-redundant sequences of maxlen words
    maxlen = 20
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(wakati_data) - maxlen, step):
        sentences.append(wakati_data[i: i + maxlen])
        next_chars.append(wakati_data[i + maxlen])

    # Function invoked at end of each epoch. Prints generated word.
    print()
    print('----- Generating word after Epoch:')

    tokenizer = Tokenizer()

    start_index = random.randint(0, len(wakati_data) - maxlen - 1)
    # start_index = 0  # 毎回、「彼は老いていた」から文章生成
    for diversity in [0.2]:  # diversity = 0.2 のみとする
        print('----- diversity:', diversity)

        generated = ''
        # sentence = wakati_data[start_index:start_inde+maxlen]
        # sentence = ''.join(wakati_data[start_index:start_inde+maxlen])
        # input_words = "クィディッチ"
        # input_words = "箒は得意"
        # input_words = "菩薩は得意だよ"
        # input_words = "君の目にマルフォイしている"
        input_words = input_words
        input_word = tokenizer.tokenize(input_words, wakati=True)
        input_word = list(val_word for val_word in input_word if val_word in wakati_data)
        if len(input_word) > 20:
            input_word = input_word[-20:]

        sentence = wakati_data[start_index:start_index+maxlen]
        generated += ''.join(input_word)
        print('----- Generating with seed: "' + ''.join(input_word) + '"')
        # sys.stdout.write(generated)

        for i in range(80):
            # keras.backend.clear_session()
            x_pred = np.zeros((1, maxlen, len(chars)))
            if i == 0:
                arg_str = input_word
            else:
                arg_str = sentence
            for t, char in enumerate(arg_str):
                x_pred[0, t, char_indices[char]] = 1.

            global graph
            with graph.as_default():
                preds = model.predict(x_pred, verbose=0)[0]
                
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            # print("next_char= ", next_char)

            generated += next_char
            sentence = sentence[1:]
            sentence.append(next_char)
            # sentence = wakati_data[next_char]

            
            # sys.stdout.write(next_char)
            # sys.stdout.flush()
            if (next_char == "！") or (next_char == "？") or (next_char == "。"):
                break

    return generated

if __name__ == "__main__":
    input = input("ハリーポッターに出てくるようなワードを入力してください：")
    print(run(input))