import numpy as np
import string
from PIL import Image

def my_round(x):
    if abs(x - int(x)) > 0.3:
        return int(x) + 1
    else:
        return int(x)

def sygmoid(x):
    return 1/(1 + np.exp(-x))

# ввод = 256 нейронов
# 1й скрытый слой = 20
# 2й скрытый слой = 20
# результат = 26


learning_rate = 0.1
weight_0_1 = np.random.normal(0.0, 1, (20, 256))
weight_1_2 = np.random.normal(0.0, 1, (20, 20))
weight_2_3 = np.random.normal(0.0, 1, (26, 20))
alphabet = string.ascii_uppercase
matrix_of_expected = np.eye(26)

inp = np.zeros((256,), dtype=int)
expected = np.zeros((26,), dtype=int)
activate = np.vectorize(sygmoid)

for epoches in range(500):
    # np.savetxt('result_weights.txt', result_weight)
    # np.savetxt('hidden_weights.txt', hidden_weight)
    for letter in alphabet:
        for counter in range(4):
            img = Image.open("tests/{0}{1}.jpg".format(letter, counter))
            pix = img.load()
            counter = 0
            for w in range(img.size[0]):
                for l in range(img.size[1]):
                    red = pix[w, l][0]
                    green = pix[w, l][1]
                    blue = pix[w, l][2]
                    absolute = (red + green + blue) / 3
                    absolute = 1 - absolute / 255
                    inp[counter] = my_round(absolute)
                    counter += 1

            #расчёт
            hidden1 = activate(np.dot(weight_0_1, inp))
            hidden2 = activate(np.dot(weight_1_2, hidden1))
            result = activate(np.dot(weight_2_3, hidden2))

            #метод обратного распространения ошибки
            expected = matrix_of_expected[alphabet.index(letter)]
            error_layer3 = result - expected
            weight_delta_3 = result * (1 - result) * error_layer3
            x = (np.dot(weight_delta_3.reshape(len(weight_delta_3), 1), hidden2.reshape(1, len(hidden2))))
            weight_2_3 -= (np.dot(weight_delta_3.reshape(len(weight_delta_3), 1), hidden2.reshape(1, len(hidden2)))) * learning_rate

            error_layer2 = np.dot(weight_delta_3.reshape(1, len(weight_delta_3)), weight_2_3)
            weight_delta_2 = error_layer2 * hidden2 * (1 - hidden2)
            x = np.dot(weight_delta_2.T, hidden1.reshape(1, len(hidden1)))
            weight_1_2 -= np.dot(weight_delta_2.T, hidden1.reshape(1, len(hidden1))) * learning_rate

            error_layer1 = np.dot(weight_delta_2, weight_1_2)
            weight_delta_1 = error_layer1 * hidden1 * (1 - hidden1)
            weight_0_1 -= np.dot(weight_delta_1.T, inp.reshape(1, len(inp))) * learning_rate

img = Image.open("input.jpg")
# img = img.resize((16, 16), Image.ANTIALIAS)
pix = img.load()
counter = 0
for w in range(img.size[0]):
    for l in range(img.size[1]):
        red = pix[w, l][0]
        green = pix[w, l][1]
        blue = pix[w, l][2]
        absolute = (red + green + blue) / 3
        absolute = 1 - absolute / 255
        inp[counter] = my_round(absolute)
        counter += 1

hidden1 = activate(np.dot(weight_0_1, inp))
hidden2 = activate(np.dot(weight_1_2, hidden1))
result = activate(np.dot(weight_2_3, hidden2))

# np.savetxt('result_weights.txt', result_weight)
# np.savetxt('hidden_weights.txt', hidden_weight)

for i in range(len(result)):
    result[i] = round(result[i], 3)

print(result)
