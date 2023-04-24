import re
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



def tokenize(message):
    """
    Tokeniza a mensagem em uma lista de palavras.
    """
    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)
    return set(all_words)


def get_subject_data(path):
    """
    Lê todos os arquivos de emails no diretório especificado e extrai o assunto (subject) de cada email,
    identificando se é spam ou não com base no nome do arquivo (spam.txt ou ham.txt).
    """
    data = []
    subject_regex = re.compile(r"^Subject:\s+")
    for fn in glob.glob(path):
        is_spam = "ham" not in fn
        with open(fn, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                if line.startswith("Subject:"):
                    subject = subject_regex.sub("", line).strip()
                    data.append((subject, is_spam))
    return data


class Classifier:
    """
    Classe para treinar e realizar a classificação de spam em mensagens de email.
    """
    def __init__(self):
        """
        Inicializa o classificador com um modelo de Naive Bayes Gaussiano e uma lista vazia de features.
        """
        self.used_features = []
        self.gnb = GaussianNB()

    def msgs_to_data_frame(self, data):
        """
        Converte a lista de mensagens e seus respectivos rótulos (spam ou não) em um DataFrame do Pandas.
        """
        r = []
        for message, is_spam in data:
            words = list(tokenize(message))
            d = {word: 1 for word in words}
            d.update({'__is_spam': is_spam, '__message': message})
            r.append(d)
        return pd.DataFrame(r).fillna(0)

    def train(self, data):
        """
        Treina o classificador com os dados fornecidos, identificando as features mais relevantes para a classificação.
        """
        self.used_features = data.columns.drop(['__is_spam', '__message'])
        self.gnb.fit(data[self.used_features], data['__is_spam'])

    def predict(self, msg):
        """
        Realiza a classificação da mensagem fornecida como spam ou não.
        """
        words = tokenize(msg)
        words = [w for w in words if w in self.used_features]
        arr = pd.Series(index=self.used_features).fillna(0)
        arr[words] = 1
        return self.gnb.predict([arr])[0]


msgs = get_subject_data("emails/*/*")
train, test = train_test_split(msgs, test_size=0.25, random_state=1)

classifier = Classifier()
train_data = classifier.msgs_to_data_frame(train)
classifier.train(train_data)

for message, is_spam in test:
    prediction = classifier.predict(message)
    print(f"Mensagem: {message}")
    print(f"Classificação: {'spam' if prediction else 'não spam'}")
    print(f"Esperado: {'spam' if is_spam else 'não spam'}")
    print()
    
predictions = []
true_labels = []
for message, is_spam in test:
    prediction = classifier.predict(message)
    predictions.append(prediction)
    true_labels.append(is_spam)

cm = confusion_matrix(true_labels, predictions)
print(cm)    
 