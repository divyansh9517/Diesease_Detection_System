#import mysql.connector

from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# importing model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model_linear.pkl', 'rb'))


# function to transfrom user input
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('service.html')


@app.route('/predict', methods=['POST'])
def predict():
    ui1 = request.form['Symptom1']
    ui2 = request.form['Symptom2']
    ui3 = request.form['Symptom3']

    ui = ui1 + ui2
    ui = ui + ui3
    user_input = str(ui)

    # 1 Preprocess
    transformed_input = transform_text(user_input)

    # 2 Vectorization

    vector_input = tfidf.transform([transformed_input])

    # 3 Predict
    result = model.predict(vector_input)[0]



"""
    cnx = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Secure@rea1",
        database="symptoms",
        auth_plugin='mysql_native_password'
    )

    cursor = cnx.cursor()
    query = "INSERT INTO sympt (symptom, label) VALUES (%s, %s)"
    data = (user_input, str(result))
    cursor.execute(query, data)
    cnx.commit()
    cursor.close()
    cnx.close()
    return render_template('blog.html', data=result)



if __name__ == "__main__":
    app.run(debug=True)
    """
