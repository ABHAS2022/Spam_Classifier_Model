from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.snowball import SnowballStemmer
word = SnowballStemmer('english')
tf = pickle.load(open("vectorizer.pkl","rb"))
mnb = pickle.load(open("mnb.pkl","rb"))

def transformed_text(text,tf,mnb):
    text = nltk.word_tokenize(text)
    count = 0
    for i in text:
        text[count] = i.lower()
        text[count] = word.stem(str(i))
        count = count + 1
        if i in string.punctuation:
            text.remove(i)
    for i in text:
        if i in stopwords.words('english'):
            text.remove(i)
    s = " ".join(text)
    l = tf.transform([s])
    pred = mnb.predict(l)
    if (pred == 1):
        return 'Spam'
    return "Not Spam"

# Import the Flask class from the flask module
from flask import Flask, render_template

# Create a new instance of the Flask class
app = Flask(__name__)

# Define the route for the home page
@app.route('/nothing')
def home():
    # Render the home.html template
    # return render_template('home.html')
    return "Hello World!!"

@app.route('/')
def fun1():
    return render_template('template.html')

@app.route('/classify', methods = ['POST'])
def fun():
    text = request.form.get('email')
    prediction = transformed_text(text,tf,mnb)
    return render_template('output.html', prediction=prediction)
    
# Run the app if this file is executed as the main script
if __name__ == '__main__':
    app.run(debug=True)
