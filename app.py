from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
     return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

data = pd.read_csv(r"C:\Users\divyabharathi\Downloads\archive\fake_news_dataset.csv")
