from flask import Flask, render_template, redirect, request

import stand
import joblib

# __name__ == __main__
app = Flask(__name__)
model = joblib.load("emotion.pkl")

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

import re
import json
def preprocess_sentence(w):
    w = re.sub(r"generated_text+","",w) #removing "text" from every sentence
    w = re.sub(r"\n+","",w) #removing "text" from every sentence
    #w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w=re.sub(r'@\w+', '',w)
    return w

@app.route('/')
def hello():
	return render_template("index.html")

caption = ""
@app.route('/', methods= ['POST'])
def marks():
	if request.method == 'POST':

		f = request.files['userfile']
		path = "./static/{}".format(f.filename)# ./static/images.jpg
		f.save(path)

		caption = stand.Caption_this_image(path)
		emotion = model.predict([caption])[0]
		x = generator(caption, max_length=150, num_return_sequences=1)
		x = json.dumps(x[0])
		x = preprocess_sentence(x)
		#print(caption)
		#caption = emstd.predict(caption)
		
		result_dic = {
		'image' : path,
		'caption' : x,
		'emotion' : emotion
		}
	return render_template("index.html", your_result =result_dic)

@app.route('/', methods= ['re'])
def mark():
	if request.method == 're':
		x = generator(caption, max_length=150, num_return_sequences=1)
		x = json.dumps(x[0])
		x = preprocess_sentence(x)

		result_dic = {
		'caption' : x
		}

	return render_template("index.html", your_result =result_dic)


if __name__ == '__main__':
	# app.debug = True
	# due to versions of keras we need to pass another paramter threaded = Flase to this run function
	app.run(debug = False, threaded = False)
