from flask import Flask, jsonify, request, render_template
from controllers import *
from summarizer import Summarizer
import torch
#import pytube
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/video", methods=['GET', 'POST'])
def get_video_url():
	for key, value in request.form.items():
		if key == 'url': 
			url = value
	
	path = download(url)
	audio = convertAudio(path)
	text = generateText(audio)
	summary = textsummary(text)
	return render_template('result.html', url=url, text=text, summary=summary)

if __name__=="__main__":
	app.run(debug=True)