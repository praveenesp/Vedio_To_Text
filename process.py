from pytube import YouTube
import moviepy.editor as mp
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import math
import wave
import contextlib
from pydub.utils import make_chunks
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from summarizer import Summarizer
import torch
import glob
import re

def download(url, outpath="C:\\Users\\PRAVEEN-PP\\OneDrive\\Desktop\\video_text\\videos"):
	yt = YouTube(url)
	print("Downloading video...")
	path = yt.streams.filter(file_extension="mp4").get_by_resolution("360p").download(outpath)
	print("Path of the downloaded file:",path)
	return path

def convertAudio(path):
	print("Converting to audio file")
	clip = mp.VideoFileClip(path)
	clip.audio.write_audiofile("C:\\Users\\PRAVEEN-PP\\OneDrive\\Desktop\\video_text\\audios\\test.wav",codec='pcm_s16le')
	print("Audio file has been generated")
	return 'C:\\Users\\PRAVEEN-PP\\OneDrive\\Desktop\\video_text\\audios\\test.wav'

def generateText(path):
    print("1")
    audio = AudioSegment.from_file(path)
    segment_duration = 10000
    start_time = 0
    r = sr.Recognizer()
    l=[]
    while start_time < len(audio):
        end_time = start_time + segment_duration
        if end_time > len(audio):
            end_time = len(audio)
        segment = audio[start_time:end_time]
        segment.export("temp_audio_file.wav", format="wav")
        
        segment_duration = 10000
        with sr.AudioFile("temp_audio_file.wav") as source:
               audio_data = r.listen(source)
               try:
                      text = r.recognize_google(audio_data)
                      l.append(text)
                      print("Segment from {} to {}: {}".format(start_time, end_time, text))
               except:
                      print("not recognized")
               start_time = end_time
               text='.'.join(l) 
    return text

def generateSummaryAb(text, x = 0.25):
	print("Executing Abstractive Method")
	tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-wikihow")
	model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-wikihow")
	listTemp = []
	main_listText = []
	inputList = list(text.split('.'))
	l = len(inputList)
	chunk_size = math.ceil(l * x)      
	i = 0
	j = chunk_size - 1
	iterations = l / (chunk_size)
	while i < l:
		listTemp.append(inputList[i])
		if len(listTemp) >= chunk_size:
			main_listText.append('.'.join(listTemp))
			listTemp = []
		i += 1
	if len(listTemp) != 0:
		main_listText.append('.'.join(listTemp))
	token = tokenizer(main_listText, truncation = True, padding = "longest", return_tensors = "pt")
	summary = model.generate(**token)
	i = 0
	summaryText = ""
	while(i < len(summary)):
		summaryText += tokenizer.decode(summary[i])
		i += 1
	print("Summary:")
	print(summaryText)
	clean_text=re.sub('<.*?>','',summary)
	return clean_text
	
def clearFiles(path = 'C:\\Users\\PRAVEEN-PP\\OneDrive\\Desktop\\video_text\\audios\\chunks'):
	files = glob.glob(path)
	for f in files:
		os.remove(f)

