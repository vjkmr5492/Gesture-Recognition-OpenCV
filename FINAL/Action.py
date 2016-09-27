import os
import subprocess
import ctypes, random

loc='I:/HGR'
actions={}
doit=1

def chrome(url):
	subprocess.call(['C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',url])

def speak(what):
	subprocess.call(['./data/Tools/Say.exe',what])

def wallpaper():
	SPI_SETDESKWALLPAPER = 20 
	ctypes.windll.user32.SystemParametersInfoA(SPI_SETDESKWALLPAPER, 0, loc+'/data/Wallpapers/WP_('+str(random.randrange(1, 25))+').jpg' , 0)

def performaction(strs):
	if doit!=1:
		return
	if strs=='Line': 
		speak("Line detected")
	elif strs=='V': 
		wallpaper()
	elif strs=='Circle': 
		speak("Hello, good morning!")
	elif strs=='L': 
		chrome("http://bit-bangalore.edu.in")
	elif strs=='Alpha': 
		speak("Alpha detected. I need some help.")
	else:
		print strs



