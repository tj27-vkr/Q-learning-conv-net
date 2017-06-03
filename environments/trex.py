import cv2
from PIL import ImageGrab
import numpy as np
import time
import os
import win32com.client
shell = win32com.client.Dispatch("WScript.Shell")

GAME_WINDOW = (519, 287, 877, 645)
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.png")
class GameState:
	def __init__(self):
		shell.AppActivate("Chrome")
		self.frame  = ImageGrab.grab(bbox=GAME_WINDOW)
		self.game_over = cv2.imread(TEMPLATE, 0)
		
	def frame_step(self, input_actions):
		reward = 0.1
		terminal = False
		
		# input actions [1, 0]: nothing [0, 1]: trex jump
		if input_actions[1] == 1:
			send_command()
			
		shell.AppActivate("Chrome")
		image = ImageGrab.grab(bbox=GAME_WINDOW)
		self.frame  = image
		
		#if the game is over, decrease reward
		template_match = cv2.cvtColor(np.array(self.frame), cv2.COLOR_BGR2GRAY)
		res = cv2.matchTemplate(template_match,self.game_over,cv2.TM_CCOEFF_NORMED)
		#print(res)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		#print(max_loc)
		#template_match = np.array(self.frame)[337:358, 184:259]
		#template_match = cv2.cvtColor(template_match, cv2.COLOR_BGR2GRAY)
		
		if max_loc == (184, 337):
			print("Game Over")
			reward = -0.1
		#cv2.imshow("win",np.array(self.frame))
		#cv2.waitKey(1)
		#cv2.imwrite("test.png", np.array(self.frame))
		return np.array(self.frame), reward, terminal
		
def send_command():
	shell.AppActivate("Chrome")
	shell.SendKeys(" ",0)
	time.sleep(0.2)
	
if __name__ == "__main__":
	env = GameState()
	while True:
		env.frame_step([0,1])
		#break