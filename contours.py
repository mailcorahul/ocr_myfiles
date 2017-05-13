import sys
import math
import numpy as np
import cv2 
from os import listdir
from os.path import isfile, join
h = 0 ; w = 0 ;
def lineDist(p1,p2) :

	x = (p2[0] - p1[0]) ** 2 ;
	y = (p2[1] - p1[1]) ** 2 ;
	return math.sqrt(x + y) ;

def findExtremeLine(box,lines) :

	p1 = box[lines[0][0]] ; 
	p2 = box[lines[0][1]] ;
	p3 = box[lines[1][0]] ; 
	p4 = box[lines[1][1]] ;
	lineI = -1 ;
	# horizontal 
	if abs(p1[0] - p2[0]) > abs(p1[1] - p2[1]) :
		if abs(p1[1] - 0) < abs(p1[1] - h) :
			if min(p1[1] , p2[1]) < min(p3[1] , p4[1]) :
				lineI = 0 ;
			else :
				lineI = 1 ;
		else :
			if max(p1[1] , p2[1]) > max(p3[1] , p4[1]) :
				lineI = 0 ;
			else :
				lineI = 1 ;
	# vertical 
	elif abs(p1[0] - p2[0]) < abs(p1[1] - p2[1]) :
		isX = True ;
		if abs(p1[0] - 0) < abs(p1[0] - w) :
			if min(p1[0] , p2[0]) < min(p3[0] , p4[0]) :
				lineI = 0 ;
			else :
				lineI = 1 ;
		else :
			if max(p1[0] , p2[0]) > max(p3[0] , p4[0]) :
				lineI = 0 ;
			else :
				lineI = 1 ;

	return lineI ;

def findContours(onefile) :

	mypath = "./EdgeDetection_Output/" ;
	dest = "./Curves_to_Lines/" ;
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] ;
	global h , w ;
	count = 1 ;
	for file in onlyfiles :

		print 'File ' , count ;
		count += 1 ;
		# if file != onefile :
		# 	continue ;
		im = cv2.imread(mypath + file) ;
		im = cv2.imread('./test/821.png') ;
		h , w = im.shape[:2] ;
		cv2.imshow("orig",im) ;
		imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		ret,thresh1 = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
		im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		rects = sorted(contours, key = cv2.contourArea, reverse = True)
		copy = im.copy() ;
		for c in rects:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)

			# curvy line of length 2 
			if len(approx) == 2 :
				rect = cv2.minAreaRect(c)
				if rect[1][0] > rect[1][1] :
					nmax = rect[1][1] ;
				else :
					nmax = rect[1][0] ;

				box = cv2.boxPoints(rect)
				box = np.int0(box)
				dist = {} ; pts = [] ;
				for i in range(len(box)) :
					for j in range(i + 1 , len(box)) :
						temp = copy.copy() ;
						d = lineDist(box[i] , box[j]) ;
						if d in dist :
							dist[d].append([i , j]) ;
						else :
							dist[d] = [] ;
							dist[d].append([i , j]) ;
						#print d , box[i] , box[j] ;
						# cv2.line(temp,(box[i][0],box[i][1]),(box[j][0],box[j][1]),[0,255,0],2)
						# cv2.imshow('temp',temp);
						# cv2.waitKey(0) ;

				dist = sorted(dist.items()) ;
				#print dist ;
				lines = [] ;
				ptI = [] ;
				c = 0 ;
				for item in dist :		
					# line with smallest distance found		
					for it in item[1] :
						ptI.append(it) ;
						c += 1 ;
						if c == 2 :
							break ;
					if c == 2 :
						break ;

				# finding a line which approximates the curve 
				for pt in ptI :
					p1 = box[pt[0]] ;
					p2 = box[pt[1]] ;
					mX = (p1[0] + p2[0] )/ 2 ; mY = (p1[1] + p2[1]) / 2 ;
					lines.append( [mX,mY] ) ;

				# blacking out the curve
				cv2.drawContours(copy,[box],0,(0,0,0),-1) # prev :-  copy = ...
				cv2.line(copy,(lines[0][0],lines[0][1]),(lines[1][0],lines[1][1]),[255,255,255],2) ;
				# cv2.imshow('Curve to Line',copy);
				# cv2.waitKey(0) ;
				# lineI = findExtremeLine(box,lines) ;
				
				# print 'Extreme Lines' , box[lines[lineI][0]] , box[lines[lineI][1]] ;
				# box = np.array( [box[lines[lineI][0]] , box[lines[lineI][1]]] ) ;
				# copy = cv2.drawContours(copy,[box],0,(0,0,255),2)
				# cv2.imshow('Cnt',copy) ;
				# cv2.waitKey(0) ;
		print 'Writing ' , file ;
		#cv2.imwrite(dest + file,copy) ;
		cv2.imwrite('./test/891_1.png',copy) ;

if len(sys.argv) == 2 :
	findContours(sys.argv[1]) ;
else :
	findContours(-1) ;