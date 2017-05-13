import cv2
import numpy as np
from cv2 import ximgproc
import os
import time
import ast ;
import sys ;
from os import listdir
from os.path import isfile, join
pDollar = ximgproc.createStructuredEdgeDetection( './model.yml.gz' ) ;

cv2.namedWindow("img", cv2.WINDOW_NORMAL) ;
cv2.resizeWindow("img",600,600) ;
cv2.namedWindow("edges", cv2.WINDOW_NORMAL) ;
cv2.resizeWindow("edges",600,600) ;
cv2.namedWindow("orig_edges", cv2.WINDOW_NORMAL) ;
cv2.resizeWindow("orig_edges",600,600) ;
def main() :

	
	mypath = './Images/Non-Digital_Orig/' ;
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] ;
	i = 1 ;
	edge_map = {} ;

	for file in onlyfiles :


		if file != '21924000004125259.jpeg' :
			continue ;
			
		img = cv2.imread( mypath + file ) ;
		#print 'Structured Forests for Fast Edge Detection' ; 
		print (file) ;
		print (img.shape) ;		
		img = np.float32(img) ;
		img = img / 255 ;
		#print img.dtype ;
		#print img.shape ;
		edges = pDollar.detectEdges( img ) ;
		edge_map[file] = np.max(edges) ;
		maxV = np.max(edges) ; minV = 0 ;
		orig = edges.copy() ;
		thresh_val = 0.84 ; # edges usually has a intensity greater than 0.84
		# apply average thresholding , if the max thresh in the image is > thresh_val
		if maxV > thresh_val :
			avg = ( maxV - minV ) / 2 ; 
			# orig[ orig > avg ] = 255 ;
			# orig[ orig <= avg ] = 0 ;
		else :
			avg = thresh_val ;

		edges[edges > avg] = 255; 
		edges[edges <= avg] = 0; 
		cv2.imshow('img',img) ;
		cv2.imshow('orig_edges',orig) ;
		cv2.imshow('edges',edges) ;
		cv2.waitKey(0) ;
		#edges = edges * 255 ;
		print i ;
		#cv2.imwrite('../Structured_Forests/All/' + str(i) + '.png' , edges) ;
		i += 1 ;


	file = open('./Structured_Forests/non_digital.txt','w+') ;
	file.write(str(edge_map)) ;

main() ;

def calc(inp_map) :

	max_v = -1 ; mean_v = 0 ;
	for k , v in inp_map.items() :
		mean_v += v ;
		if v > max_v :
			max_v = v ;

	mean_v = mean_v / len(inp_map) ;	

	return max_v , mean_v ;

def find_edges(img) :

	img = np.float32(img) ;
	img = img / 255 ;
	edges = pDollar.detectEdges( img ) ;
	return edges ;

def analyse() :

	f1 = open('./Structured_Forests/digital.txt','r') ;
	f2 = open('./Structured_Forests/non_digital.txt','r') ;

	dig_map = ast.literal_eval(f1.read()) ;
	non_dig_map = ast.literal_eval(f2.read()) ;

	dig_max_v , dig_mean_v = calc(dig_map) ;		
	print 'Digital ' , dig_max_v , dig_mean_v ;

	ndig_max_v , ndig_mean_v = calc(non_dig_map) ;		
	print 'Non-digital ' , ndig_max_v , ndig_mean_v ;


	cv2.namedWindow("img", cv2.WINDOW_NORMAL) ;
	cv2.resizeWindow("img",600,600) ;
	cv2.namedWindow("edges", cv2.WINDOW_NORMAL) ;
	cv2.resizeWindow("edges",600,600) ;

	for k , v in non_dig_map.items() :
		if v < ndig_mean_v :
			print k , ': ' , v ;
			img = cv2.imread('./Images/Non-Digital_Orig/' + k) ;
			edges = find_edges(img) ;			
			cv2.imshow('img' , img) ;
			cv2.imshow('edges' , edges) ;
			cv2.waitKey(0) ;

#analyse() ;

