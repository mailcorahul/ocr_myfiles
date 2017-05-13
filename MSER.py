import cv2
import numpy as np 
#from text_cnn import classifyMSER ;
from os import listdir
from operator import itemgetter ;
from sklearn.cluster import KMeans
import os 
from os.path import isfile, join
import rlsa , weighted_distance 
import time 
import math ; 
import msers_filter ;
import msers_filters ;
import sys

colors = [[0,0,0],[255,0,0],[0,255,0],[0,0,255]] ;
rightP = [] ; leftP = [] ;
single = False ;

# params
# orig - starting mser points  , mser - mser filled image , win - sliding window points , left - 'slide left/right'(bool)
def getBoundaryPoints(mser , orig , win , left) : 

	include = True ;
	h , w = mser.shape ;
	#minX = win[0] ; minY = win[1] ; maxX = win[2] ; maxY = win[4] ;
	minX , minY , maxX , maxY = win[:4] ;
	nminX , nminY , nmaxX , nmaxY = win[:4] ;
	# search top	
	i = minX ; j = maxY ;
	first = False ;

	while (j >= 0) :

		# first black pixel found
		if ( np.amin(mser[j,minX : maxX]) == 0 ) and first == False :
			first = True ;
		elif ( np.amin(mser[j,minX : maxX]) != 0 ) and first == True :
			stY = j ;
			# update the minY param ( check for height change )
			if j < minY :
				nminY = j ;
			break ;

		j -= 1 ;
	
	# search bottom 
	i = minX ; j = minY ;
	first = False ;
	while (j < h) :

		# first black pixel found
		if ( np.amin(mser[j,minX : maxX]) == 0 ) and first == False :
			first = True ;
		elif ( np.amin(mser[j,minX : maxX]) != 0 ) and first == True :
			enY = j ;
			# update the maxY param ( check for height change )
			if j > maxY :
				nmaxY = j ;
			break ;

		j += 1 ;

	# search left 
	if (left) :

		i = maxX ; 
		first = False ;
		while (i >= 0) :

			# first black pixel found
			if ( np.amin(mser[minY : maxY,i]) == 0 ) and first == False :
				first = True ;
				enX = i ;
			elif ( np.amin(mser[minY : maxY,i]) == 0 ) and first == True :
				# update the minX param
				nminX = i ;
				break ;

			i -= 1 ;

		nmaxX = orig[2] ;
		stX = nminX ;
	# search right 
	else :

		i = minX ; j = minY ;
		first = False ;
		while (i < w) :

			# first black pixel found
			if ( np.amin(mser[minY : maxY,i]) == 0 ) and first == False :
				first = True ;
				stX = i ;
			elif ( np.amin(mser[minY : maxY,i]) != 0 ) and first == True :
				# update the maxX param
				nmaxX = i ;
				break ;

			i += 1 ;
		nminX = orig[0] ;
		enX = nmaxX ;

	# stX , stY , enX , enY - represents the near-by mser's bounding rectangle co-ordinates
	width = enX - stX ;
	height = enY - stY ;

	# print orig ;
	# print stX , stY , enX , enY ;


	# if abs( (orig[2] - orig[0]) - width ) > 250 or abs( (orig[3] - orig[1]) - height ) > 250 :
	# 	include = False ;	

	return [ nminX , nminY , nmaxX , nmaxY ] , include ;

def trackText(orig,img,pts,slideW) :

	h , w  = img.shape ;
	
	# search right 
	while True :

		# reached right end
		if ( pts[2] + 1 + slideW > w ) and ( w != pts[2] + 1 ) :
			slideW =  w - pts[2] - 1 ;
		elif ( pts[2] + 1 == w ) :
			break ;

		sli_win = img[pts[1] : pts[3] , pts[2] + 1 : ( pts[2] + 1 ) + slideW ] ;
		# print sli_win.shape ;
		# print pts[1] , pts[3] , pts[2] + 1 , ( pts[2] + 1 ) + slideW ;

		# sliding window has no mser inside 
		if np.amin(sli_win) != 0 :
			break ;

		prev_pts = pts ;
		# if it contains mser , search right and extend the word boundary 
		pts , include = getBoundaryPoints(img , pts , [pts[2] + 1 ,pts[1] ,( pts[2] + 1 ) + slideW ,pts[3] ], False) ;
		
		# if the near-by mser has a different width/height , exclude it and stop searching
		if include == False :
			pts = prev_pts ;
			break ;
		# cv2.rectangle(orig,(pts[0],pts[1]),(pts[2],pts[3]),(0,0,255),3) ;
		# cv2.imshow('orig',orig) ;
		# cv2.waitKey(0) ;

	# search left 
	while True :

		# reached left end
		if (pts[0] - 1 - slideW < 0) and ( pts[0] - 1 != 0 ):
			slideW = pts[0] - 1 ;
		elif ( pts[0] - 1 == 0 ) :
			break ;

		sli_win = img[pts[1] : pts[3] , (pts[0] - 1) - slideW : pts[0] - 1 ] ;

		# sliding window has no mser inside 
		if np.amin(sli_win) != 0 :
			break ;

		prev_pts = pts ;
		# if it contains mser , search left and extend the word boundary 
		pts , include = getBoundaryPoints(img , pts , [(pts[0] - 1) - slideW , pts[1] , pts[0] - 1 , pts[3] ] , True) ;
		
		# if the near-by mser has a different width/height , exclude it and stop searching
		if include == False :
			pts = prev_pts ;
			break ;
		# cv2.rectangle(orig,(pts[0],pts[1]),(pts[2],pts[3]),(0,0,255),3) ;
		# cv2.imshow('orig',orig) ;
		# cv2.waitKey(0) ;

	# pts - word boundary
	return pts ;


def findBRect(img,hulls) :

	rectToarea = {} ;
	rect = [] ;
	boxes = [[],[],[],[]] ;
	i = 0 ;
	imgH , imgW  = img.shape ;
	meanArea = 0 ;

	for hull in hulls :
		#print hull ;
		#areas.append(cv2.contourArea(hull)) ;
		minX = img.shape[1] + 1 ;
		minY = img.shape[0] + 1 ;
		maxX = -1 ;
		maxY = -1 ;
		for h in hull :
			if h[0][0] < minX :
				minX = h[0][0] ;
			if h[0][0] > maxX :
				maxX = h[0][0] ;
			if h[0][1] < minY :
				minY = h[0][1] ;
			if h[0][1] > maxY :
				maxY = h[0][1] ;
		#print minX , minY , maxX , maxY ;
		#areas.append( cv2.contourArea(np.array([[[minX , minY]] ,[[minX , maxY]],[[ maxX , maxY ]] , [[maxX , minY]]])) );
		rectToarea[i] = (maxX - minX) * (maxY - minY) ;#cv2.contourArea(np.array([[[minX , minY]] ,[[minX , maxY]],[[ maxX , maxY ]] , [[maxX , minY]]])) ;
		if rectToarea[i] > (imgH * imgW) / 3 : 
			continue ;

		rect.append([minX,minY,maxX,maxY]) ;
		boxes[0].append(minX) ;
		boxes[1].append(minY) ;
		boxes[2].append(maxX) ;
		boxes[3].append(maxY) ;
		meanArea += rectToarea[i] ;
		
		i += 1 ;
		#cv2.rectangle(vis,(minX,minY),(maxX,maxY),(0,255,0),1) ;
		#val += areas[i] ; 
		#i += 1 ; 
	return rect , boxes , rectToarea ;

def findBRect_C(img,hulls) :

	rectToarea = {} ;
	rect = [] ;
	boxes = [[],[],[],[]] ;
	i = 0 ;
	imgH , imgW  = img.shape ;
	meanArea = 0 ;

	for hull in hulls :
		#print hull ;
		#areas.append(cv2.contourArea(hull)) ;
		minX = hull[0] ; minY = hull[1] ; maxX = hull[2] ; maxY = hull[3] ;
		#print minX , minY , maxX , maxY ;
		#areas.append( cv2.contourArea(np.array([[[minX , minY]] ,[[minX , maxY]],[[ maxX , maxY ]] , [[maxX , minY]]])) );
		rectToarea[i] = (maxX - minX) * (maxY - minY) ;#cv2.contourArea(np.array([[[minX , minY]] ,[[minX , maxY]],[[ maxX , maxY ]] , [[maxX , minY]]])) ;
		if rectToarea[i] > (imgH * imgW) / 3 : 
			continue ;
		rect.append([minX,minY,maxX,maxY]) ;
		boxes[0].append(minX) ;
		boxes[1].append(minY) ;
		boxes[2].append(maxX) ;
		boxes[3].append(maxY) ;
		meanArea += rectToarea[i] ;
		i += 1 ;
		#cv2.rectangle(vis,(minX,minY),(maxX,maxY),(0,255,0),1) ;
		#val += areas[i] ; 
		#i += 1 ; 
	return rect , boxes , rectToarea ;

def delOverlapRects(copy ,rect, ordRects) :

	copy.fill(255) ;
	for i in range(len(ordRects)) :
		pts = rect[ordRects[i][0]] ;
		cropped = copy[pts[1]:pts[3],pts[0]:pts[2]] ;

		# if the area is completely black , then ignore the rect
		if ( np.sum(cropped) == 0 ) :
			rect[ordRects[i][0]] = [] ; # mark rectangle as ignored 
		# enclosing / overlapping rectangle
		else :
			copy[pts[1]:pts[3],pts[0]:pts[2]] = 0 ;

	return rect ;

def findMean() :

	group = [ 0 for i in range(len(rect)) ] ;
	gnum = 1 ;
	img1 = img.copy() ;
	mean_width = 0 ; mean_height = 0 ; 
	widths = [] ; heights = [] ;
	wh_ratio = [] ;
	nzero = 0 ;
	mean_ratio = 0 ;
	for r in rect :
		if len(r) > 0 :
			widths.append(r[2] - r[0]) ;
			heights.append(r[3] - r[1]) ;
			tw = widths[nzero] ;
			th = heights[nzero] ;
			if widths[nzero] < heights[nzero] :
				tw = heights[nzero] ;
				th = widths[nzero] ; 

			wh_ratio.append( float(tw) / th ) ;
			#print widths[nzero] , heights[nzero] , float(widths[nzero])/ heights[nzero] ;
			mean_ratio += wh_ratio[nzero] ;
			mean_width += r[2] - r[0] ;
			mean_height += r[3] - r[1] ;
			nzero += 1 ;

	if nzero > 0 :
		mean_width = mean_width / nzero ;
		mean_height = mean_height / nzero ;
	#print 'Mean : ' , mean_height , mean_width ; 
	std_w = np.std(widths) ; std_h = np.std(heights) ;
	deviation = 10 ;
	mean_ratio = mean_ratio / nzero ;
	std = np.std(wh_ratio) ;
	print 'Mean Ratio : ' , mean_ratio ;
	print 'Std : ' , std ;
	
def ARFilter(rect) :

	i = 0 ;
	# remove rectangles based on aspect ratio
	for br in rect :
		if len(br) > 0 :
			tw = br[2] - br[0] ; th = br[3] - br[1] ;
			if tw < th :
				temp = th ;
				th = tw ;
				tw = temp ;
			if (float(tw) / th) > 5 :
			#print float(br[2] - br[0]) / (br[3] - br[1]) ;
				#print 'Wrong : ' , float(tw) / th ;
				rect[i] = [] ;
		i += 1 ;

	return rect ;


def non_max_suppression(boxes, overlapThresh ,rect) :


	boxes = np.array(boxes) ;
	# if there are no boxes, return an empty list
	if len(boxes) == 0 :
		return []
 
	# initialize the list of picked indexes
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = np.array( boxes[0] ) ;
	y1 = np.array( boxes[1] ) ;
	x2 = np.array( boxes[2] ) ;
	y2 = np.array( boxes[3] ) ;
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1) * (y2 - y1) ; #( x2 - x1 + 1 ) * ( y2 - y1 + 1 )
	#print area[0] , rect[0] ;
	idxs = np.argsort(y2) ;
	skip = 1 ;
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0 :
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos] ;
 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 
 			'''
			# check if one of the boxes is completely contained in the other and doesn't have equal co-ords
			if area[i] != area[j] : #(x1[i] != x1[j] or x2[i] != x2[j] or y1[i] != y1[j] or y2[i] != y2[j]) :
				if ( x1[j] <= x1[i] and x2[j] >= x2[i] and y1[j] <= y1[i] and y2[j] >= y2[i] ) or ( x1[i] <= x1[j] and x2[i] >= x2[j] and y1[i] <= y1[j] and y2[i] >= y2[j] ) :
					#print 'Skipping Suppression' ;
					skip += 1 ;
					continue ; # skip suppression
			'''

			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
 
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh :

				# j is completely enclosed by i
				if x1[i] < x1[j] and y1[i] < y1[j] and x2[i] > x2[j] and y2[i] > y2[j] : 
					# print 'Full Overlap!' ;					
					# print (x1[i],y1[i],x2[i],y2[i]) ;
					# print (x1[j],y1[j],x2[j],y2[j]) ;
					if abs( (y2[i] - y1[i]) - (y2[j] - y1[j]) ) <= 10 :
						if abs( (x2[i] - x1[i]) - (x2[j] - x1[j]) ) > 5 :
							#print 'Possible individual character' ;
							continue ;

				suppress.append(pos) ;
 
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
 
 	#print 'Skipped ' , skip ;
	# return only the bounding boxes that were picked
	return pick ;

# method to find MSERs from a given image and return word boundaries 
def MSER(img , orig , file , folder , scale_factor) :

	vis = orig.copy() ;
	vis1 = img.copy() ;
	im = img.copy() ;
	msers = img.copy() ;
	msers.fill(255) ;
	copy = img.copy() ;

	print '***** MSER ****' ;
	mser = cv2.MSER_create() ;
	#mser = cv2.MSER_create(5 ,60 ,14400 ,0.25, .2 ,200 ,1.01,0.003) ;
	regions = mser.detectRegions(img) ;
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
	#print len(regions[0]) , regions[0][0] , regions[0][1] ;
	#cv2.polylines(vis, hulls[1] , 1, (0,0,255)) ;
	#cv2.drawContours(imcopy,hulls1,-1,(0,0,255),1) ;
	# cv2.imshow("vis",vis) ;
	# cv2.imwrite('./MSER/Plain MSERs/' + file , vis) ;
	# cv2.waitKey(0) ;
	areas = [] ;
	rectToarea = {} ;
	rect = [] ;
	
	val = 0 ;
	print 'shape : ' , img.shape ;

	# find bounding rectangle from msers detected
	rect , boxes ,  rectToarea = findBRect(img,hulls) ;

	# sort rectangles by area
	ordRects = sorted(rectToarea.items(), key=lambda x: x[1] ,reverse = True) ;
	vis1.fill(255) ;
	origc = orig.copy() ;

	for br in rect :
		cv2.rectangle(vis1,(br[0],br[1]),(br[2],br[3]),(0,0,0),-1) ;
		cv2.rectangle(origc,(br[0],br[1]),(br[2],br[3]),(0,0,255),3) ;

	#cv2.imwrite('./MSER/New/WORD/DIGI_2/' + folder + '/' + str(scale_factor) + '_' + 'mser_' + file ,vis1 );	
	#cv2.imwrite('./MSER/New/WORD/DIGI_2/ ' + folder + '/' + str(scale_factor) + '_' + 'orig_' + file ,origc );

	# use a color fill algorithm to remove overlapping rectangles
	#rect = delOverlapRects(copy ,rect , ordRects) ;

	
	# print boxes_idxs ;
	# print len(boxes_idxs) ;
	# print len(boxes[0]) ;

	'''
	print 'Applying Non-Maximal Suppression' ;
	boxes_idxs = non_max_suppression(boxes , 0.75, rectToarea ) ;
	rect = [] ;
	for i in range(len(boxes[0])) :
		if i in boxes_idxs :
			# print boxes[0] ;
			# print boxes[0][i]
			rect.append( [boxes[0][i] , boxes[1][i] , boxes[2][i] , boxes[3][i]] ) ;
	'''

	print 'Applying Aspect Ratio Filter' ;
	# remove rectangles based on aspect ratio
	rect = ARFilter(rect) ;

	# removing empty rects
	rect = filter(None,rect) ;
	w , h = img.shape ;
	i = 1 ;
	print len(rect) ;
	
	cc = orig.copy() ;
	cc.fill(255) ;
	word_bounds = [] ;
	
	# scaling msers to original image shape -- since multiple scales are used
	for j in range(len(rect)) :	
		for i in range(len(rect[j])) :
			rect[j][i] = int( float(rect[j][i]) * scale_factor ) ;

	return rect ;

	'''
	for i in range(len(ordRects)) :
		#copy = img1.copy() ;
		rect_i = ordRects[i][0] ;
		box = rect[rect_i] ;
		if len(box) > 0 :  #and is_text[i] == 0 :

			word = [] ;
			if np.amin( cc[box[1]:box[3],box[0]:box[2]] ) == 0 :
				continue ;
			word = trackText(orig.copy(),vis1,box,30) ; # param 4 - neighbourhood size

			# storing word boundaries
			word_bounds.append(word) ;

			# drawing word boundaries 
			cv2.rectangle(vis,(word[0] , word[1]) , (word[2] , word[3]) ,(0,0,255),3) ;

			# marking the new word found as visible
			cv2.rectangle(cc,(word[0],word[1]) , (word[2] , word[3]) ,(0,0,0),-1) ;
		
	#print 'Word bounds length ' , len(word_bounds) ;
	#print word_bounds ;
	# scaling co-ordinates accordingly
	for j in range(len(word_bounds)) :	
		for i in range(len(word_bounds[j])) :
			word_bounds[j][i] = int( float(word_bounds[j][i]) * scale_factor ) ;

	#print word_bounds ;

	#cv2.imwrite('./MSER/New/WORD/DIGI_2/' + folder + '/' + str(scale_factor) + '_' + file ,vis ) ;

	return word_bounds ;
	'''

def segmentWord(crop) :

	chars = [] ;
	fB = False ;
	orig = crop.copy() ;
	crop = np.array(crop , dtype = np.float32 ) ;

	print 'Total White' , crop.shape[0] * 255 ;
	maxS = crop.shape[0] * 255 ;
	crop = crop / maxS ;
	crop_sum = np.sum( crop , axis = 0 ) ;
	print crop_sum ;
	print np.sum( orig , axis = 0 ) ;
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)) ;
	# crop = 255 - crop ;
	# crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel) ;
	# crop = 255 - crop ; 
	# im2, contours, hierarchy = cv2.findContours(crop,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# contours = sorted(contours, key = cv2.contourArea, reverse = True) ;

	# for c in contours :
	# 	cv2.drawContours(orig,[c],-1,(255,255,255),1)	;
	# 	cv2.imshow('crop' , orig) ;
	# 	cv2.waitKey(0) ;
	# cv2.imshow('crop' , orig) ;
	# cv2.waitKey(0) ;
	# candidate_cols = [] ;
	# prev = 1 ;

	'''
	for i in range(len(crop_sum)) :
		# unusual peak in no. of white pixels ( threshold - 50% increase in white pixels )
		if( crop_sum[i] - prev  >= 0.5 ) :
			# check to include only characters with width greater than 4 column
			if len(candidate_cols) > 0 : 
				if ( i - candidate_cols[len(candidate_cols) - 1] >= 4 ) :
					candidate_cols.append(i) ;
			else :	
				candidate_cols.append(i) ;

	'''
	# simple segmentation algorithm - finds the last black pixel before a white pixel
	for i in range(crop.shape[1]) :

		# first black col
		if np.amin(crop[:,i]) == 0 and fB == False :
			fB = True ;
			fiB = i ;

		# a white col after a black col is found
		elif np.amin(crop[:,i]) != 0 and fB == True :
			fB = False ;
			chars.append(orig[:,fiB : i]) ;

	# if no white col is found 
	if fB == True :
		chars.append(orig[:,fiB : orig.shape[1]]) ;


	return chars ;
	
def sortMSERs(boxes) :

	#print boxes[:4] ;
	t_b_boxes = sorted(boxes, key=itemgetter(0)) ;
	#print t_b_boxes ;
	#s_boxes = sorted(t_b_boxes, key=itemgetter(0)) ;
	#print s_boxes ;

	return t_b_boxes ;


def findOutlier(line , X) :

	kmeans = KMeans(n_clusters=2, random_state=0).fit(X) ;
	labels = kmeans.labels_ ;

	# get the label of max width -- usually multi-char
	maxI = np.argmax( np.array(X) ) ;
	index = labels[maxI] ;
	out = [] ; 
	pos = [] ;
	mean = 0 ;
	# segment groups
	for i in range(len(line)) :

		if labels[i] == index :
			out.append(i) ;
		else :
			mean += X[i] ;
			pos.append(X[i]) ;

	mean /= len(pos) ; # mean width of a character in a line
	multi = [] ;

	print 'Mean value ' , mean ;
	print 'Outlier ' , out ;

	# character segmentation 
	for i in range(len(out)) :
		# possible multi-character candidate , segment multi-character into n individual chars - based on mean difference
		diff = float(abs(X[out[i]] - mean)) / 10 ;
		if diff >= 0.7 : # greater by 70% 

			diff = math.ceil(diff) ;
			newW = X[out[i]] / diff ;
			pts = line[out[i]] ; # char to be segmented
			del line[out[i]] ; 
			new_pts = [] ;
			k = 1 ;
			st_pt = pts[0] ;
			while k <= diff : # diff - no. of cut points 
				end_pt = st_pt + newW ;
				line.insert( out[i] + (k - 1) , [st_pt , pts[1] , end_pt , pts[3]] ) ; # new point , insert it in the original list
				st_pt = end_pt ;
				k += 1 ;

	return line ;


# method to save word and line spacing positions 
def word_line_spacing(line_group , dest) :

	word_spaces = [] ;
	line_spaces = [] ;
	offset = 0 ;
	for line in line_group :

		mean_width = 0 ;
		# calculating mean width for a line
		for br in line :
			mean_width += ( br[2] - br[0] ) ; 
		mean_width = mean_width / len(line) ;
		print 'Mean width ' , mean_width ; 

		# marking spaces
		for i in range(1 , len(line)) :
			print  i , line[i][0] - line[i - 1][2] ;
			# if distance between two msers is >= mean_width , save mser index position
			if line[i][0] - line[i - 1][2] >= mean_width :
				word_spaces.append(offset + ( i - 1 ) )  ;

		# marking line space
		line_spaces.append( offset + (len(line) - 1)) ;
		
		offset += len(line) ;
	
	return ;

def main() :

	#yf = sys.argv[1] ;
	mypath = './Images/Digital_Orig/' ;
	mypath = './test/Input/' ;
	#mypath = './Final/Transformed/' ;
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] ;   
	#orig = './MSER/New/Line/Digital/' ;
	dest = './test/Output/temp/' ;
	#dest = './MSER/All_MSERs/' ;
	meanArea = 0 ;
	cont = True ;
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) ; 
	kernel = np.ones((5,5),np.float32)/25 ;
	found = True ;
	for file in onlyfiles :
		
		start_time = time.time() ;	

		if file != 'xss_1.png' :
			continue ;

		# when running complete pipeline
		
		# dest_path = file[ : file.index(".") ] ;
		# print dest_path ;
		# if not os.path.exists(dest + dest_path) :
		# 	os.makedirs(dest + dest_path) ;
		# else :
		# 	continue ;
		
		print(' Processing File : ', file);
		
		gray = cv2.imread(mypath + file , 0) ;
		rgb = cv2.imread(mypath + file) ;
		#orig_img = cv2.imread(orig + file) ;
		
		#retina = cv2.imread("./Final/Digital_Retina/" + file) ;
		# gray = cv2.filter2D(gray,-1,kernel) ;
		# rgb = cv2.filter2D(rgb,-1,kernel) ;
		#gray = cv2.bilateralFilter(gray,9,75,75)
		#rgb = cv2.bilateralFilter(rgb,9,75,75)
		# cv2.imshow('gray',gray) ;
		# cv2.imshow('rgb',rgb) ;
		# cv2.waitKey(0) ;

		print gray.shape ;

		# scale x(1/2)
		scale_1 = cv2.resize(gray, (gray.shape[1] / 2 , gray.shape[0] / 2 )) ;
		rgb_scale_1 = cv2.resize(rgb, (rgb.shape[1] / 2 , rgb.shape[0] / 2 )) ;

		# scale x4
		scale_2 = cv2.resize(gray, (gray.shape[1] * 4 , gray.shape[0] * 4 )) ;
		rgb_scale_2 = cv2.resize(rgb, (rgb.shape[1] * 4 , rgb.shape[0] * 4 )) ;

		# gray scale image
		g_scale_1 = cv2.resize(gray, (gray.shape[1] / 2  , gray.shape[0] / 2)) ;
		g_scale_2 = cv2.resize(gray, (gray.shape[1] * 4  , gray.shape[0] * 4)) ;

		#cv2.imwrite('./MSER/New/WORD/DIGI_2/Gray/gray_' + file , gray );
	
		# Applying OTSU Binarization
		ret1,thresh1 = cv2.threshold(scale_1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;
		ret2,thresh2 = cv2.threshold(scale_2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;
		ret3,thresh3 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;

		print 'rgb' , rgb_scale_1.shape ;

		wb1 = [] ; wb2 = [] ; wb3 = [] ; wb4 = [] ;
		# multiple scale MSER only for low-res images
		if ( rgb.shape[0] < 1000 or rgb.shape[1] < 1000 ) :
			
			# OTSU Binarized Image / Scale 1
			#wb1 = MSER(thresh1 , rgb_scale_1 , file , 'Thresh' , 2) ;

			# OTSU Binarized Image / Scale 2
			wb2 = MSER(thresh2 , rgb_scale_2 , file , 'Thresh' , 0.25 ) ;

			# GrayScale Image
			#wb3 = MSER(g_scale_1 , rgb_scale_1 , file , 'Gray' , 2 ) ;
			
			# GrayScale Image
			wb4 = MSER(g_scale_2 , rgb_scale_2 , file , 'Gray' , 0.25 ) ;

		print '1 : ' , time.time() - start_time ;
		
		# OTSU Binarized Image / Orig Scale
		wb5 = MSER(thresh3 , rgb , file , 'Thresh' , 1) ;
		print '2 : ' , time.time() - start_time ;
		
		# GrayScale Image
		wb6 = MSER(gray , rgb , file , 'Gray' , 1) ;
		print '3 : ' , time.time() - start_time ;

		words = wb2 + wb4 + wb5 + wb6 ;
		words_c = words ;
		
		# combining msers from multiple channels introduces repetitive regions -- applying non-max suppression again 
		all_rect , all_boxes , all_rectToarea = findBRect_C(gray,words) ;
		words = all_rect ;
		
		vis = rgb.copy() ;
		bw = gray.copy() ;
		#i = 1 ;

		segmented_crops = [] ;
		
		bw.fill(255) ;

		#i = 1 ;
		for br in words :
			cv2.rectangle(bw,(br[0],br[1]),(br[2],br[3]),(0,0,0),-1) ;
			#cv2.rectangle(vis ,(br[0],br[1]),(br[2],br[3]),(0,0,255),2) ; 
			'''			
			crop = rgb[br[1]:br[3],br[0]:br[2]] ;
			crop = cv2.cvtColor(crop ,cv2.COLOR_BGR2GRAY) ;
			ret,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;
			'''
			#crop = cv2.resize(crop , (28,28)) ;
			
			#cv2.imwrite('./test/Output/all/' + str(i) + '.png' , crop) ;
			#i += 1 ;
			#cv2.imwrite('./MSER/New/Crops/3_thresh/' + name , crop ) ;
			# if i == 3 :
			# 	break ;
		cv2.imwrite('./test/bw.png',bw) ;
		ret , bw = cv2.threshold(bw,127,255,cv2.THRESH_BINARY_INV) ;
		bw = bw / 255 ;
		th = bw.shape[1] ;
		rlsa_img = rlsa.RLSO(bw , th , 0) ;
		sorted_lines = rlsa.sortLines(rlsa_img) ;
		#rlsa_img = rlsa_img * 255 ;
		rlsa_img = 255 - rlsa_img ;

		cv2.imwrite('./test/rlsa.png',rlsa_img) ;
		#cv2.imshow('rlsa',rlsa_img) ;
		#cv2.waitKey(0) ;
		line_group = rlsa.sortMSERs(rlsa_img.copy() , sorted_lines , words) ;
		#line_group = [line_group[len(line_group) - 1]] ;
		filtered_line_group = [] ; # filtered line group
		suppr_line_group = [] ; # non-max-suppressed line group

		print 'RLSA ' , time.time() - start_time ;
		img_new = rgb.copy() ;
		lc = 0 ;
		
		'''		
		# remove duplicate rects 
		for k in range(len(line_group)) :
			rect = line_group[k] ;
			for i in range(len(rect)) :
				for j in range(len(rect)) :
					if i != j and len(rect[i]) > 0 and len(rect[j]) > 0 :				
						if rect[i][0] == rect[j][0] and rect[i][1] == rect[j][1] and rect[i][2] == rect[j][2] and rect[i][3] == rect[j][3] :				
							rect[j] = [] ;

			rect = filter(None , rect) ;
			line_group[k] = rect ;
		#line_group[0] = rect ;
		'''

		img_new = rgb.copy() ;

		# applying non-max suppression for individual lines rather than the whole image
		for k in range(len(line_group)) :
			rect , boxes , rectToarea = findBRect_C(gray,line_group[k]) ;
			boxes_idxs = non_max_suppression(boxes , 0.75, rectToarea) ;
			temp_word = [] ;
			#print 'Before ' ,len(line_group[k]) ;
			for i in range(len(boxes[0])) :
				if i in boxes_idxs :
					# print boxes[0] ;
					# print boxes[0][i]
					temp_word.append( [boxes[0][i] , boxes[1][i] , boxes[2][i] , boxes[3][i]] ) ;	
					cv2.rectangle(img_new , (boxes[0][i] , boxes[1][i]) , (boxes[2][i] ,boxes[3][i]) ,(0,255,0) ,2) ;
					
					crop = rgb[boxes[1][i]:boxes[3][i],boxes[0][i]:boxes[2][i]] ;
					#crop = cv2.cvtColor(crop ,cv2.COLOR_BGR2GRAY) ;
					#ret,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;		
					cv2.imwrite('./test/suppr/' + str(k) + '_' + str(boxes[0][i]) + '_' + str(boxes[1][i]) + '_'+ str(boxes[2][i]) +'_'+ str(boxes[3][i]) + '.png' , crop) ;
					
					
			#print 'After ' ,len(temp_word) ;

			lc += len(temp_word) ;
			suppr_line_group.append(temp_word) ;

		img_crop = rgb.copy() ;
		i = 0 ;
		# mser filter code --- added 18/7/2017
		for k in range(len(suppr_line_group)) :
			filtered_line = msers_filters.filters(thresh3.copy() ,suppr_line_group[k]) ;
			filtered_line_group.append(filtered_line) ;
			'''
			for bb in line_group[k] :
				cv2.rectangle(img_new,(bb[0],bb[1]),(bb[2],bb[3]) , (0,255,0) , 2) ;
				cv2.imwrite('./test/crops/' + str(i) + '.png',img_crop[bb[1]:bb[3],bb[0]:bb[2]]) ;
				i += 1 ;
				# cv2.imshow('img',img_new) ;
				# cv2.waitKey(0) ;
			'''
		cv2.imwrite('./test/not_non_max.png',img_new) ;
		#filtered_line_group = suppr_line_group ; 
		print 'Suppr ' , time.time() - start_time ;
		print 'all ' , len(words) ;
		print 'suppr' , lc ;

		cv2.imwrite('./test/non_max.png' , img_new) ;
		#cv2.imshow('all',vis) ;
		#cv2.imshow('non-max',img_new) ;
		#cv2.waitKey(0) ;
		#return ;
		#print 'Orig ' , len(line_group) ;
		#print 'Suppr ' , len(suppr_line_group) ;

		#line_group = [line_group[4]] ;
		#suppr_line_group = [] #[suppr_line_group[len(suppr_line_group) - 2]] ;
		sorted_line_group = [] ;
		# sort rlsa contours 

		for line in filtered_line_group :
			#print 'Before ' , line ;
			s_line_group = weighted_distance.groupMSERs( line , rgb.copy() ) ;
			sorted_line_group += s_line_group ;	
			print len(s_line_group) ;
			print len(sorted_line_group) ;
			#print line_indexes ;

			'''
			for line_i in s_line_group :

				temp_line = [] ;
				print 'New line' ;

				for i in line_i :
					br = line[i] ;
					crop = rgb[br[1]:br[3],br[0]:br[2]] ;
					crop = cv2.cvtColor(crop ,cv2.COLOR_BGR2GRAY) ;
					ret,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;
					crop = cv2.resize(crop , (28,28)) ;
					# cv2.imshow('mser',crop) ;
					# cv2.waitKey(0) ;
					temp_line.append(line[i]) ;

				# append the new line group 
				sorted_line_group.append(temp_line) ;	
			'''
		
		print 'sort MSERs ' , time.time() - start_time ;
		print len(sorted_line_group) ;
		#sorted_line_group = [line_group[12]] ;

		l_img = bw.copy() ;
		l_img.fill(255) ;

		# cv2.imshow('orig',orig_img) ;
		# cv2.waitKey(0) ;

		i = 1 ;
		j = 0 ;
		

		'''
		# find word spacing 
		for l in sorted_line_group :
			print 'Line ' ;
			width_bin = [] ;
			print 'Before seg' , len(l) ;
			for pts in l :
				print pts[2] - pts[0] ;
				width_bin.append([pts[2] - pts[0]]) ;
			print ;

			# find the outliers and segment multi-char
			l = findOutlier(l , np.array(width_bin)) ;
			print 'After seg' , len(l) ;
			print ;
		'''
		
		word_line_spacing(sorted_line_group , 'temp') ;

		for line in sorted_line_group :
			j += 1 ;
			line = ARFilter(line) ; # removes unwanted vertically grouped noise
			line = filter(None ,line) ;
			for br in line :
				#print br ;
				# cv2.rectangle(l_img,(br[0],br[1]),(br[2],br[3]),(0,0,0),-1) ;
				# cv2.imshow('crop',l_img) ;
				# cv2.waitKey(0) ;
				crop = rgb[br[1]:br[3],br[0]:br[2]] ;
				if 0 in crop.shape :
					continue ;
				crop = cv2.cvtColor(crop ,cv2.COLOR_BGR2GRAY) ;
				ret,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;
				#crop = cv2.resize(crop , (28,28)) ;
				cv2.imwrite(dest + "/" + str(j) + '_' + str(i) + '.png' , crop) ;
				#cv2.imwrite(dest + dest_path + '/' + str(i) + '.png' , crop) ;
				i += 1 ;
		print 'After writing ' , time.time() - start_time ;
		#break ;
main() ;
def gauss() :

	mypath = './MSER/New/Crops/Retina/1_gauss/' ;
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] ;
	i = 1 ;
	for file in onlyfiles :
		img = cv2.imread(mypath + file , 0) ;
		#print file ;
		img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,2) ;
		#img = cv2.GaussianBlur(img,(3,3),0) ;
		#kernel = np.ones((1,1),np.uint8) ;
		#img = cv2.dilate(img,kernel,iterations = 1) ;
		# low = cv2.pyrDown(img) ;
		# low = cv2.pyrDown(low) ;
		# high = cv2.pyrUp(img) ;
		crop = segmentWord(img) ;

		for j in range(len(crop)) :
			n = str(i) + '_' + str(j) + '_' + '.png' ;
			cv2.imwrite('./MSER/New/Crops/Retina/Op/' + n , crop[j] ) ;
		i += 1 ;
		#cv2.imwrite('./MSER/New/Crops/3_segment/1_gaussian/high_' + file , high ) ;
		
	'''
	mypath = './MSER/New/Crops/3_segment/1_gaussian/' ;
	i = 1 ;
	for file in onlyfiles :
		img = cv2.imread(mypath + file , 0) ;
		img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) ;
		crop = segmentWord(img) ;
		for j in range(len(crop)) :
			n = str(i) + '_' + str(j) + '_' + '.png' ;
			cv2.imwrite('./MSER/New/Crops/3_segment/1_gaussian_segment/' + n , crop[j] ) ;
		i += 1 ;
	'''
#gauss() ;


