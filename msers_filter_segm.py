## msers_filter.py -- uses vertical segmentation approach instead of sub-window sum method

import cv2 ;
import numpy as np ;
from collections import OrderedDict ;

# method to sort map based on 'value' list length
def sort_map(char_map) :

	sorted_map = OrderedDict( {} ) ;
	for i in range(len(char_map)) :
		min_len = 99999 ; min_key = 0 ; max_len = -1 ; max_key = 0 ;
		for k , v in char_map.items() :
			# finding min length key
			if len(v) > max_len and not (k in sorted_map) :
				max_len = len(v) ;
				max_key = k ;

		sorted_map[max_key] = char_map[max_key] ;

	return sorted_map ;

# method to segment a multi - char crop
def vertical_segmentation(img , char , img_flag) :

	segm_chars = [] ;
	vert_proj = np.sum(img[char[1] : char[3] , char[0] : char[2]] , axis=0) ;

	# if region already segmented
	if np.amin( img_flag[char[1] : char[3] , char[0] : char[2]] ) == 255 :		
		#print 'yes' ;
		return [] ;

	img_flag[char[1] : char[3] , char[0] : char[2]] = 255 ;
	# cv2.imshow('crop',img[char[1] : char[3] , char[0] : char[2]]) ;
	# cv2.waitKey(0) ;

	max_col_val = 255 * (char[3] - char[1]) ;
	st = char[0] ;

	for i in range(vert_proj.shape[0]) :

		if vert_proj[i] == max_col_val and ( st + i - 1 ) - st > 0 :

			if st + i - 1 <= char[2] :
				segm_chars.append([st , char[1] , st + i - 1 , char[3]]) ;
				st = st + i + 1 ;

				# if start reaches maxX
				if st >= char[2] :				
					break ;
			else :
				break ;

	# segment the remaining portion if the last start point < maxX 
	if st < char[2] :
		segm_chars.append([st , char[1] , char[2] , char[3]]) ;

	return segm_chars ;


def filters(img ,rect) :

	white = img.copy() ;
	white.fill(255) ;
	
	# remove duplicate rects ---- rects = msers 
	for i in range(len(rect)) :
		for j in range(len(rect)) :
			if i != j and len(rect[i]) > 0 and len(rect[j]) > 0 :				
				if rect[i][0] == rect[j][0] and rect[i][1] == rect[j][1] and rect[i][2] == rect[j][2] and rect[i][3] == rect[j][3] :				
					rect[j] = [] ;

	rect = filter(None , rect) ;
	rectToarea = {} ;

	# area for rects 
	for i in range(len(rect)) :
		rectToarea[i] = (rect[i][2] - rect[i][0]) * (rect[i][3] - rect[i][1]) ;

	# sort rects by area
	ordRects = sorted(rectToarea.items(), key=lambda x: x[1] ,reverse = True) ;

	# char container map
	char_cont = {} ;

	# adding chars falling inside another char
	for i in range(len(ordRects)) :
		rect_i = ordRects[i][0] ;
		box = rect[rect_i] ;

		for j in range(len(rect)) :
			if j != rect_i :
				# checking if char 'rect_i' encloses char j 
				if box[0] <= rect[j][0] and box[1] <= rect[j][1] and box[2] >= rect[j][2] and box[3] >= rect[j][3] :	
					if not (rect_i in char_cont) :
						char_cont[rect_i] = [] ;
					char_cont[rect_i].append(j) ;


	# print 'Map before' ;
	# print char_cont ;

	char_cont = sort_map(char_cont) ;

	# print 'Map after' ;
	# print char_cont ;

	rect.sort(key=lambda  x:int(x[0])) ;
	img_flag = img.copy() ;
	img_flag.fill(0) ;
	img_flag_1 = img.copy() ;
	img_flag_1.fill(0) ;

	# iterate map and remove undesired characters -- char which enclose multiple chars and has an high width deviation from its neighbour chars
	for k , v in char_cont.items() :
		
		if len(rect[k]) == 0 :
			continue ;
		
		w = rect[k][2] - rect[k][0] ;
		h = rect[k][3] - rect[k][1]	;
		sim_char = 0 ;

		# similar to container char , remove it
		for i in range(len(v)) :		
			if len(rect[v[i]]) > 0 and abs( w - (rect[v[i]][2] - rect[v[i]][0]) ) <= 10 and abs( h - (rect[v[i]][3] - rect[v[i]][1]) ) <= 10 :
				#print 'Similar char' ;
				rect[v[i]] = [] ;				

		# if the container char encloses only one char, remove the enclosed char --- assuming it is a part of a char
		if len(v) == 1 :
			#print 'One char' ;
			rect[v[0]] = [] ;
			continue ;

		is_same_height = True ;
		widths_sum = 0 ;

		#print 'Cont ' , rect[k] ;

		# sum up the widths of enclosed chars
		for i in range(len(v)) :
			#print rect[v[i]] ;
			# if height of one of the enclosed chars is not close to container char , remove it
			if len(rect[v[i]]) > 0 and abs( h - (rect[v[i]][3] - rect[v[i]][1]) ) >= 10 :
				is_same_height = False ;
				# cv2.imshow('temp',img[rect[k][1] : rect[k][3] , rect[k][0] : rect[k][2]] ) ;
				# cv2.waitKey(0) ;
				break ;
			elif len(rect[v[i]]) > 0 :
				widths_sum += (rect[v[i]][2] - rect[v[i]][0]) ;

		if is_same_height :

			# cv2.imshow('temp',img[rect[k][1] : rect[k][3] , rect[k][0] : rect[k][2]] ) ;
			# cv2.waitKey(0) ;
			# contains multiple individual chars
			# print 'Widths' ;
			# print widths_sum ;
			# print w ;
			# if abs(widths_sum - w) > 20 :
			# 	is_same_height = False ;
			# # run vertical segmentation algorithm and extract the individual chars			
			# else :
			segmented_chars = vertical_segmentation(img , rect[k] , img_flag) ;
			for char in segmented_chars :
				#print char ;
				temp_char = img[ char[1] : char[3] , char[0] : char[2]] ;
				#print temp_char.shape ;
				# search for a key close to this char and check if it contains multiple chars within
						
			# if len(segmented_chars) == 0 :
			# 	cv2.imshow('temp',img[rect[k][1] : rect[k][3] , rect[k][0] : rect[k][2]] ) ;
			# 	cv2.waitKey(0) ;				

			# for i in range(len(v)) :
			# 	if len(rect[v[i]]) == 0 :
			# 		continue ;
			# 	temp_char = img[ rect[v[i]][1] : rect[v[i]][3] , rect[v[i]][0] : rect[v[i]][2]] ;
			# 	print temp_char.shape ;
			# 	cv2.imshow('temp',temp_char) ;
			# 	cv2.waitKey(0) ;
			

			# delete container and enclosed characters
			rect += segmented_chars ;
			rect[k] = [] ;
			is_same_height = False ;
			# if sum of widths is close to container width 
			# if abs(widths_sum - w) <= 15 :
			# 	'''
			# 	# if distance b/w enclosed chars is very small / zero , remove enclosed chars - assuming they are parts of a character
			# 	for i in range(len(v) - 1) :
			# 		if len(rect[v[i]]) > 0 and len(rect[v[i + 1]]) > 0 and abs(rect[v[i + 1]][0] - rect[v[i]][2]) <= 2 :
			# 			is_same_height = False ;
			# 			break ;
			# 	'''
			# 	if is_same_height :
			# 		print 'Container char' ;
			# 		rect[k] = [] ;
				
				
			# else:
			# 	is_same_height = False ;

			

		if not is_same_height :
			#print 'Enclosed chars' ;
			for i in range(len(v)) :
				rect[v[i]] = [] ;


	rect = filter(None , rect) ;

	#print 'Final ' , rect ;
	return rect ;