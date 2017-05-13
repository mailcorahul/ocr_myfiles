import numpy as np ;
from operator import itemgetter ;
import cv2 ;
import math ;

# method to find slope 
def find_slope(line) :

	# infinite slope
	if line[2] - line[0] == 0 :
		return 99999 ;

	slope = float( abs(line[3] - line[1]) )  / abs( line[2] - line[0] ) ;

	return slope ;

# method to compute euclidean distance between two points
def euclidean_distance(pt1 , pt2) :

	dist = math.sqrt( (( pt1[0] - pt2[0] ) ** 2 ) + (( pt1[1] - pt2[1]) ** 2) ) ;

	return dist ;

# method to compute angle between two lines 
def angle_bw_lines(slope1 , slope2) :

	angle = np.arctan( abs( float((slope1 - slope2)) / ( 1 + (slope1 * slope2)) )) ;

	return angle ;

# function which gives a weight for an angle
def g_func(angle) :


	# quadruple the angle
	if angle > 30 :
		C = 4 ;
	# square the angle
	else :
		C = 2 ;

	delta = ( (angle + 1) ** C ) ; # (angle + 1) to prevent the weight from zero-ing out the e_dist

	return delta ;

# method to find weighted distance between two rectangles
def findWeightedDistance(rect1 , rect2) :

	# computing euclidean distance 
	midpt1 = [ rect1[2] , (rect1[1] + rect1[3]) / 2 ] ;
	midpt2 = [ rect2[0] , (rect2[1] + rect2[3]) / 2 ] ;
	print 'MSERs ' , rect1 , rect2 ;

	e_dist = euclidean_distance(midpt1 , midpt2) ;
	print 'Euclidean distance ' , e_dist ;

	# compute the angle made by right mser with left mser
	#line1 = [ [rect1[2] , rect1[1]] , [rect1[2] , rect1[3]] ] ;
	line2 =  midpt2 + midpt1  ;

	slope2 = find_slope(line2) ;

	print 'Slope ' , slope2 ;
	
	C = 10 ;
	# inclination angle of line 2 , i.e line joining vertical midpoints of both msers
	inc_angle = int( abs(np.arctan(slope2)) * ( 180 / 3.14 ) ) ;

	if inc_angle <= 30 :
		inc_angle = 0 ;

	print 'Inclination Angle ' , inc_angle ;

	# weighted distance
	w_dist = e_dist * ( inc_angle + 1 ) #g_func(l_angle) ;
	print '*****Weighted distance ' ,w_dist ,'*******' ;
	print ;
	return w_dist ;

s_line_group = [] ;
'''
# recursive version
def trackMSERs(line, src , list_i , visited , img) :

	global s_line_group ;

	for i in range(len(line)) :

		if not visited[i] :

			#print 'for ' , i ;

			# except for first character
			if src != -1 :
				minY = max(line[src][1] , line[i][1]) ;
				maxY = min(line[src][3] , line[i][3]) ;
				w1 , w2 = line[src][2] - line[src][0] , line[i][2] - line[i][0] ;
				h1 , h2 = line[src][3] - line[src][1] , line[i][3] - line[i][1] ;
				# print 'width ' , w1 , w2 ;
				# print 'height ' , h1 , h2 ;
				# check if their intersection height is greater than 1/2 of either of its individual height
				if maxY > minY and ( (( maxY - minY ) >= float(line[src][3] - line[src][1]) * 0.75 ) or (( maxY - minY ) >= float(line[i][3] - line[i][1]) * 0.75 )) : #and (abs(w1 - w2) <= 10 and abs(h1- h2) <= 10) 
				
					# print src , i ;
					# print maxY - minY ;
					# print ;
					s_line_group[list_i].append(i) ;
					br = line[src] ;
					crop = img[br[1]:br[3],br[0]:br[2]] ;
					crop = cv2.cvtColor(crop ,cv2.COLOR_BGR2GRAY) ;
					ret,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;
					crop1 = cv2.resize(crop , (28,28)) ;

					br = line[i] ;
					crop = img[br[1]:br[3],br[0]:br[2]] ;
					crop = cv2.cvtColor(crop ,cv2.COLOR_BGR2GRAY) ;
					ret,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;
					crop2 = cv2.resize(crop , (28,28)) ;

					# cv2.imshow('left' , crop1) ;
					# cv2.imshow('right', crop2) ;
					# cv2.waitKey(0) ;



					# mark it as visited -- add it to line
					visited[i] = True ;

			# start of a new line group
			else :
				list_i += 1 ;
				visited[i] = True ;
				s_line_group.append([]) ;
				s_line_group[list_i].append(i) ;
				#print 'New Line ' , i ;

			if visited[i] :
				trackMSERs(line , i , list_i , visited , img) ;

'''

# iterative version
def trackMSERs(line , visited , img) :

	global s_line_group ;
	intr_list = [] ;
	avg_height = 0 ;

	# creating an intersection list for each character
	for i in range(len(line)) :
		avg_height += line[i][3] - line[i][1] ;
		intr_list.append([]) ;
		for j in range(len(line)) :
			if i != j :
				minY = max(line[i][1] , line[j][1]) ;
				maxY = min(line[i][3] , line[j][3]) ;

				# check if their intersection height is greater than 0.75 of either of its individual height
				if maxY > minY and ( (( maxY - minY ) >= float(line[i][3] - line[i][1]) * 0.5 ) or (( maxY - minY ) >= float(line[j][3] - line[j][1]) * 0.5 )) :
					intr_list[i].append(j) ;

	# average width of a char in 'line' 				
	avg_height /= len(line) ;

	line_c = 0 ;
	char_to_line = {} ;

	# grouping characters in lines
	for i in range(len(line)) :
		for j in range(len(line)) :
			if i != j and (not(i in char_to_line) or not(j in char_to_line)) :
				# set difference between two lists to get characters not intersecting with char i and j 
				no_intr_list_1 = list( set(intr_list[i]) - set(intr_list[j]) ) ;
				no_intr_list_2 = list( set(intr_list[j]) - set(intr_list[i]) ) ;
				intr = list( set(intr_list[i]) & set(intr_list[j]) ) ;
				
				# if i == 0 :
				# 	print 'Non 1' , len(no_intr_list_1) ;
				# 	print 'Non 2' , len(no_intr_list_2) ;
				# 	print 'i ' , len(intr_list[i]) ;
				# 	print 'j ' , len(intr_list[j]) ;

				# i and j fall in the same line 
				if len(no_intr_list_1) < len(intr) and len(no_intr_list_2) < len(intr) :

					# if the characters are too small in height , skip them -- avoids the case where low height chars form a seperate line
					if (line[i][3] - line[i][1] < avg_height - 10) and (line[j][3] - line[j][1] < avg_height - 10) :
						ignore = True ;
						'''
						for k in range(len(intr)) :
							if line[intr[k]][3] - line[intr[k]][1] >= avg_height :
								ignore = False ;
								print 'Inside ignore' , i , j , k ;
								break ;
						'''
						if ignore :
							#print 'Inside ignore true'
							continue ;

					if i in char_to_line :
						char_to_line[j] = char_to_line[i] ;
					if j in char_to_line :
						char_to_line[i] = char_to_line[j] ;
					if not(i in char_to_line) and not(j in char_to_line) :
						'''
						intr = list(set(intr_list[i]) & set(intr_list[j])) ;
						for k in range(len(intr)) :
							if k in char_to_line :
								char_to_line[i] = char_to_line[k] ;
								char_to_line[j] = char_to_line[k] ;
								break ;
						'''
						if not( i in char_to_line ) :
							char_to_line[i] = line_c ;
							char_to_line[j] = line_c ;
							line_c += 1 ;

	
	# assigning a line for missed-out characters
	for i in range(len(line)) :
		if not(i in char_to_line) :
			for j in range(len(line)) :
				if i != j and j in char_to_line :
					minY = max(line[i][1] , line[j][1]) ;
					maxY = min(line[i][3] , line[j][3]) ;

					# check if they intersect each other -- even by 1 percent
					if maxY - minY > 0 :
						char_to_line[i] = char_to_line[j] ;
						break ;

			# no line has been assigned yet , put it in a new line
			if not( i in char_to_line ) :
				char_to_line[i] = line_c ;
				line_c += 1 ;

	#print 'Map length ' , len(char_to_line) , len(line) ;

	# forming lists from char_to_line map 
	for i in range(max(char_to_line.values()) + 1) :
		s_line_group.append([]) ;

	print char_to_line ;
	for i in range(len(line)) :
		s_line_group[char_to_line[i]].append(i) ;
'''

# iterative version
def trackMSERs(line , visited , img) :

	global s_line_group ;

	for i in range(len(line)) :

		if not visited[i] :

			visited[i] = True ;
			s_line_group.append([]) ;
			s_line_group[ len(s_line_group) - 1 ].append(i) ;
			cur_ind = len(s_line_group) - 1 ;

			for j in range(i + 1 ,len(line)) :
				
				skewed = False ;

				if visited[j] :
					continue ;
				
				intr = 0 ;
				non_intr = 0 ;
				for k in range(len(s_line_group[cur_ind])) :

					line_cur = line[s_line_group[cur_ind][k]] ;
					minY = max(line_cur[1] , line[j][1]) ;
					maxY = min(line_cur[3] , line[j][3]) ;

					# check if their intersection height is greater than 0.4 of either of its individual height
					if maxY > minY and ( (( maxY - minY ) >= float(line_cur[3] - line_cur[1]) * 0.5 ) or (( maxY - minY ) >= float(line[j][3] - line[j][1]) * 0.5 )) : #and (abs(w1 - w2) <= 10 and abs(h1- h2) <= 10) 
						intr += 1 ;

					# if mser j has zero intersection with any one of the mser -- do not add it to current line
					elif maxY <= minY :
						non_intr += 1 ;
						#skewed = True ;
						#break ;

				# mser j is not a candidate char for current line
				if non_intr >= intr :
					continue ;

				# if half the msers in the list intersect with the current mser , add it 
				if not skewed and intr >= float(len(s_line_group[cur_ind])) * 0.5 :
					s_line_group[ cur_ind ].append(j) ;
					visited[j] = True ;

			## newly added
			
			for j in range(len(line)) :
				if not visited[j] :
					for k in range(len(s_line_group)) :
						intr = 0 ;
						for u in range(len(s_line_group[k])) :
							line_cur = line[s_line_group[k][u]] ;							
							minY = max(line_cur[1] , line[j][1]) ;
							maxY = min(line_cur[3] , line[j][3]) ;

							# check if their intersection height is greater than 0.4 of either of its individual height
							if maxY > minY and ( (( maxY - minY ) >= float(line_cur[3] - line_cur[1]) * 0.5 ) or (( maxY - minY ) >= float(line[j][3] - line[j][1]) * 0.5 )) : 
								intr += 1 ;
						if intr >= len(s_line_group[k]) / 2 :
							s_line_group[k].append(j) ;
							break ;

'''		

# use bubble sort -- n is too small
def sort_lines(line_group , line_bounds) :

	#print line_bounds , line_group ;
	for i in range(len(line_group)) :
		for j in range(i + 1 , len(line_group)) :
			# based on minY
			if line_bounds[i][1] > line_bounds[j][1] :
				temp = line_group[i] ; 
				line_group[i] = line_group[j] ; 
				line_group[j] = temp ;
				temp = line_bounds[i] ;
				line_bounds[i] = line_bounds[j] ;
				line_bounds[j] = temp ;

	return line_group , line_bounds ;								

# group msers which intersect vertically , ( applicable for colon )
def groupVMSERs(s_line_group) :

	for l in range(len(s_line_group)) :
		
		tline = s_line_group[l] ;
		to_del = [] ; 
		to_add = [] ;

		for i in range(len(tline)) :
			for j in range( i + 1 , len(tline)) :
				minY = max(tline[i][1] , tline[j][1]) ;
				maxY = min(tline[i][3] , tline[j][3]) ;

				minX = max(tline[i][0] , tline[j][0]) ;
				maxX = min(tline[i][2] , tline[j][2]) ;
				
				# if two msers intersect vertically				
				if maxX > minX and ( (( maxX - minX ) >= float(tline[i][2] - tline[i][0]) * 0.5 ) or (( maxX - minX ) >= float(tline[j][2] - tline[j][0]) * 0.5 )) : 

					# check if their intersection height is not greater than 1/2 of either of its individual height
					if not( maxY > minY and ( (( maxY - minY ) >= float(tline[i][2] - tline[i][0]) * 0.3 ) or (( maxY - minY ) >= float(tline[j][2] - tline[j][0]) * 0.3 ))) : 
						merged_mser = [ min(tline[i][0] , tline[j][0]) , min(tline[i][1],tline[j][1]) , max(tline[i][2] , tline[j][2]) , max(tline[i][3] ,tline[j][3]) ] ;						
						to_del.append(i) ;
						to_del.append(j) ;
						to_add.append(merged_mser) ;

		# remove vertically intersecting msers	
		for k in range(len(tline)) :
			if k in to_del :
				s_line_group[l][k] = [] ;

		# add the new merged mser back to line_group
		for k in range(len(to_add)) :
			#print 'Merged ' , to_add[k] ;
			s_line_group[l].append(to_add[k]) ;

		s_line_group[l] = filter(None , s_line_group[l]) ;
		#print 'filtered ' , s_line_group[l] ;

	return s_line_group ;


def groupMSERs( line , img ) :

	# s_line_group = [] ;
	# s_line_group_intr = [] ;
	#print 'Input length ' , len(line) ;

	global s_line_group ;
	
	#trackMSERs( line , -1 , -1 , visited , img) ;

	line_list = line ;
	t_line_group = [] ; 
	s_line_group = [] ; 
	visited = [ False for i in range(len(line_list)) ] ;
	#trackMSERs(line_list , -1 , -1 , visited , img) ;
	trackMSERs( line , visited , img ) ;
	
	print 'In groupMSERs' ;
	print 'Input length ' , len(line) ;
	#print 'Length , ' , len(s_line_group) ;

	# for line_i in s_line_group :
	# 	print line_i ;

	line_bounds = [] ;

	# find minX , minY , maxX , maxY pts for each line
	for line_i in s_line_group :
		minX = 99999 ; minY = 99999 ; maxX = -1 ; maxY = -1 ;
		
		for i in line_i :
			if line[i][0] < minX :
				minX = line[i][0] ;
			if line[i][1] < minY :
				minY = line[i][1] ;
			if line[i][2] > maxX :
				maxX = line[i][2] ;
			if line[i][3] > maxY :
				maxY = line[i][3] ;

		line_bounds.append([]) ;
		line_bounds[ len(line_bounds) - 1 ] = [minX , minY , maxX , maxY]  ;

	'''
	# map isolated characters to closest line 
	for line_i in s_line_group :
		minDist = 99999 ;
		closeI = 1 ;
		# contains only one char
		if len(line_i) == 1 :
			for i in range(len(line_bounds)) :
				# closest line group
				if abs( line[line_i[0]][1] - line_bounds[i][3] ) < minDist :
					minDist = abs( line[line_i[0]][1] - line_bounds[i][3] ) ;
					closeI = i ;
					print 'Closer line ' ,closeI ;

			s_line_group[closeI].append(line_i[0]) ;
			line_i = [] ; # empty the one-char list
			line_bounds[closeI] = [] ;
	'''

	# remove empty lines 
	s_line_group = filter(None , s_line_group) ;
	line_bounds = filter(None , line_bounds) ;

	# sort lines based on minY
	s_line_group , line_bounds = sort_lines( s_line_group , line_bounds ) ;

	# final sorted line list
	s_line = [] ;

	temp_line_group = [] ;
	#print '1 : ' , s_line_group ;
	for line_i in s_line_group :
		tline = [] ;
		for i in line_i :
			tline.append(line[i]) ;
		temp_line_group.append(tline) ;

	s_line_group = temp_line_group ;
	
	# compute vertical intersection for msers
	s_line_group = groupVMSERs(s_line_group) ;
	
	# sort each line based on minX 
	for tline in s_line_group :
		# tline = [] ;
		# for i in line_i :
		# 	tline.append(line[i]) ;
		#print 'Before ' , tline ;
		tline = sorted(tline, key=itemgetter(0)) ;
		#print 'After ' , tline ;
		s_line.append([]) ;
		s_line[ len(s_line) - 1 ] = tline ;


	#print '2 : ' , s_line ;

	return s_line ;

	# # find weighted distance for every mser i and mser j
	#for i in range(len(line)) :

		# cur_dist = 0 ;
		# min_dist = 999999 ;
		# list_i = -1 ; min_j = -1 ;
		# found = False ;
		#trackMSERs( line , i , ) 


	# 	for k in range(len(s_line_group_intr)) :

	# 		minY = max(s_line_group_intr[k][0] , line[i][1]) ;
	# 		maxY = min(s_line_group_intr[k][1] , line[i][3]) ;	

	# 		if maxY > minY : #and ( (( maxY - minY ) >= (line[prev][3] - line[prev][1])/2 ) or (( maxY - minY ) >= (line[j][3] - line[j][1])/2 )) :
	# 			s_line_group[k].append(i) ;

	# 			if minY < s_line_group_intr[k][0] and maxY > s_line_group_intr[k][1] :
	# 				s_line_group_intr[k][0] = minY ;
	# 				s_line_group_intr[k][1] = maxY ;

	# 			found = True ;
	# 			break ;
			
	# 	if not found :
	# 		s_line_group.append([]) ;
	# 		s_line_group_intr.append([]) ;
	# 		list_i = len(s_line_group) - 1 ;
	# 		s_line_group[list_i].append(i) ;
	# 		s_line_group_intr[list_i].append(line[i][1]) ;
	# 		s_line_group_intr[list_i].append(line[i][3]) ; #minY , maxY co-ordinates


		
	'''
	prev = i ;
	for j in range(i + 1 ,len(line)) :

		if i == j : 
			continue ;

		minY = max(line[prev][1] , line[j][1]) ;
		maxY = min(line[prev][3] , line[j][3]) ;

		# check if their intersection height is greater than 1/2 of either of its individual height
		if maxY > minY and ( (( maxY - minY ) >= (line[prev][3] - line[prev][1])/2 ) or (( maxY - minY ) >= (line[j][3] - line[j][1])/2 )) :
			s_line_group[list_i].append(j) ;
			prev = j ;
	'''	

	'''
		# if mser j is not completely to the right of mser i
		if line[j][0] < line[i][2] :
			continue ;

		cur_dist = findWeightedDistance(line[i] , line[j]) ;
		if i == 10 :
			print i , j , cur_dist ;
			print ;
		if cur_dist < min_dist :
			min_dist = cur_dist ;
			min_j = j ;
	'''

	'''
	# add mser i and its successor to line group 'list_i' 
	if min_j != - 1 :
		
		br = line[i] ;
		crop = img[br[1]:br[3],br[0]:br[2]] ;
		crop = cv2.cvtColor(crop ,cv2.COLOR_BGR2GRAY) ;
		ret,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;
		crop1 = cv2.resize(crop , (28,28)) ;

		br = line[min_j] ;
		crop = img[br[1]:br[3],br[0]:br[2]] ;
		crop = cv2.cvtColor(crop ,cv2.COLOR_BGR2GRAY) ;
		ret,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ;
		crop2 = cv2.resize(crop , (28,28)) ;

		# cv2.imshow('left' , crop1) ;
		# cv2.imshow('right', crop2) ;
		# cv2.waitKey(0) ;
		
		s_line_group[list_i].append(min_j) ;
	'''

	#return s_line_group ;

def main() :



	return ; 