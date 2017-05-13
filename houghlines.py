import cv2
import sys 
import math
import itertools
import imutils
import ast
import numpy as np
from os import listdir
from os.path import isfile, join

print "******Hough Line Transform*****" ;

def houghT(orig) :
	line_priori = [] ;
	img = orig.copy() ;
 	params = [] ;
	h , w = img.shape[:2] ;
	pts = [[[0,0]] , [[0,h]] , [[w,0]] ,[[w,h]]];
	# textAngle1 = (angleM * 3.14) / 180 ; textAngle2 = textAngle1 + 1.57  ;
	angleDiff =  0.0723599 ; # +- 1.5 degrees threshold
	line_seg = [] ;
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# choose no. of votes based on w & h of the image 
	votes = min(w,h) / 5 ; # use 1/4 th of the image
	# print "Text angle 1 : " , textAngle1 , " Text angle 2 : " , textAngle2 ;
	print 'Minimum Line Length ' , votes ;
	lines = cv2.HoughLines(gray,1,np.pi/180,votes) ;
	
	if lines is None :
		print 'No lines' ;
		return pts ;

	i = 0 ;
	#cv2.drawContours(img,[pts],-1, (0,255,0), 3)
	for line in lines :
		i = i + 1 ;
		for rho,theta in line:
			add = True ;

			# a = np.cos(theta)
			# b = np.sin(theta)
			# x0 = a*rho
			# y0 = b*rho
			# x1 = int(x0 + 5000*(-b))
			# y1 = int(y0 + 5000*(a))
			# x2 = int(x0 - 5000*(-b))
			# y2 = int(y0 - 5000*(a))
			# #print x1 , y1 , x2 , y2 ;
			# cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
			# cv2.imshow("Hough Lines",img)
			# cv2.waitKey(0);

			# print rho , ' ' , theta ;
			# ignoring similar lines
			for r , t in params : 
				if (abs(rho - r) <= 30 and abs(theta - t) <= 0.5) :
					add = False ;
					# print 'Similar Line' ;
					break ;

				if ( abs(rho - abs(r)) <= 30 or abs(abs(rho) - r) <= 30 ) and abs(theta - t) >= 3 :
					add = False ;
					# print 'Similar Line' ;
					break ;
				
			if add :
				# including only lines parallel/perpendicular to text orientation
				# if ( abs(theta - textAngle1) <= angleDiff or  abs(theta - textAngle1) >= 3.1  ) or ( abs(theta - textAngle2) <= angleDiff or  abs(theta - textAngle2) >= 3.1 ) :	
					# print 'Added' ;
				params.append( [rho,theta] ) ;
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 5000*(-b))
				y1 = int(y0 + 5000*(a))
				x2 = int(x0 - 5000*(-b))
				y2 = int(y0 - 5000*(a))
				#print x1 , y1 , x2 , y2 ;
				cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
				# cv2.imshow("Hough Lines",img)
				# cv2.waitKey(0);
				line_seg.append( [x1,y1,x2,y2] ) ;
				line_priori.append(1) ;
				# else :
					# print 'Not parallel or perpendicular' ;

	print 'Line seg ' , len(line_seg) ;
	cv2.imwrite('./temp/hough.png',img) ;
	cv2.imshow("Hough Lines",img)
	cv2.waitKey(0);
	return findBestQuad(line_seg, line_priori , img) ;

# coefficients of x : A , y : B and constant : C 
def lines(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

# using Cramer's Rule to solve linear equations
def intersection(L1, L2, w, h):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        if( x > w or x < 0 or y > h or y < 0 ):
        	return False ;
        return [x,y] ;
    else:
        return False

def findIntersect(line, w, h) :

	pts = [] ;
	for i in range(len(line) - 1) :
		for j in range(i + 1 ,len(line)) :
			inner = [] ;
			x1 , y1 , x2 , y2 = line[i] ; 
			x3 , y3 , x4 , y4 = line[j] ; 
			l1 = lines( [x1,y1], [x2,y2] ) ;
			l2 = lines( [x3,y3], [x4,y4] ) ;
			pt = intersection( l1,l2 ,w ,h) ;
			if pt != False :
				#inner.append(pt) ;
				pts.append(pt) ;

	#pts = np.array(pts) ;		
	#print 'Intersection for a quad : ' , pts , ' Length : ' ,len(pts) , type(pts);
	return pts ;

def lineDist(p1,p2) :

	x = (p2[0] - p1[0]) ** 2 ;
	y = (p2[1] - p1[1]) ** 2 ;
	return math.sqrt(x + y) ;

def findBestQuad(line,line_priori ,img) :

	h , w = img.shape[:2] ;
	quads = [] ;
	isThree = False ;
	extremes = [] ;
	maxArea = -10000 ; max2Area = -10000 ;
	thresholdArea = ( w * h ) / 4 ;
	maxQuad = [] ; max2Quad = [] ;
	bounds = [] ;
	priori = [] ;
	bounds.append([0,0,0,h]) ; # left
	bounds.append([0,0,w,0]) ; # top
	bounds.append([w,0,w,h]) ; # right
	bounds.append([0,h,w,h]) ; # bottom
	horz = False ; vert = False ;
	fullImage = np.array( [[0,0] , [0,h] , [w,h] ,[w,0]] ) ;

	# find quadrilateral only if hough line output contains 4 or more lines
	# if len(line) < 4 :
	# 	return fullImage ;
	
	# include boundary lines and set vote as 0 
	line.append(bounds[0]) ;
	line.append(bounds[1]) ;
	line.append(bounds[2]) ;
	line.append(bounds[3]) ;
	

	line_priori.append(0) ;
	line_priori.append(0) ;
	line_priori.append(0) ;
	line_priori.append(0) ;
		
	line_set = range( 0, len(line) ) ;
	print 'Line len ' , len(line) ;
	for cc in itertools.combinations(line_set , 4):
		list(cc) ;
		i = cc[0] ; j = cc[1] ; k = cc[2] ; l = cc[3] ;
		copy = img.copy() ;
		copy1 = img.copy() ;
		quad = findIntersect([line[i],line[j],line[k],line[l]],w,h) ;
		priori_sum = line_priori[i] + line_priori[j] + line_priori[k] + line_priori[l] ;
		#hull = cv2.convexHull(np.array(quad)) ;
		#print 'Hull points : ' , hull ;
		# area = cv2.contourArea(hull) ;
		# cv2.drawContours(copy,[hull],-1, (255,0,0), 3)
		# cv2.imshow("Hull",copy) ;
		# cv2.waitKey(0) ;
		
		if len(quad) == 4 :
			hull = cv2.convexHull(np.array(quad)) ;
			#print 'Hull points : ' , hull ;
			area = cv2.contourArea(hull) ;
			#cv2.drawContours(copy,[hull],-1, (255,0,0), 3)
			# cv2.imshow("Hull",copy) ;
			# cv2.waitKey(0) ;
			if area >= thresholdArea and len(hull) == 4 :
				# print 'Hull ' , hull ;
				# print 'Sum , ' , priori_sum ;
				# cv2.line(copy1,(line[i][0],line[i][1]),(line[i][2],line[i][3]),(0,0,255),2)
				# cv2.line(copy1,(line[j][0],line[j][1]),(line[j][2],line[j][3]),(0,0,255),2)
				# cv2.line(copy1,(line[k][0],line[k][1]),(line[k][2],line[k][3]),(0,0,255),2)
				# cv2.line(copy1,(line[l][0],line[l][1]),(line[l][2],line[l][3]),(0,0,255),2)
				# cv2.imshow("Hough Lines",copy1) ;
				# print priori_sum ;
				cv2.drawContours(copy,[hull],-1, (255,0,0), 3)
				cv2.imshow("Hull",copy) ;
				cv2.waitKey(0) ;
				quads.append(hull) ;
				priori.append(priori_sum) ;
						
	mArea = -1 ; miArea = -1 ;		
	if len(quads) > 0 :				
		maxP = max(priori) ;
		maxiP = priori.index(max(priori)) ;
		print 'Max vote ' , maxP ;
		for i in range(len(priori)) :
			if priori[i] == maxP :
				print 'Found maxP quad ' , i , ' ' , priori[i] ; 
				tArea = cv2.contourArea(quads[i]) ;
				if tArea > mArea :
					mArea = tArea ;
					miArea = i ;

	return quads[miArea] ;				


def main() :
	
	path =  './Curves_to_Lines/' ;  # './EdgeDetection_Output/'
	dest = './Best Quadrilaterals/Latest/' #'./Labels/Incorrect/Curves_To_Lines/Best Quad/' ;	
	mypath = './Inputs/' ;
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] ;
	origN = [] ; scaledN = [] ;

	wf = open("orig.txt",'r') ;
	origN = wf.read() ;
	origN = ast.literal_eval(origN) ;

	wf = open("scaled.txt",'r') ;
	scaledN = wf.read() ;
	scaledN = ast.literal_eval(scaledN) ;

	digi = [] ; right = [] ; wrong = [] ;
	flag = True ;
	try :
		for file in onlyfiles :
			
			if file != 'IMG_0557.JPG' :
				continue ;
			print 'Processing File : ' , file
			img = cv2.imread(path + file) ;
			#img = imutils.resize(img,height = 800) ;
			#img = cv2.imread('./test/821_1.png') ;
			#orig = cv2.imread("Inputs/" + origN[scaledN.index(file)] ) ;
			orig = cv2.imread(mypath + file) ;
			#orig = imutils.resize(orig,height = 800) ;
			#scaled = cv2.imread('Scaled_Inputs/' + file) ;
			scaled = orig.copy() ;
			copy = orig.copy() ;
			edged = cv2.imread(path + file) ;
			# cv2.imshow("Original" , orig) ;
			# cv2.imshow("Scaled" , scaled) ;
			h2 , w2 = orig.shape[:2] ;
			h1 , w1 = img.shape[:2] ;

			quad = np.array(houghT(img));
			# cv2.line(orig,(0,0),(0,h),(0,0,255),3)
			# cv2.line(orig,(0,h),(w,h),(0,0,255),3)
			# cv2.line(orig,(w,h),(w,0),(0,0,255),3)
			# cv2.line(orig,(w,0),(0,0),(0,0,255),3)
			# cv2.imshow("Image Bounds",orig) ;
			# cv2.waitKey(0) ;
			cv2.drawContours(img,[quad], -1, (0,255,0), 3)
			cv2.imwrite('./temp/best_quad.png',img) ;
			cv2.imshow("Best Quadrilateral",img) ;
			cv2.waitKey(0) ;
			#cv2.moveWindow("Best Quadrilateral",500,0) ;
			

			print quad ;
			pts = quad.tolist() ;
			pts.sort() ;
			#print 'Before : ' , pts ;
			if pts[0][0][1] > pts[1][0][1] :
				temp = pts[0] ;
				pts[0] = pts[1] ;
				pts[1] = temp ;

			if pts[2][0][1] > pts[3][0][1] :
				temp = pts[2] ;
				pts[2] = pts[3] ;
				pts[3] = temp ;

			
			# converting from resizes coordinates to actual coordinates 
			ptsA = np.zeros((4,1,2)) ;
			i = 0 ;
			for pt in pts :
				ptsA[i][0][0] = int(math.ceil(( float(pt[0][0]) / w1 ) * w2 )) ;
				ptsA[i][0][1] = int(math.ceil(( float(pt[0][1]) / h1 ) * h2 )) ; 
				i += 1 ;

			
			#print 'Orig filename ' , origN[scaledN.index(file)] , 'Shape ' , orig.shape ;
			cv2.drawContours(orig,[cv2.convexHull(np.int32(ptsA))], -1, (0,0,255), 3)
			cv2.drawContours(scaled,[quad], -1, (0,0,255), 3)
			
			ptsA = np.float32(ptsA) ;
			#print ptsA , pts  , np.float32(pts) ;
			pts2 = np.float32([[0,0],[0,h2],[w2,0],[w2,h2]]) ;
			M = cv2.getPerspectiveTransform(ptsA,pts2) ;
			dst = cv2.warpPerspective(copy,M,(w2,h2)) ;
			cv2.imwrite('temp/' + file , dst ) ;
			cv2.imshow("Original" , orig) ;
			cv2.imshow("Resized" , scaled) ;
			cv2.imshow("Transformed" , dst) ;
			# cv2.imwrite('Final/Quad/'+ file , orig ) ;			
			cv2.waitKey(0) ; 
			#cv2.moveWindow("Edged",img.shape[:2][1] + 400 ,0) ;
			
			#correct = cv2.waitKey(0) ;
			# digital = input('Digital?') ;
			# correct = input('Correct?') ;
			# if digital == 1 :
			# 	digi.append(file) ;
			# if correct == 49 :
			# 	cv2.imwrite('Benchmark/NewRight/'+file,orig) ;
			# elif correct == 48 :
			# 	cv2.imwrite('Benchmark/NewWrong/'+file,orig) ;
			#print 'Digital Count : ' , len(digi) ;
			# print 'Right Count : ' , len(right) ;
			# print 'Wrong Count : ' , len(wrong) ;

	except Exception , e :
		# f = open('Benchmark/digital.txt','w+') ;	
		# f.write(str(digi)) ;
		# f = open('Benchmark/right.txt','w+') ;
		#f.write(str(right)) ;
		print e ;
		# f = open('Benchmark/wrong.txt','w+') ;
		# f.write(str(wrong)) ;

main() ;
