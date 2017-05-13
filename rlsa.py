#!/usr/bin/python
# vim: set ts=2 expandtab:
"""
Module: run_length_smoothing.py
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Thursday, August 1st 2013
  Experiment to use run length smoothing
  techniques to detect vertical or horizontal
  runs of characters in cleaned manga pages.
  
"""

import numpy as np
import cv2
import sys
from operator import itemgetter 
import scipy.ndimage
from pylab import zeros,amax,median
from os import listdir
from os.path import isfile, join ,isdir
import os

def vertical_run_length_smoothing(img, v_threshold):
  vertical = img.copy()
  (rows,cols)=vertical.shape
  #print "total rows " + str(rows) + " total cols "+ str(cols)
  for row in xrange(rows):
    for col in xrange(cols):
      value = vertical.item(row,col)
      if value == 0:continue
      next_row = row+1
      while True:
        if next_row>=rows:break
        if vertical.item(next_row,col)>0 and next_row-row<=v_threshold:
          for n in range(row,next_row):
            vertical.itemset(n,col,255)
          break
        if next_row-row>v_threshold:break
        next_row = next_row+1
  return vertical

def horizontal_run_lendth_smoothing(img, h_threshold):
  horizontal = img.copy()
  (rows,cols)=horizontal.shape
  #print "total rows " + str(rows) + " total cols "+ str(cols)
  for row in xrange(cols):
    for col in xrange(rows):
      value = horizontal.item(col,row)
      if value == 0:continue
      #print "row : " + str(row) + " col: " + str(col)
      next_row = row+1
      while True:
        if next_row>=cols:break
        if horizontal.item(col,next_row)>0 and next_row-row<=h_threshold:
          for n in range(row,next_row):
            horizontal.itemset(col,n, 255)
            #horizontal[col,n]=255
          break
          #print 'setting white'
          #binary[row,col]=255
        if next_row-row>h_threshold:break
        next_row = next_row+1
  return horizontal

def RLSO(img, h_threshold, v_threshold):
  horizontal = horizontal_run_lendth_smoothing(img, h_threshold)
  #vertical = vertical_run_length_smoothing(img, v_threshold)
  #run_length_smoothed_or = cv2.bitwise_or(vertical,horizontal)
  return horizontal #run_length_smoothed_or

def RLSA(img, h_threshold, v_threshold):
  horizontal = horizontal_run_lendth_smoothing(img, h_threshold)
  vertical = vertical_run_length_smoothing(img, v_threshold)
  run_length_smoothed_and = cv2.bitwise_and(vertical,horizontal)
  return run_length_smoothed_and ;

# method to extract contours , sort them by minY 
def sortLines(rlsa_img) :

    img_c, contours, hierarchy = cv2.findContours(rlsa_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) ;
    orig = rlsa_img.copy() ;
    contours.reverse() ;

    '''
    for c in contours :

      orig.fill(255) ;
      cv2.drawContours(orig,[c],0,(0,0,0),-1) ;
      #cv2.imshow('contour',orig) ;
      #cv2.waitKey(0) ;
    '''

    return contours ;

def sortMSERs(img , lines , msers) :


  line_group = [] ;
  i = 0 ;

  # group msers based on line
  for line in lines :

    img.fill(255) ;
    cv2.drawContours(img,[line],0,(0,0,0),-1) ;
    line_group.append([]) ;
    for pts in msers :
      crop = img[pts[1] : pts[3] , pts[0] : pts[2]] ;
      # mser falls inside the line
      if np.amin(crop) == 0 :
        line_group[i].append(pts) ;

    i += 1 ;


  # sort grouped msers left to right
  for i in range(len(line_group)) :
    line_group[i] = sorted(line_group[i], key=itemgetter(0)) ;

  return line_group ;

def sortMSERs_1(img , lines , msers) :


  line_group = [] ;
  i = 0 ;

  # group msers based on line
  for line in lines :

    img.fill(255) ;
    cv2.drawContours(img,[line],0,(0,0,0),-1) ;
    line_group.append([]) ;
    for pts in msers :
      crop = img[pts[1] : pts[3] , pts[0] : pts[2]] ;
      # mser falls inside the line
      if np.amin(crop) == 0 :
        line_group[i].append(pts) ;

    i += 1 ;


  # sort grouped msers left to right
  for i in range(len(line_group)) :
    line_group[i] = sorted(line_group[i], key=itemgetter(0)) ;

  return line_group ;

def main() :

    th = 300 ;
    tv = 500 ;
    mypath = "./MSER/New/Line/Photo/" ;
    dest = "./RLSA/Photo/Horiz_shape/" ;
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))] ;

    for file in files :
      print 'Processing file : ' , file ;
      img = cv2.imread(mypath + file , 0) ;
      img = cv2.resize(img , (img.shape[1] / 4 , img.shape[0] / 4)) ;   
      th = img.shape[1] ;
      ret , img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV) ;
      # cv2.imshow('img',img) ;
      # cv2.waitKey(0) ;
      img = img / 255 ;
      rlsa_img = RLSO(img , th , tv) ;
      sorted_lines = sortLines(rlsa_img) ;
      
#main() ;

