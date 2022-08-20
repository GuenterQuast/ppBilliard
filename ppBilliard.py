#!/usr/bin/env python3
"""Project ppBilliard

   - Tracking of two round shapes in a webcam stream 
   - replace  them by symbolic representation of a proton 
     (with quarks and gluons inside)
   - evaluate the kinematics of colliding objects, equivalent to 
     energy, impact parameter and asymmetry

   Author: Guenter Quast, initial version Aug. 2022  
"""

# --- import the necessary packages
import sys, os, argparse, random, time, yaml, numpy as np

from collections import deque
# this is the packages doing most of the work
import cv2 as cv    # openCV package

# --- helper functions

class hsvRangeFinder(object):
  """Support Class for video tracking of colored objects

      - initialize webcam
      - interactively find hsv-range of trackable object
  """

  # A required callback method that goes into the trackbar function.
  @staticmethod
  def do_nothing(x):
    pass
  
  def __init__(self, vdev_id=0, video_width=None, video_height=None, fps=None):
    # initialize webcam 
    self.vcap = cv.VideoCapture(videodev_id) # 0, 2, 4 ...
    # set webcam options
    if video_width is not None:
      self.vcap.set(3, video_width)  # width
    if video_height is not None:
      self.vcap.set(4, video_height)  # height
    if fps is not None:
      self.vcap.set(5, fps)    # frame rate    

    """ VideoCapture.set() command codes
    0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
    1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
    3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    5. CV_CAP_PROP_FPS Frame rate.
    6. CV_CAP_PROP_FOURCC 4-character code of codec.
    7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
    9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
    10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
    11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
    12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
    13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
    14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
    15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
    16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
    17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
    18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
"""

    # create a window named trackbars.
    cv.namedWindow("Trackbars")
    # create 6 trackbars to control the lower and upper range of HSV parameters
    #    0 < =H <= 179, 0 <= S <= 255, 0 <= V <= 255
    cv.createTrackbar("L - H", "Trackbars", 0, 179, self.do_nothing)
    cv.createTrackbar("L - S", "Trackbars", 0, 255, self.do_nothing)
    cv.createTrackbar("L - V", "Trackbars", 0, 255, self.do_nothing)
    cv.createTrackbar("U - H", "Trackbars", 179, 179, self.do_nothing)
    cv.createTrackbar("U - S", "Trackbars", 255, 255, self.do_nothing)
    cv.createTrackbar("U - V", "Trackbars", 255, 255, self.do_nothing)

  def run(self):

    hsv_dict = None  #output dictionary

    while True:
      # Start reading the webcam feed frame by frame.
      ret, frame = self.vcap.read()
      if not ret:
        break

      # Flip the frame horizontally (Not required)
      #frame = cv.flip( frame, 1 ) # 0:x 1:y -1:x & y
    
      # Convert the BGR image to HSV image.
      hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
      # Get trackbar value in real timem
      l_h = cv.getTrackbarPos("L - H", "Trackbars")
      l_s = cv.getTrackbarPos("L - S", "Trackbars")
      l_v = cv.getTrackbarPos("L - V", "Trackbars")
      u_h = cv.getTrackbarPos("U - H", "Trackbars")
      u_s = cv.getTrackbarPos("U - S", "Trackbars")
      u_v = cv.getTrackbarPos("U - V", "Trackbars")
 
      # Set lower and upper HSV range according value selected by trackbar
      self.lower_range = np.array([l_h, l_s, l_v])
      self.upper_range = np.array([u_h, u_s, u_v])
    
      # Filter image and get binary mask; white represents target color
      mask = cv.inRange(hsv, self.lower_range, self.upper_range)
      # Converting binary mask to 3 channel image for stacking it with others
      mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
      # visualize the real part of the target color (Optional)
      result = cv.bitwise_and(frame, frame, mask=mask)
    
      # dispay mask, original and masked frames
      stacked = np.hstack((mask_3, frame, result))    
      # resize frame to 40% 
      cv.imshow('Trackbars',cv.resize(stacked, None, fx=0.4, fy=0.4))
    
      # <esc> or 'q'  exit the program
      key = cv.waitKey(1)
      if (key == 27) or (key == ord('q')): return hsv_dict
    
      # save and return if user presses `s`
      if key == ord('s'):
        hsv_dict = {}
        hsv_dict = {'hsv_l' : [l_h, l_s, l_v],
                    'hsv_h' : [u_h, u_s, u_v] }
        return hsv_dict

  def stop(self):
    # Finally, release camera & destroy windows.    
    self.vcap.release()
    cv.destroyAllWindows()

  # --- end class hsvRangeFinder

class vMouse(object):
  """provides basic mouse functionality in video windwo
  """

  def  __init__(self, window):
    self.WNam = window

    # initialize mouse coordinates
    self.xmouse = None
    self.ymouse = None
    # middle button
    self.xmouseM = None
    self.ymouseM = None
    # right button 
    self.xmouseR = None
    self.ymouseR = None
    cv.setMouseCallback(self.WNam, self.mouseklick)

    # list for buttons
    self.buttons=[]
    
  def mouseklick(self, mevent, x, y, flags, param):
    """Callback function for mouse klicks in Video Window
    """
    if mevent == cv.EVENT_LBUTTONDOWN:
      #print("  --> Left mouse klick at ", x, y)
      self.xmouse = x
      self.ymouse = y
    if mevent == cv.EVENT_MBUTTONDOWN:
    #  print("  --> Middle mouse klick at ", x, y)
      self.xmouseM = x
      self.ymouseM = y

    if mevent == cv.EVENT_RBUTTONDOWN:
    #  print("  --> Right mouse klick at ", x, y)
      self.MxmouseR = x
      self.MymouseR = y

  def createButton(self, frame, lb, w_pad, h_pad, color, text=None):
    """Create a "Button" in video frame as colored pad

         - coordinates of positon: Left - Bottom pixel
         - w_pad and h_pad: dimensions of pad
         - color: pad color
         - text:  optional text
         - returns integer id of button
    """
    h, w = frame.shape[:2]
    # calculate coordinates of right-top corner
    rt = (lb[0]+w_pad, lb[1]-h_pad)
    # store button
    button = (lb, rt, color, text) 
    self.buttons.append( button)
    self.drawButtons(frame, buttonlist=[button])
    #
    return len(self.buttons) # index of the new button

  def drawButtons(self, frame, buttonlist = None):
    """redraw button in new frame
    """
    buttons = self.buttons if buttonlist is None else buttonlist
    for i, b in enumerate(buttons):
      lb = b[0]       
      rt = b[1]
      color = b[2]
      text = b[3]
      cv.rectangle( frame, lb, rt, color, -1)
      cv.rectangle( frame, lb, rt, (150,150,150), 2)
      cv.rectangle( frame, lb, rt, (0,0,0), 1)
      if text is not None:
        w_pad = rt[0]-lb[0]
        h_pad = lb[1] - rt[1]
        cv.putText(frame, text,
                (lb[0]+w_pad//4, lb[1]-h_pad//3), 
                 cv.FONT_HERSHEY_TRIPLEX, 0.7, (50,50,50) )
    return

  def deleteButtons(self):
    """Remove all buttons from list"""
    self.buttons = []
      
  def checkButtons(self):
    """Ceck if any button klicked""" 
    if self.xmouse is not None and self.ymouse is not None:
      for i, b in enumerate(self.buttons):
        lb = b[0]
        rt = b[1]
        if (lb[0] <= self.xmouse <= rt[0]) and \
           (lb[1] >= self.ymouse >= rt[1]):
          self.xmouse = None
          self.ymouse = None
          return i+1
      self.xmouse = None
      self.ymouse = None
    # no klick in one of the buttons, return 0     
    return 0    
# <-- end class vMouse

def smoothImage(frame):
  # blur it convert it to the HSV color space
  blurred = cv.GaussianBlur(frame, (11, 11), 0)
  return cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

def findObject_byColor(hsv_image, colLower, colUpper, algo = "Contours"):  
  # construct a mask for the color range given by colLower - colUpper,
  #  then perform a series of dilations and erosions to remove
  #  any small blobs left in the mask
  mask = cv.inRange(hsv_image, colLower, colUpper)
  mask = cv.erode(mask, None, iterations=2)
  mask = cv.dilate(mask, None, iterations=2)

  if algo=="Hough":
    # find circles with Hough transformation after edge detection  
    edges = cv.Canny(mask, 50, 150)
    cv.imshow("canny", edges)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT,
                              dp=4, minDist=200, param1=50, param2=200,
       minRadius=30, maxRadius=200)
    xy, r = None, 0
    if circles is not None:
      circles = np.round(circles[0, :]).astype("int")
      for c in circles:
        if c[2]>r:
          r = c[2]
          xy =(c[0],c[1])
    return xy, r
  
  # find contours in the mask and (x, y) of center
  cnts, _obj2 = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
         cv.CHAIN_APPROX_SIMPLE)
  xy , r = None,  0
  # only proceed if at least one contour was found
  if len(cnts) > 0:
    # find the largest contour in the mask, then use it to
    # compute the minimum enclosing circle and centroid
    c = max(cnts, key=cv.contourArea)
    ( (x, y), r) = cv.minEnclosingCircle(c)
    xy, r = (int(x), int(y)), int(r)
    #M = cv.moments(c)
    #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
  return xy, r

def plotTrace(frame, points, lw=3, color=(100,100,100)):
  # draw list of points (an object trace)
  for i in range(1, len(points)):
    # if either of the tracked points are None, ignore them
    if points[i - 1] is None or points[i] is None:
      continue
    # otherwise, draw the connecting lines
    cv.line(frame, points[i - 1], points[i], color, lw)

def blur_region(frame, xy, wxy, ksize=7):
  """Blur rectangular region  in frame 

      xy: tuple (x,y) of coordinates
      wxy: tuple of x- and y-widht
      region of interest: x + wx,  y + wy or 
                          x +/- wx, y +/- wx if wx, wy negative
  """
  x = xy[0]
  y = xy[1]
  wx = wxy[0]
  wxm=wx if wx <0 else 0
  wy = wxy[1]
  wym=wy if wy < 0 else 0
  
  h, w = frame.shape[:2]
  roi = frame[ max(0, y+wym) : min(h, y+abs(wy)),
               max(0, x+wxm) : min(w, x+abs(wx)) ]
  frame[max(0, y+wym) : min(h, y+abs(wy)),
        max(0, x+wxm) : min(w, x+abs(wx))] = \
    cv.blur(roi, (ksize, ksize))
  return frame      

def blur_circular_region(frame, xy, r, ksize=7):
  """Blur circular in frame 
      xy: tuple (x,y) of centre coordinates
      r: radius 
      region of interest: x +/- r, y +/- r 
  """
  x = xy[0]
  y = xy[1]
  
  # extract region of interest (roi) from image
  h, w = frame.shape[:2]
  roi = frame[ max(0, y-r) : min(h, y+r),
               max(0, x-r) : min(w, x+r) ]
 # create masks with (white on black and black on white) circles 
  roi_shape = ( roi.shape[0], roi.shape[1], 1 )
  mask_circ = np.full( roi_shape, 0, dtype=np.uint8)
  cv.circle(mask_circ, (r, r), r, (255, 255, 255),  -1)
  invmask_circ = cv.bitwise_not(mask_circ)
 # create a temporary image for blurred version 
  tmpImg = roi.copy()
  tmpImg = cv.blur(tmpImg, (ksize, ksize))
 # copy blurred circle and background to its position in image
  frame[max(0, y-r) : min(h, y+r), max(0, x-r) : min(w, x+r)] = \
    cv.add( cv.bitwise_and(tmpImg, tmpImg, mask = mask_circ),
            cv.bitwise_and(roi, roi, mask = invmask_circ) )
  return frame

def blur_elliptical_region(frame, xy, ab, angle, ksize=7):
  """Blur elliptical region in frame 
      xy : tuple (x,y) of centre coordinates
      ab : tuple (a,b) of axes of ellipse
      angle: rotation angle 
  """
  x = xy[0]
  y = xy[1]
  a = ab[0]
  b = ab[1]  
 # extract region of interest (roi) from image
  h, w = frame.shape[:2]
  w0 = max(a,b)
  roi = frame[ max(0, y-w0) : min(h, y+w0),
               max(0, x-w0) : min(w, x+w0) ]
 # create masks with (white on black and black on white) ellipses
  roi_shape = ( roi.shape[0], roi.shape[1], 1 )
  mask_ellipse = np.full( roi_shape, 0, dtype=np.uint8)
  cv.ellipse(mask_ellipse, (w0, w0), ab, angle, 0, 360,
                 (255, 255, 255),  -1)
  invmask_ellipse = cv.bitwise_not(mask_ellipse)
 # create a temporary image for blurred version 
  tmpImg = roi.copy()
  tmpImg = cv.blur(tmpImg, (ksize, ksize))
 # copy blurred circle and background to its position in image
  frame[max(0, y-w0) : min(h, y+w0), max(0, x-w0) : min(w, x+w0)] = \
    cv.add( cv.bitwise_and(tmpImg, tmpImg, mask = mask_ellipse),
            cv.bitwise_and(roi, roi, mask = invmask_ellipse) )
  return frame

def crossfade(win, img1, img0, alpha0=0., nframes=20):
  """cross-fade from img0 to img1, starting with alpha0,
     display in window win 
  """
  dalpha = 1. - alpha0
  for i in range(nframes+1):
    a = alpha0 + i * dalpha/nframes        
    cv.imshow(win,
              cv.addWeighted(img1, a, img0, 1-a, 0.) )
    cv.waitKey(40) # show blended image
  return img1
     

class proton(object):
  """ Draw a proton as 
    (r-g-b colored circles as quarks and lines as gluons inside a big circle)
  """
  #  define properties of objects inside proton as class variables
  c120 = -0.5    # cos 120
  s120 = 0.866  # sin 120
  cred = (0, 0, 255)
  cgreen = (0, 255, 0)
  cblue = (255, 0 ,0)
  cbkg = (80, 80, 150)
  cquark= [cred, cgreen, cblue]
  cglue = (255, 240, 240)
  showQuarkCharges = True # quakrs as up/down triangles
  blur = True             # blur to show quantum uncertainty 

  # pp collision
  ccolor = (200,255,255)

  # helper fuctions
  @staticmethod
  def rnd(r):
    # produce random number for quark position inside proton of radius r
    return random.gauss(0, 1.) * r/10

  @staticmethod
  def triangleUp(r, center):
    # quark as triangle tip up
    c = np.array(center)
    xy1 = c+ np.array([0., r])
    xy2 = c+ np.array([-0.866*r, -r/2])
    xy3 = c+ np.array([0.866*r, -r/2])
    cnt = np.int32( np.array([xy1, xy2, xy3]) )
    return cnt

  @staticmethod
  def triangleDown(r, center):
    # quark as triangle tip down 
    c = np.array(center)
    xy1 = c+ np.array([0., -r])
    xy2 = c+ np.array([-0.866*r, r/2])
    xy3 = c+ np.array([0.866*r, r/2])
    cnt = np.int32( np.array([xy1, xy2, xy3]) )
    return cnt
  
  # code to draw as classmethod (gives acces to data an methods of class)
  @classmethod
  def draw(c, frame, x, y, r, nobkg=False):
    # draw a symbolic Proton with three quarks on top of tracked ball  
    rq = int(r/5) # radius of a circle for quark
    r12 = r/2
    # simulate "quantum movement" by adding random displacements
    x1, y1 = (int(x+r12+c.rnd(r)), int(y+c.rnd(r))) 
    x2, y2 = (int(x + c.c120*r12+c.rnd(r)), int(y + c.s120*r12+c.rnd(r)))
    x3, y3 = (int(x + c.c120*r12+c.rnd(r)), int(y - c.s120*r12+c.rnd(r)))
    #   draw cirlce in background color          
    if not nobkg:
      cv.circle(frame, (int(x), int(y)), int(r), c.cbkg, -1)
    # draw "gluons" connecting quarks
    cv.line(frame, (x1,y1), (x2,y2), c.cglue, thickness = 5)
    cv.line(frame, (x2,y2), (x3,y3), c.cglue, thickness = 5)
    cv.line(frame, (x3,y3), (x1,y1), c.cglue, thickness = 5)
    # randomize quark color
    cq = c.cquark
    random.shuffle(cq)
    # draw quarks ...
    if not c.showQuarkCharges: 
    # d ... as circles
      cv.circle(frame, (x1, y1), rq, cq[0], -1)
      cv.circle(frame, (x2, y2), rq, cq[1], -1)
      cv.circle(frame, (x3, y3), rq, cq[2], -1)
    else:
      # ... as triangles (up or down) with randomized quark charge
      drawQ = [c.triangleUp, c.triangleUp, c.triangleDown]
      random.shuffle(drawQ)
      cnt1 = drawQ[0](rq, (x1,y1)) 
      cnt2 = drawQ[1](rq, (x2,y2)) 
      cnt3 = drawQ[2](rq, (x3,y3))
      cv.drawContours(frame, [cnt1], 0, cq[0], -1)
      cv.drawContours(frame, [cnt2], 0, cq[1], -1)
      cv.drawContours(frame, [cnt3], 0, cq[2], -1)

    if c.blur:
      _ = blur_circular_region(frame, (x, y), r, ksize=max(2, r//4))  

  @classmethod
  def drawCollision(c, frame, v_C, dist, angle):
    """plot proton-proton collision:
      
       input:
         - vector to center of collision (int, in pixels)
         - distance vector between protons (int, in pixels)
    """

    # distance between collisions
    R = (2*dist)//3
    x = v_C[0]
    y = v_C[1]
    sign= 1 if random.random()>0.5 else -1
    x1 = int(x+R/2)
    y1 = int(y-sign*R/2)
    x2 = int(x-R/2)
    y2 = int(y+sign*R/2)

    rq = int(R/5) # radius of a circle for quark

    # draw an ellipse
    a = (3*R)//2
    b = R
    cv.ellipse(frame, v_C, (a,b),
               angle, 0, 360,
               c.ccolor, -1)

    # put proton remnants inside
    c.draw(frame, x+R//5, y, 3*R//4, nobkg=True) 
    c.draw(frame, x-R//5, y, 3*R//R, nobkg=True)

    # some scattered quarks
    cq = c.cquark
    random.shuffle(cq)
    if not c.showQuarkCharges: 
    # d ... as circles
      cv.circle(frame, (x1, y1), rq, cq[0], -1)
      cv.circle(frame, (x2, y2), rq, cq[1], -1)
    else:
      # ... as triangles (up or down) with randomized quark charge
      drawQ = [c.triangleUp, c.triangleUp, c.triangleDown]
      random.shuffle(drawQ)
      cnt1 = drawQ[0](rq, (x1,y1)) 
      cnt2 = drawQ[1](rq, (x2,y2)) 
      cv.drawContours(frame, [cnt1], 0, cq[0], -1)
      cv.drawContours(frame, [cnt2], 0, cq[1], -1)
    # add a gluon  
    cv.line(frame, (x1,y1), (x2,y2), c.cglue, thickness = 5)

    if c.blur:
      _ = blur_elliptical_region(frame, v_C, (a,b), angle, ksize=max(2, R//4))
      
def bgr2hsv(b, g, r):
  # convert Blue-Green-Red to Hue-Saturation-Value
  c = np.uint8([[[b, g, r]]])
  return cv.cvtColor( c, cv.COLOR_BGR2HSV)


class frameRate():
  """Calculate video frame rate from calls after getting each frame
  """
  def __init__(self):
    self.t0 = time.time()
    self.dt = 0
    self.Nframes = 0
    self.nframes = 0
    self.i = 0
    self.ch_toggle = ['*','+']
    self.rate = 0.

  def timeit(self, quiet=False):    
    # to be called after every frame recieved
    #     prints frame rate every 10 frames
    t = time.time()
    self.dt += t-self.t0
    self.t0 = t
    self.nframes += 1
    self.Nframes += 1
    if self.Nframes%10 == 0:
      self.rate = self.nframes/self.dt
      if not quiet:
        print(self.ch_toggle[self.i],"      frame rate: ",
              int(self.nframes/self.dt)
            , "/s     ", end='\r')
        self.i = 1 - self.i
      self.dt=0
      self.nframes = 0
    return(self.rate)

#---  end helpers --------------------------------------------

class ppBilliard(object):
  """Track colored objects in a video and replace by symbolic Proton
  """

  def __init__(self, vdev_id=0, video_width=None, video_height=None, fps=24):

    self.first = True # first round
    self.playIntro = True # show intro if true
    
    # set webcam options
    self.vdev_id = vdev_id
    self.cam_width = video_width
    self.cam_height = video_height
    self.cam_fps = fps
      
    # properties of trackable objects
    hsv_from_file = True # get colors from yaml-file if true
    
    # read colors of trackable objects in hsv-format from files (if present)
    col_pink = 165  # pink 
    if hsv_from_file:
      try:
        fn = 'object1_hsv.yml'  
        with open(fn, 'r') as fs:
          d = yaml.load(fs, Loader=yaml.Loader)
          self.obj_col1 = [np.array(d['hsv_l'], dtype=np.int16),
                           np.array(d['hsv_h'], dtype=np.int16)]
      except:
        print("!!! reading file " + fn + " failed, using defaults")  
        col1Lower = (col_pink - 5, 140, 140)
        col1Upper = (col_pink + 5, 220, 220)
        self.obj_col1 = [col1Lower, col1Upper]
    else:
      col1Lower = (col_pink - 5, 140, 140)
      col1Upper = (col_pink + 5, 220, 220)
      self.obj_col1 = [col1Lower, col1Upper]

    # color range of 2nd object (pink):
    col_yel = 18    # dark yellow
    if hsv_from_file:
      try:
        fn = 'object2_hsv.yml'  
        with open(fn, 'r') as fs:
          d = yaml.load(fs, Loader=yaml.Loader)
          self.obj_col2 = [np.array(d['hsv_l'],dtype=np.int16),
                           np.array(d['hsv_h'], dtype=np.int16)]
      except:
        print("!!! reading file " + fn + " failed, using defaults")  
        col2Lower = (col_yel - 5, 65, 150) 
        col2Upper = (col_yel + 5, 200, 200)
        self.obj_col2 = [col2Lower, col2Upper]
    else:
      col2Lower = (col_yel - 5, 65, 150) 
      col2Upper = (col_yel + 5, 200, 200)
      self.obj_col2 = [col2Lower, col2Upper]

    # color range of reserve object (light-green):
    col_green = 55  # green (RGB: 0, 255, 0)
    col3Lower = (col_green - 10, 60, 60)
    col3Upper = (col_green + 10, 200, 200)
    self.obj_col3 = [col3Lower, col3Upper]

    #col_blue = 100  # blue

    self.obj_min_radius = 5

    #
    # --- define video parameters
    #
    # width of video
    self.max_video_width = 1024 # 800 or 600 if CPU limits

    self.pts1 = deque(maxlen=args["buffer"]) # queue for object1 xy
    self.pts2 = deque(maxlen=args["buffer"]) # queue for object2 xy

    # no video source yet
    self.vs = None
    self.useCam = True if not args.get("video", False) else False
    
    #
    # --- output greeter
    #  
    print("\n Script ", sys.argv[0], " executing")
    if self.useCam:
      print("  reading video device ", videodev_id)
    else:  
      print("  reading from file ", args["video"])
    print("      type 'q' or <esc> to exit in video window\n")

    # allow the camera or video file to warm up
    time.sleep(0.5)

  def init(self):
    """initialize new run"""
    #
    self.nframes = 0 #frame counter 
    #
    # variables to store result
    self.CollisionResult = None
    self.resultFrame = None
    #
    self.pts1.clear() # queue for object1 xy
    self.pts2.clear() # queue for object2 xy
    #
    # load background image to be overlayed on webcam frames
    try:
      self.bkgimg = cv.imread('images/RhoZ_empty.png', cv.IMREAD_COLOR)
    except:
      self.bkgimg = None
    #
    # (re-)initialize video input
    if self.vs is None or (
       self.vs is not None and not self.vs.isOpened()):
      if not args.get("video", False):
        # if no video path supplied, grab reference to webcam
        self.useCam = True
        self.vs = cv.VideoCapture(self.vdev_id)
        if self.cam_width is not None:
          self.vs.set(3, self.cam_width)
          # print("setting cam width: ", self.cam_width)
        if self.cam_height is not None:
          self.vs.set(4, self.cam_height)
          # print("setting cam height: ", self.cam_height)
        if self.cam_fps is not None:
          # print("setting cam fps: ", self.cam_fps)
          self.vs.set(5, self.cam_fps)      
      else:
        # otherwise, grab a reference to the video file
        self.useCam = False
        self.vs = cv.VideoCapture(args["video"])
    #    
    # create video window for program output (webcam + tracked objects)
    self.WNam = "VWin"
    cv.namedWindow(self.WNam, cv.WINDOW_AUTOSIZE)
    if args["fullscreen"]:
      cv.setWindowProperty(self.WNam,
                         cv.WND_PROP_FULLSCREEN,
                         cv.WINDOW_FULLSCREEN)
    # initialize mouse in video
    self.mouse = vMouse(self.WNam)
    
  def showIntro(self, frame):
    """Play a short sequence of frames as introduction
    """
    h, w = frame.shape[:2]
    img = cv.imread('ppBilliard_intro.png', cv.IMREAD_COLOR)
    img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)

    # animate two protons on Intro image
    r = h//25
    y = h//2
    nf = 25 # numer of frames to show

    tmpimg = img.copy() 
    roi1 , roi2 = None, None
    for i in range(nf):
      x1 = r +  (i * (w-r))//2 // nf
      x2 = w - r -  (i * (w-r))//2 // nf
      # save background, then draw proton
      roi1 = tmpimg[ max(0, y-r-1) : min(h, y+r+1),
                 max(0, x1-r-1) : min(w, x1+r+1) ].copy()
      proton.draw(tmpimg, x1, y, r)
      # save background, then draw proton2 
      roi2 = tmpimg[ max(0, y-r-1) : min(h, y+r+1),
                 max(0, x2-r-1) : min(w, x2+r+1) ].copy()
      proton.draw(tmpimg, x2, y, r)      
# --- simpler version, but needs copy of full image each time in loop      
#      tmp = img.copy() 
#      proton.draw(tmp, x1, y, r)      
#      proton.draw(tmp, x2, y, r)      
#  ---
      cv.imshow(self.WNam, tmpimg)
      cv.waitKey(70) # wait until key pressed
      # write background (overwrite proton image)
      if roi1 is not None:
        tmpimg[max(0, y-r-1) : min(h, y+r+1),
               max(0, x1-r-1) : min(w, x1+r+1)] = roi1      
      if roi2 is not None:
        tmpimg[max(0, y-r-1) : min(h, y+r+1),
               max(0, x2-r-1) : min(w, x2+r+1)] = roi2      
    # <-- end proton animation

    # draw collision symbol 
    proton.drawCollision(tmpimg, (w//2, h//2), 2*r, 0.)
    cv.imshow(self.WNam, tmpimg)

    # cross-fade to camera image 
    frame = crossfade(self.WNam, frame, tmpimg)


  def analyzeCollision(self, v_c1, v_c2, v_c1_0, v_c2_0):
    """ analyze kinematics of collision of two objects

      Input: 

        - 2d-vectors of object coordinates (int) for actual and 
        - previous positions (2 frames time distance)

      Returns: 
        - result dictionary:

          - Coordinates of impact
          - length and angle of distance vector at impact
          - Escore: collision energy [~cm²/s²] for object with mass 1
          - Iscore:    impact parameter[0, 1] 
          - Asymmetry: momentum asymmetry [-1, 1]
          - Score:     Escore * IScore 
    """
   #   coordinates in pixels, time in 1/framerate
   # collision point 
    v_C = np.int32(0.5*(v_c1 + v_c2))
   # distance of obcect centers when colliding
    v_dist = v_c2 - v_c1
    dist = np.sqrt(np.inner(v_dist, v_dist))
    angle = int(np.arctan2(v_dist[1], v_dist[0]) * 180/np.pi) 
   # velocities in pixels/time_beweeen_frames
    nf = 2
    v_v1 = (v_c1 - v_c1_0)/nf
    v1 = max(1, np.sqrt(np.inner(v_v1, v_v1)))
    v_v2 = (v_c2 - v_c2_0)/nf
    v2 = max(1, np.sqrt(np.inner(v_v2, v_v2)))
   # centre-of-mass   
    v_cms = v_v1 + v_v2
    v_vcms1 = v_v1 - v_cms
    vcms1sq = np.inner(v_vcms1, v_vcms1)             
    v_vcms2 = v_v2 - v_cms
    vcms2sq = np.inner(v_vcms2, v_vcms2)             
   # collision energy in centre-of-mass system
    pixels_per_cm = 10
    # correct for frame rate, assume image resolution 10 pixels/cm 
    Escore = 0.5*(vcms1sq + vcms2sq)/pixels_per_cm**2 * self.framerate**2
   # impact parameter             
    Iscore = 0.5 * (abs(np.inner(v_dist/dist, v_v1/v1) ) + 
                    abs(np.inner(v_dist/dist, v_v2/v2) ) )
   # asymmetry
    Asym = np.sqrt(np.inner(v_cms,v_cms)) / (v1+v2)
    Asym = Asym if v1>v2 else -Asym
    # construct total score
    Score = int(Escore * Iscore)
    self.CollisionResult={
        'Coordinates' : (v_C[0], v_C[1]),
        'distance'    : dist,
        'angle'       : angle, 
        'Escore'      : Escore,
        'Iscore'      : Iscore,
        'Asymmetry'   : Asym,
        'Score'       : Score                          }           

  def printCollisionResult(self):
    """Print collsion parameters"""
    
    if self.CollisionResult is None:
      print(" !!! No object 'CollisionResult' found ")
      return
    
    print("\n  -->>> Collision detected <<<--")
    d = self.CollisionResult
    print("           distance: {:.2g} ".format(d['distance']))      
    print("    scores: energy: {:.4g}  ".format(d['Escore']) +
          "impact: {:.2g}  ".format(d['Iscore']) +
          "asymmetry: {:.2f}".format(d['Asymmetry']) )
    print(10*" ", "    -->>>    ** ", d['Score'], " **     <<<--")
    
  def run(self):  
    """Read Camera, track objects and detect Collision
    """

    # skip some frames after re-start in order to remove old ones
    nskip = 0 if self.first else 3
    self.first = False
    
    # initialize frame rate counter
    rate = frameRate()
    #
    # --- start loop over video frames 
    while self.vs.isOpened():

      # grab current frame
      ret, frame= self.vs.read()
      self.framerate = rate.timeit()
      
      # end of video file reached ?
      if frame is None:
        print ('no frame recieved - ending!')
        break

      # empty video buffer 
      if rate.Nframes <= nskip:
        print ('!!! skipping frame ', rate.Nframes)
        continue

      self.nframes +=1   # count frames

      if self.nframes == 1: # frame number one
        h, w = frame.shape[:2]
        print("  Video resolution: {:d}x{:d}".format(w, h))
        self.scaleVideo = False
        if w > self.max_video_width:
          self.scaleVideo = True
          self.swidth = self.max_video_width
          self.sheight = int((h*self.swidth)/w)
        else:
          self.sheight = h
          self.swidth = w
        if self.bkgimg is not None:
          self.bkgimg = cv.resize(self.bkgimg, (self.swidth, self.sheight),
                             interpolation=cv.INTER_AREA)
          
      # resize frame (may save computation time)
      if self.scaleVideo:
        frame = cv.resize(frame, (self.swidth, self.sheight),
                          interpolation=cv.INTER_AREA)      
      if self.playIntro:       
         self.showIntro(frame)
         self.playIntro = False
        
      # smooth and convert to HSV color space
      hsvImg = smoothImage(frame)

      # find objects in video frame
      xy1, r1 = findObject_byColor(hsvImg,
                                self.obj_col1[0], self.obj_col1[1])
      # only proceed if the radius meets a minimum size
      if r1 >= self.obj_min_radius:
        # draw "proton with three quarks and gluons
        proton.draw(frame, xy1[0], xy1[1], r1)      
        #cv.circle(frame, (int(xy1 [0]), int(xy1[1])), int(2),
        #       (0, 255, 255), 2)

      xy2, r2 = findObject_byColor(hsvImg,
                                self.obj_col2[0], self.obj_col2[1])
      # rprotonflionly proceed if the radius meets a minimum size
      if r2 > self.obj_min_radius:
        # draw "proton with three quarks and gluons
        proton.draw(frame, xy2[0], xy2[1], r2)      
        #cv.circle(frame, (int(xy2[0]), int(xy2[1])), int(radius2),
        #        (0, 255, 255), 2)

      # update the tracked-points queue
      self.pts1.appendleft(xy1)
      self.pts2.appendleft(xy2)

      # add detector contours 
      if self.bkgimg is not None:
        frame = cv.addWeighted(frame, 0.75, self.bkgimg, 0.25, 0.)

      # check for collisions of objects
      sawCollision = False
      l = len(self.pts1)
      if l>2 and \
        self.pts1[0] is not None and self.pts2[0] is not None \
        and self.pts1[2] is not None and self.pts2[2] is not None :
        v_c1 = np.array(xy1)
        v_c2 = np.array(xy2)
        v_dist = v_c2 - v_c1
        dist = np.sqrt(np.inner(v_dist, v_dist))
        if dist < 1.01*(r1+r2):  # objects (nearly) touched
          sawCollision=True
          v_c10 = np.array(self.pts1[2])
          v_c20 = np.array(self.pts2[2])
          # analyze kinematics
          self.analyzeCollision(v_c1, v_c2, v_c10, v_c20)
          # draw Collision
          dist = int(self.CollisionResult['distance'])
          angle = int( self.CollisionResult['angle'])
          v_C = self.CollisionResult['Coordinates']
          proton.drawCollision(frame, v_C, dist, angle)
          self.resultFrame=frame.copy()
          break

      # plot object traces from list of tracked points    
      plotTrace(frame, self.pts1, lw=2, color=(0, 0, 100) )
      plotTrace(frame, self.pts2, lw=2, color=(0, 100, 100))

      # put text on first frames
      if self.nframes < 20:
        cv.putText(frame, "New Game",
                   (self.swidth//10, self.sheight//20),
                    cv.FONT_HERSHEY_TRIPLEX, 1, (0,255,100))
      
      # show the frame on screen
      cv.imshow(self.WNam, frame)

      key = cv.waitKey(1) & 0xFF
      # if the <esc> key is pressed, stop the loop
      if (key == 27) or (key == ord('q')):
        # end  
        break
      elif (key == ord('p')):
        # pause
          cv.waitKey(-1) # wait until any key pressed

    return self.CollisionResult
    # <-- end of loop

  def showResult(self, img_path='images/3DTower_empty.png', score=None):
    """Show resultFrame and overlay image from pp collision 
    """
    # show the frame on screen
    cv.imshow(self.WNam, self.resultFrame)
    
    cv.waitKey(2000) # wait until key pressed
    h, w = self.resultFrame.shape[:2]
    event_img = cv.imread(img_path, cv.IMREAD_COLOR)
    event_img = cv.resize(event_img, (w, h), interpolation=cv.INTER_AREA)
    # crossfade to result frame
    frame = crossfade(self.WNam, event_img, self.resultFrame, 0., 50)
    print("\n    - type 'c' to continue, or 'q' or <esc> to exit in video window   - ")

    # show score:
    if score is not None:
      cv.putText(frame, " *Score: " + str(score), 
                   (w-self.swidth//3, self.sheight//25),
                    cv.FONT_HERSHEY_TRIPLEX, 1, (100,50,100))

    # draw colored fields for mouse interactions
    if len(self.mouse.buttons) == 0:
      w_pad = self.swidth//30
      h_pad = self.sheight//25
      id0 = self.mouse.createButton(frame, (0, self.sheight),
                                  w_pad, h_pad, (0,255,0), text='c')
      id1 = self.mouse.createButton(frame, (self.swidth-w_pad, self.sheight),
                                  w_pad, h_pad, (0,0,255), text='q')
    else:
      self.mouse.drawButtons()
    #  
    # show image with buttons ...
    cv.imshow(self.WNam, frame)
    #
    # ... and wait for user reply
    key = None
    while True:
      key = cv.waitKey(500) # wait 500ms or until key pressed
      id = self.mouse.checkButtons()
      if id == 1:  # green button clicked
        key = ord('c')
        break
      elif id ==2: # red button clicked
          key = ord('q')
          break
      if key is not None and key != -1:
        break
    #  
    return key
      
  def stop(self):  
  # --- clean-up at the end
  #
  # release video stream and close all windows
    self.vs.release()
    cv.destroyAllWindows()


if __name__ == "__main__":  # ------------run it ----

#
# --- parse the arguments
#
  ap = argparse.ArgumentParser()
  ap.add_argument("-f", "--fullscreen", action='store_const',
        const=True, default = False,
	help="run in fullscreen mode")
  ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
  ap.add_argument("-b", "--buffer", type=int, default=32,
	          help="max buffer size for object traces")
  ap.add_argument("-s", "--source", type=int, default=0,
	help="videodevice number")
  ap.add_argument("-c", "--calibrate", type=int, default=0,
	help="find hsv range of trackable object <1/2>")
  args = vars(ap.parse_args())

  videodev_id = args["source"] # video device number of webcam
  if args['calibrate']:
    # set up camera and interactive window
    hsv = hsvRangeFinder(videodev_id)
    # run loop
    hsv_dict=hsv.run()
    if hsv_dict is not None:
      fnam= 'object'+str(args['calibrate'])+'_hsv.yml'
      with open(fnam, 'w') as of: 
        yaml.dump(hsv_dict, of, default_flow_style=True)
      print("hsv range saved to file ",fnam) 
    hsv.stop()
  else:
   #
   # run ppBilliards
   #
   # set paths to pp event pictures
    wd = os.getcwd()
    eventpath = '/events/'
    lowScore= ('empty/', 'Cosmics/')
    highScore = ('2e/', '2mu/', 'general/')
    superScore = ('2mu2e/', '4mu/')
    
   # initialize video analysis
    ppB = ppBilliard(videodev_id,
                       video_width = 1024,
                       video_height = 800,
                       fps = 24)
  # -- loop 
    key = ord('c') 
    while key == ord('c') or key==ord(' '):
      ppB.init()
    # run video analysis from camera
      result = ppB.run()
      if result is None:
        break
    # show result 
      frame = ppB.resultFrame
      ppB.printCollisionResult()
      score = result['Score']
    # evaluate score, select event picture and show it
      # print("  *==* Your score is", int(score) )
      if score > 2000:
        eventclass = random.choice(superScore)
      elif score > 750:
        eventclass = random.choice(highScore)
      else:
        eventclass = random.choice(lowScore)
      path = os.getcwd()+eventpath+eventclass
      filelist = os.listdir(path)
      i = int(random.random() * len(filelist))
      event_img = path+filelist[i]
      # - show picture on video screen
      key = ppB.showResult(event_img, score)
      
      if key == ord('c') or key == ord(' '):       
        print("\n      running again ...\n")
    # <-- end while key == ord('c')
    
    # clean up     
    print("\n      bye, bye !\n")
    ppB.stop()