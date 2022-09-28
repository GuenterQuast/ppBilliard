****************************************
## Project **p(roton)p(roton) Billiard**
****************************************

> > [Link zur Anleitung in deutscher Sprache](README_de.md)


About
-----

Collide two balls on a table and see what happens if they
were protons in the Large Hadron Collider at CERN.

>   ![](ppBilliard.png)

This project uses a table-top game board for video tacking of colliding
round colored objects with a webcam.
Collision parameters like the equivalent centre-of-mass energy,
impact distance and asymmetry are automatically determined.
These are scaled to correspond to the parameters of a proton-proton 
collision in a high-energy collider and pictures of the traces resulting
from particle collision are shown. 


The Python Code here relies heavily on the Open Source Computer
Vision Library [OpenCV](https://opencv.org/).  
Thanks to the developers for the great work!


Created by Guenter Quast, initial Version Aug. 2022 

Installation
------------

On a standard Python (>3.6) environment the OpenCV library 
is installed via the command  
> `pip3 install opencv-python`.

After downloading the ppBilliard package, type 
`ppBilliard.py -h`, producing the following output:

```
  usage: ppBilliard.py [-h] [-f] [-v VIDEO] [-b BUFFER] [-s SOURCE]
                       [-c CALIBRATE]

  optional arguments:
    -h, --help            show this help message and exit
    -f, --fullscreen      run in fullscreen mode
    -v VIDEO, --video VIDEO
                          path to the (optional) video file
    -b BUFFER, --buffer BUFFER
                          max buffer size for object traces
    -s SOURCE, --source SOURCE
                          videodevice number
    -c CALIBRATE, --calibrate CALIBRATE
                          find hsv range of trackable object <1/2>
    -C CONFIG, --config CONFIG
                          configuration file
```

Usually, the number of the video device for the webcam is 0; if 
not, use the parameter '-s n' with the device number 'n'.


Usage of the Program
--------------------

Starting the Program without any other arguments than the device number 
will display a short trailer and then switch to a display of the webcam
output. If a trackable object like a colored ball or circle is identified, 
it is replaced by a symbolic, animated representation of a proton
and tracked in the video as you move it around.

If no object is recognized, a calibration step is needed, as described
below. After completing the color calibration, restart the program and 
move the objects around.  They should now be recognized and their round 
shapes replaced by a symbolic, animated pictures of protons, and the object 
paths are drawn on the video. If these traces come closer than the
sum of the radii of the two objects, a "collision" is detected
and the collision parameters are printed:
the equivalent centre-of-mass energy (in units of cm²/s²
(assuming 10 pixels/cm image resolution) for an object
of unit mass), the impact parameter (0 to 1) and the
momentum asymmetry in the centre-of-mass frame (-1 to 1).
Depending on the "intensity" of the collision, an suitable 
event picture of a proton-proton collision in the LCH
appears on the video screen.

Example
--------

As a demonstration, you may want to run the program on a prepared 
video file, which shows short sequences of collisions of a red
and a green rubber ball. Just execute

  > `python3 ppBilliard.py -v videos/Flummies.webm`

from the command line. The necessary calibration files for
the color detection are included in the package. If you
want to try out the calibration yourself, just add the
"-c 1" option. 
This simple arrangement was realized using a red and a green
rubber ball of approximately 2.5 cm diameter on a black cloth
recorded with a USB-camera with a resolution of 800x600 pixels
at 30 frames/s. 


Practical tips for building the game board
------------------------------------------

The best way to build a compact game board is to use black fabric or 
non-reflective black cardboard of about 40cm x 80cm in size. Ideally, 
you should use a webcam with a wide-angle lens so that the full image of 
the board is board is possible at a camera distance of 75cm - 100cm. 
The optimisation of the camera settings, especially brightness, contrast, 
colour saturation or hue, can be done well with the program
[webcamoid](https://webcamoid.github.io/) available on all platforms. 
This program can also be used to determine the parameter ranges that apply
to the ranges of possible values, which are should be entered in the 
configuration file (see below) under the heading "# web cam parameters" 
in order to be able to adjust camera settings in the in the monitoring 
window of *ppBilliard*. The display of the monitoring window is switched 
on by specifying the line `showMonitoring: true` in the configuration file. 

Coloured rubber balls of 2-3 cm in diameter are recommended as colliding 
objects. The surface should be matt, i.e. non-reflective, in order 
to obtain a stable image of the rolling balls. A colour calibration 
is necessary so that the balls can be recognised and tracked 
by the programme. The procedure for colour calibration is described below. 

When you start the programme and move the balls, a track of the coordinates 
should should appear on the video screen. Make sure that the marks are evenly
spaced, i.e. that no points are missing! Make sure that the colour tones of the
two balls are clearly distinguishable. If necessary, adjust the brightness of
the camera and the lighting conditions. Repeat the calibration if necessary.

To be sensitive to the objects only in the central area, the parameters
`fxROI` and `fyROI` can be adjusted. The option `motionDetection: true` 
enables to be sensitive only to objects that have changed position between 
two video frames. These settings are useful for excluding static background 
objects or activities in the border area. 

The camera should be aligned relative to the board in such a way that 
the board is completely within the camera's field of view and the centre
of the board is in the centre of the video window. For alignment it is 
helpful to put coloured marks on the edges and in the centre. 

Now the set-up is ready for the detection of collisions of the two balls.
To do this, two playing partners should try to bump the balls in a 
coordinated way, so that they meet in the middle with as much speed as 
possible. If a collision is detected, a quantity proportional to the energy 
in the centre-of-gravity system and the impact parameter are calculated. 
A score for the two players is calculated from the product of these two 
values. Depending on the score an image of a real proton-proton collision 
is superimposed in the video window. For this demonstration, on-line
event picutes recorded during the eraly data-taking in 2022 by the
[CMS Detector](https://cms.cern) at CERN are used as an example. 

The directories of the image files, from which one is randomly selected 
in each case, as well as the corresponding score values, are listed in 
the configuration file under the heading `# directories with event pictures`. 
By changing entries or the replacement of event pictures, adjustments and
adaptations are possible.


#### Color Calibration 

Start *ppBilliard.py*  with the option '-c1' to set the parameters 
for the first object in 'hsv color space' (hue, saturation, value). 
The procedure is interactive in a graphical window - adjust the
silders for minimum and maximum hsv values such that only the
object 1 is clearly visible in the right video window. Type 's'
in the video window to store the parameters for object 1. Repeat
the same procedure for the second object with parameter '-c 2';
note that the second object must must have color different from
object 1! The data generated in this way are stored in the files
`object1_hsv.yml` and `object2_hsv.yml`. If they are not present
or were deleted, the default settings from the configuration file 
are used.

#### Configuration 

The configuration of *ppBilliard* can be adapted very flexibly via 
a configuration file.  Here is the content of the supplied file 
*ppBconfig.yml*:

```
## configuration parameters for *p(roton)p(roton)Billiard*
#  -------------------------------------------------------

# web cam parameters (acutal values to be used depend on camera model)
                      # null means take system defaults
camWidth:  1280       # desired picture width  (null)
camHeight:  720       # desired picture height (null)
camFPS:      30       # frame rate             (null)
camExposure: null     # exposure time in ms    (null)
camSaturation: null   # color saturation, depends on camera model (null)
# ranges of camera parameters (e.g. from ouput of v4l2-ctl --all)
#   used for trackbars if showMonitoring set to true
#  values for ELP USBFHD01M-SFV wide-angle camera with OV2710 Chip
rangeBrightness: [-64, 64] # brightness ([0, 100])
rangeContrast: [0, 64]     # contrast   ([0, 100])
rangeSaturation: [0, 128]  # saturation ([0, 100])
rangeHue: [-40, 40]        # hue        ([0, 100])
rangeGamma: [72, 500]      # Gamma      ([0, 100])

# properties of trackable objects
objRmin:  10       # minimum radius in pixels (10)
objRmax: 100       # maximum radius in pixels (100)
#                      settings here are for example video "Flummies.webm"
obj_col1 : [[23, 0, 50],[51, 255, 255]]  # hsv color range for green object
obj_col2: [[0, 50, 50], [22, 255, 255]]  # hsv color range for red object

# collision detection
fRapproach: 3.0    # scaling of object size close approach (3.0)
fRcollision: 1.75  # scaling of object size for collion detection region (1.5)
fTarget:     0.11  # required target region around centre of image (0.11)
fxROI:  0.7        # tracking region x as fraction of i mage width  (1.0)
fyROI:  1.         # tracking region y as fraction of image height (1.0)

# score rankings (score = CMS energy * impact distance)
superScore: 10000 #
highScore: 1000
lowScore: 500
# directories with event pictures
dir_events: 'events/'   # name of top-directory for event pictures
dirs_noScore:    ['empty/']
dirs_lowScore:   ['Cosmics/']
dirs_highScore:  ['2e/', '2mu/', 'general/']
dirs_superScore: ['2mu2e/', '4mu/']

# video processing
maxVideoWidth: 1000   # maximum width of video frames (1024)
pixels_per_cm:   10   # video pixels per cm (10)

# running options 
playIntro: true        # show trailer if true (true)
motionDetection: true  # differentiate frames to detect ball movements (false)
showMonitoring: true   # show screen with detected objects (false)
verbose: 0             # print detailed information if>0  (0)
eventTimeout: 10       # timeout for event picture display (-1)

# images for intro and background (in subdirectory images/)
introImage: 'ppBilliard_intro.png'
bkgImage: 'RhoZ_black.png'

# video files
defaultVideoFPS: 15     # frame rate for replay of video files  (15)
``` 


Further ideas
-------------

This is just an initial version intended as a demonstrator. 

The "objects" in a final version should be real objects on a 
play field which are kicked by two players. The play field
might be a simple board for a mobile version, or a true play 
ground with (colored) footballs in a sports field - although
placing the camera in the latter scenario may be a challenge. 

The selection of collision images could be more sophisticated
and better didactically motivated. 
As a result, a true competition of teams for the largest harvest
of interesting results and discussions on the physics behind the
event pictures could arise.

The present selection of collision events from the CMS experiment 
can easily be replaced by collections of images from other sources.

**Please contribute!**
