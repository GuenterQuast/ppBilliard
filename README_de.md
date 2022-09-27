****************************************
## Projekt **P(roton)P(roton) Billiard**
****************************************

Übersicht
---------


Lass zwei Kugeln auf einem Tisch zusammenprallen und sieh, was passiert,
wenn sie 
  >  >  >  **Protonen im Large Hadron Collider am CERN** wären!

> ![](ppBilliard.png)

Dieses Projekt verwendet ein Tischspielbrett für die Videoaufnahme
kollidierender farbiger Objekte mit einer Webkamera.
Kollisionsparameter wie die äquivalente Schwerpunktsenergie,
Aufpralldistanz und Asymmetrie werden automatisch ermittelt.
Diese werden so skaliert, dass sie den Parametern einer
Proton-Proton-Kollision in einem Hochenergie-Collider entsprechen 
und Bilder entsprechender Spuren von Teilchenkollisionen werden 
zufällig ausgewählt und gezeigt. 

Dieser Python-Code stützt sich stark auf die Open Source Computer
Vision Library [OpenCV] (https://opencv.org/).  
Vielen Dank an die Entwickler für die großartige Arbeit!

Erstellt von Guenter Quast, erste Version Aug. 2022 

### Installation:

Auf einer Standard-Python-Umgebung (>3.6) wird die OpenCV-Bibliothek 
über den Befehl  

> `pip3 install opencv-python` installiert.

Nachdem Sie das *ppBilliard*-Paket heruntergeladen haben, geben Sie

```
  python3 ppBilliard.py -h
```

auf der Kommandozeile ein, was die folgende Ausgabe erzeugt:

```
  usage: ppBilliard.py [-h] [-f] [-v VIDEO] [-b BUFFER] [-s SOURCE]
                       [-c CALIBRATE]

  optionale Argumente:
    -h, --help         diese Hilfemeldung anzeigen und beenden
    -f, --fullscreen   Ausführen im Vollbildmodus
    -v VIDEO, --video VIDEO
                       Pfad zur einer (optionalen) Videodatei
    -b BUFFER, --Puffer BUFFER
                       maximale Puffergröße für Objektspuren
    -s SOURCE, --Quelle SOURCE
                       Nummer des Videogeräts
    -c CALIBRATE, --kalibrieren CALIBRATE
                       hsv-Bereich des verfolgbaren Objekts finden <1/2>
    -C CONFIG, --config CONFIG     
                       Konfigurationsdatei 
```

Normalerweise ist die Nummer des Videogeräts für die Webcam 0; falls 
nicht, verwenden Sie den Parameter '-s n' mit der Gerätenummer 'n'.


### Umgang mit dem Programm *ppBilliard.py*

Wenn Sie das Programm ohne andere Argumente als die Gerätenummer starten 
wird ein kurzer Trailer angezeigt und dann auf die Anzeige der Webcam-
Ausgabe umgeschaltet. Wenn ein verfolgbares Objekt wie ein farbiger Ball 
oder Kreis identifiziert wird, wird es durch eine symbolische, animierte 
Darstellung eines Protons ersetzt und im Video verfolgt, während Sie es
herumbewegen.

Wenn kein Objekt erkannt wird, ist ein Kalibrierungsschritt erforderlich, 
wie weiter unten beschrieben. 

Starten Sie das Programm nach Durchführung der Farbkalibration neu und 
verschieben Sie die Objekte. Die Objekte sollten nun erkannt werden und 
ihre runden Formen durch symbolische animierte Darstellungen von Protonen 
ersetzt werden. Wenn sich die Objekte nahe genug kommen und ihre 
extrapolierten Spuren näher als die Summe der Radien beieinander liegen, wird 
eine  "Kollision" erkannt und die Kollisionsparameter werden auf dem Bildschirm
angezeigt:  
die äquivalente Schwerpunktsenergie (in Einheiten von cm²/s²
(unter der Annahme einer Bildauflösung von 10 Pixeln/cm) für ein Objekt
einer Masseneinheit), der Stoßparameter (0 bis 1) und die
Impulsasymmetrie im Massenschwerpunkt (-1 bis 1). Als Maß für die
"Intensität" der Kollision wird im Video-Fenster ein Punkte-Score für
die Kollision ausgegeben, mit dem zufällig ein entsprechendes
Ereignisbild aus dem [CMS-Detektor] (https://cms.cern) am CERN auf dem
Video-Bildschirm  angezeigt wird. 


Beispiel
--------
Zur Demonstration können Sie das Programm auf eine vorbereitete Videodatei
anwenden, die kurze Sequenzen von Zusammenstößen zwischen einem roten und
einem grünen Flummi zeigt. Führen Sie einfach
  > `python3 ppBilliard.py -v videos/Flummies.webm`

auf der Kommandozeile aus.´
Die notwendigen Kalibrierungsdaten für die Farberkennung in diesem
Beispiel sind im Paket vorkonfiguriert. 
Wenn Sie die Kalibrierung selbst ausprobieren möchten,
fügen Sie die Option "-c 1" hinzu.  
Dieses einfache Arrangement wurde mit einem roten und einem grünen
Flummi von etwa 2,5 cm Durchmesser auf einem schwarzen Tuch
mit einer USB-Kamera bei einer Auflösung von 800x600 Pixeln mit
30 Bildern/s aufgenommen. 


Praktische Hinweise zum Aufbau
------------------------------

Ein kompakter Aufbau eines Spielbretts gelingt am besten, wenn man 
schwarzen Stoff oder nicht-reflektierende schwarze Pappe von etwa
40cm x 80cm Größe verwendet. Idealerweise verwendet man eine WebCam
mit Weitwinkel-Optik, damit die volle Abbildung des Spielbretts
bei einem Kamerabstand von 75cm - 100cm möglich wird. Die Optimierung 
der Kameraeinstellungen, insbesondere Helligkeit, Kontrast, Farbsättigung 
oder Farbton, gelingt gut mit dem auf allen Plattformen verfügbaren
Programm [webcamoid](https://webcamoid.github.io/). Mit diesem Programm
kann man auch die für die einzelnen Werte geltenden Parameterbereiche
ermitteln, die in der Konfigurationsdatei (s. unten unter der Überschrift
"# web cam parameters") eingetragen werden sollten, um Kameraeinstellungen 
auch im Monitoring-Fenster von *ppBilliard* vornehmen zu können. 
Die Anzeige des Monitoring-Fensters wird eingeschaltet durch Angabe
der Zeile `showMonitoring: true` in der Konfigurationsdatei. 

Als kollidierende Objekte empfehlen sich farbige Gummibälle ("Flummies")
von 2-3 cm Durchmesser. Die Farbe sollte möglichst matt, also nicht-reflektierend
sein, um ein stabiles Bild auch der rollenden Kugeln zu erhalten. Nun ist noch
eine Farbkalibration notwendig, damit die Bälle vom Programm erkannt und 
verfolgt werden können. Die Vorgehensweise zur Farbkalibration ist weiter
unten beschrieben. 

Wenn Sie nun das Programm starten und die Bälle bewegen, sollte eine 
Spur der Koordinaten im Videofenseter erscheinen. Achten Sie darauf, dass
die Eingezeichneten Markierungen gleichmäßig sind, also keine Punkte fehlen!
Achten Sie darauf, dass die Farbtöne der beiden Bälle klar unterscheidbar sind.
Passen Sie ggf. die Helligkeit der Kamera und die Lichtverhältnisse an. 
Wiederholen Sie ggf. die Kalibration.

Um nur im Zentralbereich auf die Objekte empfindlich zu sein, können die Parameter
`fxROI` und `fyROI` angepasst werden. Die Option `motionDetection: true` ermöglicht
es, nur auf Objekte empfindlich zu sein, die ihre Position zwischen zwei Videobildern
verändert haben. Diese Einstellungen sind hilfreich, um statische Hintergrundobjekte
oder Aktivitäten im Randbereich auszublenden. 

Nun sollte die Kamera relativ zum Spielbrett so ausgerichtet werden, dass 
sich das Spielbrett vollständig im Aufnahmebereich der Kamera befindet und
die Mitte des Bretts in der Mitte des Videofensters liegt. Zur Ausrichtung
ist es hilfreich, an den Rändern und in der Mitte farbige Markierungen 
anzubringen. 

Nun ist der Aufbau bereit für die Erkennung von Kollisionen der beiden Bälle.
Dazu sollten zwei Spielpartner versuchen, die Bälle koordiniert so anzustoßen,
dass sie sich mit möglichst großer Geschwindigkeit in der Mitte treffen. Wenn
eine Kollision erkannt wird, wird eine zur Energie im Schwerpunktsystem  proportionale
Größe sowie der Stoßparameter berechnet; aus dem Produkt dieser beiden Größen
ergibt sich die Punktzahl ("Score") für die eiden Spieler. Abhängig vom Score
wird ein Bild einer echten Proton-Proton-Kollision im Videofenster überblendet.
Die Verzeichnisse der Bilddateien, aus denen jeweils zufällig eine ausgewählt wird,
sowie die zugehörigen Score-Werte, sind in der Konfigurationsdatei unter der 
Überschrift `# directories with event pictures` eingetragen. Durch Ändern
dieser Einträge bzw. auch den Ersatz von Ereignisbildern sind weitgehende
Anpassungen möglich. 


#### Durchführung der Farbkalibration

Durch Starten des Programms mit der Option `-c1`, also durch Eingabe von  

  > `python3 ppBilliard.py -c1`  

gelangt man in den Kalibrationsmodus, mit dem die um die Farbparameter 
für das erste Objekt im "*hsv*-Farbraum" (Farbton, Sättigung, Wert) 
eingestellt werden. 
Das Verfahren funktioniert interaktiv in einem grafischen Fenster -
stellen Sie die minimalen und maximalen *hsv*-Werte so ein, dass nur das
Objekt 1 im rechten Videofenster deutlich sichtbar ist. Tippen Sie 's'
in das Videofenster ein, um die Parameter für Objekt 1 zu speichern.
Wiederholen Sie das gleiche Verfahren für das zweite Objekt mit dem
Parameter "-c2". Beachten Sie, dass das zweite Objekt eine andere Farbe
haben muss als Objekt 1! Die so erzeugten Daten werden in den Dateien
`object1_hsv.yml` und `object2_hsv.yml` gespeichert. Wenn Sie nicht 
vorhanden sind oder gelöscht wurden, werden die Voreinstellungen aus
der Konfigurationsdatei verwendet. 

#### Konfiguration

Die Konfiguration von *ppBilliard* kann sehr flexibel über eine
Konfigurationsdatei angepasst werden. 
Hier der Inhalt der mitgelieferten Datei *ppBconfig.yml*:

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

Weitere Ideen
-------------

Dies ist nur eine erste Version, die als Demonstrator gedacht ist. 

Die "Objekte" in einer endgültigen Version sollten bunte Bälle oder Kugeln
auf einem interessanten Spielfeld sein, die von zwei Spielern gekickt
werden. Das Spielfeld könnte ein einfaches Spielbrett für eine mobile Version
sein, oder auch ein echtes Spielfeld mit (farbigen) Fußbällen auf einem
Sportplatz - allerdings dürfte die Platzierung der Kamera in letzterem
Szenario eine Herausforderung sein. 

Die Auswahl der Kollisionsbilder könnte deutlich ausgefeilter sein.
So könnte ein echter Wettbewerb der Teams um die größte Ausbeute
von interessanten Ereignissen und eine interessante Diskussion um
deren Bedeutung entstehen. Die derzeitige Auswahl von Kollisionsereignissen 
aus dem CMS-Experiment kann leicht durch Sammlungen von Bildern aus 
anderen Quellen ersetzt werden.

**Bitte tragen Sie bei!**
