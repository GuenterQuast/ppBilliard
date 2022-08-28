****************************************
## Projekt **P(roton)P(roton) Billiard**
****************************************

Übersicht
---------


Lass zwei Kugeln auf einem Tisch zusammenprallen und sieh, was passiert,
wenn sie Protonen im Large Hadron Collider am CERN wären!

> ![](ppBilliard.png)

Dieses Projekt verwendet ein Tischspielbrett für die Videoaufnahme
kollidierender runder farbiger Objekte mit einer Webkamera.
Kollisionsparameter wie die äquivalente Schwerpunktsenergie,
Aufpralldistanz und Asymmetrie werden automatisch ermittelt.
Diese werden so skaliert, dass sie den Parametern einer Proton-Proton 
Proton-Proton-Kollision in einem Hochenergie-Collider entsprechen und
Bilder entsprechender Spuren der Teilchenkollisionen werden gezeigt. 

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
```

Normalerweise ist die Nummer des Videogeräts für die Webcam 0; falls 
nicht, verwenden Sie den Parameter '-s n' mit der Gerätenummer 'n'.


### Umgang mit dem Programm *ppBilliard.py*

Wenn Sie das Programm ohne andere Argumente als die Gerätenummer starten 
wird ein kurzer Trailer angezeigt und dann auf die Anzeige der Webcam-
Ausgabe umgeschaltet. Wenn ein verfolgbares Objekt wie ein farbiger Ball 
oder Kreis identifiziert wird, wird es durch eine symbolische, animierte Darstellung eines Protons ersetzt und im Video verfolgt, während Sie es herumbewegen.

Wenn kein Objekt erkannt wird, ist ein Kalibrierungsschritt erforderlich.
Schließen Sie das Programm durch Eingabe von `<esc>` oder 'q' im Videofenster
und starten Sie es erneut mit dem Parameter '-c 1', um die Farbparameter für
das erste Objekt im "*hsv*-Farbraum" (Farbton, Sättigung, Wert) einzustellen. 
Das Verfahren funktioniert interaktiv in einem grafischen Fenster -
stellen Sie die minimalen und maximalen *hsv*-Werte so ein, dass nur das
Objekt 1 im rechten Videofenster deutlich sichtbar ist. Tippen Sie 's'
in das Videofenster ein, um die Parameter für Objekt 1 zu speichern.
Wiederholen Sie das gleiche Verfahren für das zweite Objekt mit dem
Parameter "-c 2". Beachten Sie, dass das zweite Objekt eine andere Farbe
haben muss als Objekt 1!

Starten Sie das Programm neu und verschieben Sie die Objekte.  Die Objekte 
sollten nun erkannt werden und ihre runden Formen durch symbolische,
animierte Darstellungen von Protonen ersetzt werden.
Wenn sich die Spuren der Objekte nahe genug kommen und ihre extrapolierten
Spuren näher als die Summe der Radien beieinander liegen, wird eine 
"Kollision" erkannt und die Kollisionsparameter werden auf dem Bildschirm
angezeigt:  
die äquivalente Schwerpunktsenergie (in Einheiten von cm²/s²
(unter der Annahme einer Bildauflösung von 10 Pixeln/cm) für ein Objekt
einer Masseneinheit), der Stoßparameter (0 bis 1) und die
Impulsasymmetrie im Massenschwerpunkt (-1 bis 1).

Abhängig von der "Intensität" der Kollision wird dann ein Ereignisbild
aus dem [CMS-Detektor] (https://cms.cern) am CERN auf dem
Video-Bildschirm  angezeigt. 

Beispiel
--------
Zur Demonstration können Sie das Programm auf eine vorbereitete Videodatei
anwenden, die kurze Sequenzen von Zusammenstößen zwischen einem roten und
einem grünen Gummiball zeigt. Führen Sie einfach
```
     python3 ppBilliard.py -v videos/Flummies.webm
```
auf der Kommandozeile aus.
Die notwendigen Kalibrierungsdateien für die Farberkennung sind im
Paket enthalten. Wenn Sie die Kalibrierung selbst ausprobieren möchten,
fügen Sie die Option "-c 1" hinzu. 


### Weitere Ideen

Dies ist nur eine erste Version, die als Demonstrator gedacht ist. 

Die "Objekte" in einer endgültigen Version sollten bunte Bälle oder Kugeln
auf einem interessanten Spielfeld sein, die von zwei Spielern gekickt
werden. Das Spielfeld könnte ein einfaches Spielbrett für eine mobile Version
sein, oder auch ein echtes Spielfeld mit (farbigen) Fußbällen auf einem
Sportplatz - allerdings dürfte die Platzierung der Kamera in letzterem
Szenario eine Herausforderung sein. 

Die Auswahl der Kollisionsbilder könnte deutlich ausgefeilter sein.
So könnte ein echter Wettbewerb der Teams um die größte Ausbeute
von interessanten Ereignissen entstehen. Die derzeitige Auswahl von Kollisionsereignissen aus dem CMS-Experiment kann leicht durch 
Sammlungen von Bildern aus anderen Quellen ersetzt werden.

**Bitte tragen Sie bei!**
