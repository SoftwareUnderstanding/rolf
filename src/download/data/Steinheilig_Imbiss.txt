# Imbiss

Das Programm **imbiss.py** ist eine 1:1 Implementation (der Spielmechanik) von <br>
"Imbiss-Bude" von F. Brall 1983 für Apple II, Homecomputer 8/1983, S. 37-41<br>
"Imbiss" von O. Schwald 1984 für Commodore C64, Homecomputer 10/1984, S. 55-58 <br>

Darüber hinaus reproduziert das Programm die Spielmechanik von <br>
"Imbiss" von T. Bauer für IBM PC, Public Domain Version v. 5.4<br>

<img src="./images/Python_Version_Screenshot_v02.JPG" width="60%">

### ToDo
* Prüfen Eingabe 
* Überspringen, wenn keine Eingabe
* Code refaktorisieren

## Spielgeschichte
F. Brall    1983,   Apple II,   "Imbiss-Bude"<br>
O. Schwald  1984,   C64,        "Imbiss"<br>
T. Bauer    1984,   VZ200,      "Imbiss"        *nicht veröffentlicht*<br>
T. Bauer    1986,   CPC664,     "Imbiss"        *nicht veröffentlicht*<br>
O. Schwald  1987,   Amiga,      "Imbiss III"<br>
T. Bauer    1991,   PC,         "Imbiss"<br>
A. Kern     1992,   Amiga,      "Tom's Imbiss"<br>
A. Kern     2009,   PC,         "Tom's Imbiss"<br>
A. Kern     2011,   Crossplat*  "Tom's Imbiss"<br>
L. Brämer   2019,   Amiga,      "Tom's Imbiss"<br>
T. Bauer    2021,   Android,    "Imbiss Classic"<br>
Steinheilig 2021,   PC,         "Imbiss"<br>
  
*Crossplatform:  AmigaOS 3 (m68k) AmigaOS 4 (ppc) Android (arm) AROS (x86)
iOS (arm) Linux (x86) Linux (x64)
Linux (arm) Linux (ppc) macOS (x86) macOS (x64) macOS (ppc)
MorphOS (ppc) WarpOS (m68k/ppc) Windows (x86) Windows (x64)

## Bugs: Versionen 1983/1984/1991
* "Kaufen koste es was es wolle" in den Spielversionen von 1983/1984:<br>
Der maximale Preis der verschiedenen Eissorten wird nicht geprüft, dadurch kann in 10% der Eiskauffälle eine Eissorte mit beliebig hohen Verkaufspreis verkauft werden.
* "Cola Bug" in den Spielversionen von 1983/1984:<br>
Kein Einfluss des Colapreises auf Kundenverhalten. Colakunden sind zufällige Eiskunden UND keine Prüfung von Colapreis!
* "Bratwurst Anomalie" in der Spielversion von 1991:<br>
Wird der Verkaufspreis einer Ware aus einer Warengruppe zu hoch angesetzt (z.B. die Cola oder die Bratwurst), 
werden die anderen Elemente der Warengruppe nicht mehr nachgefragt (da es keine Kunden gibt). 
Also können z.B. alle Eissorten beliebig billig angeboten werden, sobald die Cola mehr als X(T,Wochentag) kostet, wird kein Eis mehr verkauft. 
Das Gleiche gilt auch für die Bratwurst, wenn diese zu teuer angeboten wird, werden keine Pommes mehr verkauft.     

## Exploits: Versionen 1983/1984
Exploits der Spielmechanik für die Spielversionen von 1983/1984:<br>
1) 10% der Fälle, keine Prüfung von Eispreis => Ein Eis super billig -> EK  (nur abhängig von kleinstem Eispreis (1440))  // Anderen Eissorten max Preis wird bezahlt<br>
2) 10% der Fälle, keine Prüfung von Bratwurst-/Pommespreis=> Eins super billig -> BK  (nur abhängig von kleinstem Preis (1440))  // Andere max Preis wird bezahlt<br>
3) Kein Einfluss des Colapreises auf Kunden (Colakunden sind zufällige Eiskunden) / keine Prüfung von Colapreis => Jeder Cola Preis wird bezahlt in ca. für EK/4 % Der Fälle<br>

## Analyse der Spielmechanik "Imbiss" für PC von 1991
Auf Basis des C64 LISTINGs, kann die Spielmechanik ihrer PC Version von 1991 bestimmt werden.
Im Prinzip ist diese sehr nah am Orginal von 1983/4 (bis auf leichte Parametervariation, z.B. Temperaturverhalten und Offset der Kundenfunktionen).
Um den Fehler in der C64 Version zu korrigieren (hier wurde das Minimum des Preises einer Warengruppe zum Bestimmen der Kundenzahl herangezogen, in Kombination mit einem Zufallsziehen aus der Warengruppe konnten Waren mit beliebig hohen Preisen verkauft werden) wird dar maximale Preis der Elemente der Warengruppe benutzt. Allerdings entsteht hierdurch die "Cola bzw. Bratwurst Anomalie". 
Die Paramter wurden durch Auswerten des Kundenverhaltens für verschiedene Temperatur, Wochentage, Verkaufspreise bestimmt. Der Quellcode lag leider nicht vor. In der Datei **imbiss_analyse.py** wird die Analyse inklusive der manuell ermittelten Daten dargelegt.

## Optimale Verkaufsstrategie - "Imbiss" für PC von 1991
Auf Basis der Spielmechanik können optimale Verkaufsstrategien, d.h. den Gewinn maximierende Verkaufspreise der Waren als Funktion der Temperatur und des Wochentages abgeleitet werden. Vergl. **imbiss_analyse.py**.
<!---
![plot](./images/OptimaleBratwurstStrategie_WE.jpg))
-->
<img src="./images/OptimaleBratwurstStrategie_WE.jpg" width="70%">

## Einsatz von maschinellem Lernen, um optimale Verkaufsstrategie zu ermitteln
Mit Hilfe von Methoden des maschinellen Lernens können optimale Verkaufsstrategien, d.h. den Gewinn maximierende Verkaufspreise der Waren als Funktion der Temperatur und des Wochentages, ermittelt werden. In **imbiss_ML.py** bzw **imbiss_RL_gym.py** (+**imbiss_env.py**) werden verschiedenen Ansätze illustriert:
1. Eine Möglichkeit ist es, ein Modell des zu erwartenden Gewinns zu trainieren. Als Grundlage dienen Input/Output Wertepaare, sogenannte samples. Mit einer großen Menge von solchen Wertepaaren kann ein Modell, bzw. ein Funktionsapproximation, zwischen Input und Output trainiert werden. Im Anschluss kann ein Optimierer verwendet werden, um die besten (Gewinn maximierenden) Verkaufspreise auf Basis des Gewinnmodells zu bestimmen. <br> Im folgenden Beispiel werden GradientBoosted Decision Trees (n=300, may_depth=10) mit 1e6 Samples trainiert und grid search verwendet, um die optimalen Verkaufspreise zu bestimmen: <br> <img src="./images/OptimaleBratwurstStrategie_DT_1e6_10_300.jpg" width="50%">
2. Eine weitere Möglichkeit ist Methoden des sogenannten verstärkenden Lernens (Reinforcement learning, RL) zu verwenden. Hierbei werden Wertefunktionen (Value Functions) für die möglichen Aktionen (Verkaufspreise) und Systemzustände (Temperatur, Wochentage) ermittelt. Als Wertefunktion wird der zu erwartenden Gewinn, gegeben der Verkaufspreise (Aktion) und Temperatur/Wochentag (Zustand des Systems), verwendet. Auf Basis der Wertefunktion, gegeben des Systemzustandes, wählt ein sogenannter Handelnder (RL-Agent) Aktionen/Verkaufspreise nach einem vorgegeben Regelwerk (Policy) aus. Die Policy muss hierbei das Ausnutzen des in der Wertefunktion repräsentierten Wissens (Exploitation) mit einem Ausprobieren alternativer Lösungsmöglichkeiten (Exploration) ausbalancieren. <br> Im folgenden Beispiel wird ein constant-alpha Monte Carlo (MC) Ansatz mit epsilon-greedy (epsilon=0.5, alpha=0.2) Policy und Tabellen (tabular solution method) gewählt. Aufgrund der großen Anzahl von möglichen Aktionen (Verkaufspreisen), wird die Wertefunktion dynamisch nur für die besten 200 Aktionen im Speicher behalten und die Preise in 5Pf Schritten qunatisiert & variiert. <br> <img src="./images/OptimaleBratwurstStrategie_RL_quant_5_200_alpha_2_epsilon_5_2ndVersion_40kepochs.jpg" width="50%">
3. Die Kombination der beiden ersten Ansätze, Modellapproximation des erwarteten Gewinns sowie der Einsatz von Reinforcement Learning, wobei ein tiefes neuronales Netzwerk zur Modellapproximation (genauer der aus dem Modell resultierenden genährten Wertefunktionen) und ggf. der Policy Repräsentation, verwendet wird, wird als Deep Reinforcement Learning (DeepRL) bezeichnet.<br>Beispielsweise sind Deep Q-Learning Netzwerke (DQN) bei diskreten Aktions- und Zustandswerten anwendbar. Des weiteren können kontinuiertliche Aktions- und Zustandswerten (mittels approximation solution methods der Wertefunktion) verwendet werden. Beispielhaft sei hier der Soft Actor-Critic (SAC, [10]) Algorithmus erwähnt. Ein Beispiellösung eines DeepRL Ansatz mit kontinuiertliche Aktionswerten mittels SAC ist in **imbiss_RL_gym.py** implementiert. Hierzu wird zunächst in **imbiss_env.py** eine Umgebung im OpenAI Gym definiert und danach die SAC Implementation aus der Stable-Baselines Python Bibliothek verwendet (Zeitschritte=2e5, gamma=1e-2, lr=5e-4). <br> <img src="./images/OptimaleBratwurstStrategie_SAC_g001_lr0005_t2e5.jpg" width="50%">

## Danksagung 
Vielen Dank an **Oliver Schwald**, der mit die Listings von F. Brall "Imbiss-Bude" aus dem Homecomputer 8/1983 sowie seiner C64 Implementation aus dem Homecomputer 10/1984 eingescannt und geschickt hat! Vielen Dank an **Thomas Bauer**, für die vielen interessante Information und Anekdoten zu seiner PC Version von 1991 und dessen Entstehung. 

## Literatur & Links 
[1] Franz Ablinger (2016), Homecomputer - Zur Technik- und frühen Computerspielkultur anhand einer Zeitschrift der Jahre 1983 und 1984, Doktorarbeit
https://docplayer.org/44038258-Homecomputer-zur-technik-und-fruehen-computerspielkultur-anhand-einer-zeitschrift-der-jahre-1983-und-1984.html<br>
[2] "Imbiss", PC Version von 1991, http://www.gameseller.de/mein-erstes-pc-spiel-imbiss-manager-testet-es-selbst/#more-250<br>
[3] DOS Emulator, DosBox, https://www.dosbox.com/<br>
[4] "Imbiss", C64 Version von 1984, http://www.germanc64.de/ggames.php?c=i&s=1<br>
[5] C64 Emulator, CCS64 3.9.2, https://www.heise.de/download/product/ccs64/download <br>
[6] "Imbiss III", Amiga Version von 1987, https://archive.org/details/Imbiss_III_1987_Schwald_O._de_h_Pawlowski_o2 <br>
[7] "Tom's Imbiss", Amiga OS1.3 Version von 1994/2021 https://www.kehosoft.de/imbiss_hollywood.html <br>
[8] "Imbiss Classic", Android von 2021, https://play.google.com/store/apps/details?id=de.minibits.imbiss_classic&hl=de&gl=US <br>
[9] Sutton and Barto (2018), "Reinforcement Learning, 2nd edition", MIT Press <br>
[10] Haarnoja et al. (2018), "Soft Actor Critic (SAC) Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", https://arxiv.org/abs/1801.01290 <br>

## Gameplay Videos
* "Imbiss III", Amiga Version von 1987 https://youtu.be/qTg-1BV26RM<br>
* "Imbiss III", Amiga Version von 1987 https://youtu.be/Xo-Z71IkOe4?t=136<br>
* Tom's Imbiss 1.0, Amiga 1992, https://youtu.be/7lVctKjUY9g<br>
* Tom's Imbiss 1.0, Amiga 1992, Teil 2, https://youtu.be/4-wFfIBef5s<br>
* Tom's Imbiss Imbiss v3.1, Amiga 2012, https://youtu.be/Eho4QnMx3Eg<br>


