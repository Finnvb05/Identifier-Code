Elke file en de commands om hem te gebruiken:

Wordt automatisch gebruikt door andere files --> geen command voor nodig: \
chord_measure.py : Meet de aftsand van pixels op verschillende intervals over het blokje heen. \
gauge.py : main bestand voor meten breedte/dikte/lengte \
gauge-touch.py : bestand voor het displayen van de interface op het touchscreen \
pi_capture.py : helper functies voor measure en dual. \

pi_measure.py : python3 pi_measure.py --thickness X --log file.csv: Meetbestand voor een enkele camera, heeft dikte nodig voor accurate readings, moet geupdate worden zodat het dat niet nodig heeft sinds de afstand standaard wordt (3/9/2026). Geeft een ip die bekeken kan worden, vanuit daar kan ook calibratie worden gedaan. \
pi_dual.py :python3 pi_dual.py --eff_dist0 X --eff_dist1 X --log file.csv :  Zelfde als measure.py, maar dan met twee cameras, heeft de afstand van beide cameras tot het blokje nodig voor eff_dist0\

verouderd en kunnen verwijderd worden: \
pi_live.py \
