# Tick és OHLC adatok lekérdezésének metódikája.

## Program indulásakor a történelmi adatok lekérdezése a háttérben

-  Az első belépéskor, ha még nem talált adatbázist, vagy adatot akkor egy előre meghatározott dátumig (Ezt szintén a setupban lehet állítani.) az összes tick-et és ohlc adatot letölti a Metatrade5 rendszerből.
- Belépéskor ha már talált adatot akkor lekérdezi az utolsó dátumot, visszamenőlegesen letölti az utolsó dátum és jelenlegi dátum közötti hiányzó tickeket- ohlc- adatokat.

## Program futása közben

- A program futása közben háttérben történik egy tick és ohlc adat lekérdezés a Metatrade 5-ben. 
    - Tick adatok esetében: 0.2 mp (Setupban állítható) 
    - OHLC adatok esetében: 0.6 mp (Setupban állítható)

## Új instrumentum kiválasztása esetén:

A múltbéli adatokat abban az esetben is le kell tölteni, amikor új instrumentumot választok ki.
  - A Setupban meghatározott kezdeti dátumot lekérdezzük
    - Letölteni az összes kiválaszott idősíkra az új instrumentum OHLC adatát
    - Letölteni az új instrumentum kezdeti dátumtól keletkezett tick adatokat.

## Új idősík kiválasztása esetén:

Ha kiválasztok a beállításokban egy új idősíkot, akkor a felsorolásban található összes instrumentumra az setupban meghatározott idő alapján létrehoni az új OHLC adatot az adott idősíkra.

## Kezdeti időszak változtatása esetén:

Ha megváltoztatjuk a kezdeti dátumot, akkor az új és a régi dátum közötti összes kiválasztott instrumentum:
 - összes kiválasztott idősíkjában az ohlc adatok letöltése
 - Összes tick adatok letöltése.

 ## Záradék:

 Ha a setupban a dátum kiválasztásánál egy késöbbi dátumot választunk ki, abban az esetben **nem kell visszatörölni az adott dátumig** az OHLC, TICK adatokat
 pl: Eredeti dátum: a 2025-05-01
 - Ebben az esetben minden tick mind az ohlc adat innen kezdődik
 - Ha kiválasztom hogy : 2025-06-01 legyen az új dátumom, ebben az esetben nem kell visszatörölnie a 2025-05-01- és a 2025-06-01 közötti értékeket.
 - Ebben az esetben figyelmeztető ablakot megjeleníteni!
 


