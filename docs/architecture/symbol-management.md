## Az instrumentumok kezelése a progamban

### Az instrumentumok (Symbol) letöltése:
Az instrumentumokat a MT5 rendszerből töltjük le. 
Egy lenyíló dropdown mezőben megjelenítjük az összes nevet. Továbbá legyen egy hozzáadás és egy elvétel gomb is. Illetve legyen egy sorrend et állító fel-le gomb, és egy információ nyomógomb is, mely megjeleníti az instrumentum paramétereit (lásd lejjebb.)
Legyen egy fő lista, ami tartalmazza az összes olyan instrumentumot, amit a korábbi hozzáadás gombbal hozzáadtunk a rendszerhez.
- Amiket kiválasztottunk instrumentumokat, azoknak mutatja folyamatosan a tick értékeit (Bid, ask)
Az alábbi táblázat mutatja, hogy milyen adatokat szeretnék megjeleníteni:

| Symbol | Bid | Ask | Spread | Aktív kereskedés |M1 | M3 |M5| M15| H1 |  
|--------|-----|-----|--------|------------------|----|----|--|----|----|
| EURUSD | 1.17682| 1.17685 | 12 | 0 | ⬆️| ⬆️ |⬇️ | ⬇️ |⬆️

#### Magyarázat a táblázathoz:

#### Sorok színeinek jelentése
Az egyes sorok (Symbol, Bid, Ask, Spread, Aktiv kereskedés ) (fontos, hogy a többinek saját színkódja van!) az alábbi színkóddal jelenhetnek meg.
- Piros: Ha egy symbol túl nagy spreaddet használ akkor azt pirossal jelzi. (Az hogy mit tekintünk nagy spreadnek, azt szintén a beállítások menüben lehet meghatározni.)
- Zöld: A symbol kereskedhető. 

#### Aktív kereskedés 
Megmutatja, hogy éppen az adott instrumentumon hány db pozíció van nyitva.

#### Idősíkok (M1, M3, M5, M15 ...)
A megjelenített idősíkok ugyanazok amik a központi setupban be vannak állítva. Tehát ha ott be van állítva az M1, M2, M15, és D1 akkor a táblázatban is ezek jelennek meg.

#### Nyilak jelentése az idősíkokban
 A nyilak azt mutatják meg, hogy az adott idősíkban milyen irányban áll a trend.

##### Trend meghatározása
A trend meghatározása A MACD indikátor fogja meghatározni. A Setupban be lehet állítani, hogy hanyas periodust figyelje, ezt alapértelmezetten a 120-ra lesz állítva. Későbbiekben lehetőséget kell biztosítani, hogy más indikátorral is meghatározható legyen a trend iránya.

###### Középérték számolása: 
~~~
((Nyitó + Záró / 2) + (High + Low /2) /2 ) 
~~~

##### Trend irány számítás MACD-vel
Fontos hogy három irányba gondolkozzunk (Emelkedő, csökkenő, oldalazó!)

- Emelkedő trend (felfelé nyil zöld): 
    - vagy A középárfolyam a MACD felett van.
    - vagy MACD - X bar < MACD 
- Csökkenő trend (lefelé nyíl piros):
    - vagy A középárfolyam a MACD alatt van.
    - vagy MACD - X bar > MACD 
- Oldalazó trend (Jobb irányba álló nyíl sárga):
    - vagy A középárfolyam a MACD-t többször keresztezi adott gyertyán belül
    - vagy MACD - X bar = MACD  +- tűréshatár

#### Symbolumok mentése adatbázisba (symbol_info)
Az MT5 kapcsolódás után a symbolumok listáját, és az MT5 általa megjeleníthetó paramétereit letöltjük a memóriába, egyúttal az SQLLite adatbázisba is. lásd: database-architecture.md / symbol_info
- Másodszori bekapcsoláskor a symbolumok frissítése elengedhetetlen.

### Kiválasztott instrumentumok mentése és betöltése adatbázisba
Minden beállítást, amit a user eszközöl a program automatikusan menti, illetve a kövekező indulásnál automatikusan visszatölti. Ilyen a kiválasztott instrumentumok listája is. Ezt is elmenti az adatbázisba, és az újabb belépéskor automatikusan betölti. (Fontos, a sorrend is!)

### Információs panel. (információs nyomógomb hatása)
Az kiválasztott instrumentum után, van egy információ- nyomógomb, megnyomása esetén egy felugró ablakban megjelenik minden olyan paraméter az adott instrumentumról, amit az MT5-ből ki lehet halászni.
    