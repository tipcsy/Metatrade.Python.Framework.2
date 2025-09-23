# MetaTrader Python Framework 2

## Rövid leírás a projektről
A projekt célja egy olyan Python alapú kereskedő program fejlesztése, amely közvetlen kapcsolatot teremt a MetaTrader 5 kereskedési platformmal. A rendszer képes lekérdezni a brókernél elérhető instrumentumokat, amelyek közül a felhasználó kiválaszthatja a számára releváns instrumentumokat. A kiválasztott instrumentumok tick és ohlc adatait a program eltárolja, majd különböző elemzéseken keresztül kiértékeli.

## Kereskedési stratégiák implementálása

A szoftver többféle kereskedési stratégiát tartalmaz, amelyek teljesítményét instrumentumonként figyeli és összehasonlítja. Amikor valamelyik stratégia belépési jelet generál, a rendszer pozíciónyitási (BUY/SELL) megbízást küld a MetaTrader 5 felé.

A program lekérdezi a számla aktuális adatait is, így képes dinamikusan kiszámítani az egyszerre kockáztatható tőke nagyságát. A megoldás részeként fejlett stop-loss és take-profit kezelés, valamint pozícióépítés is megvalósításra kerül, biztosítva a rugalmas és hatékony kockázatkezelést, továbbá a pozíció bontást is tudja kezelni

A felhasználói élmény növelése érdekében a szoftver egy ablakos grafikus felülettel (GUI) is rendelkezik, amely a WinForms jellegű vezérlést követi. Ezen a felületen keresztül a felhasználó könnyen elérheti a beállításokat, monitorozhatja a futó stratégiákat, nyomon követheti a számla és pozíció adatait, valamint kényelmesen kezelheti a kereskedési folyamatot.
Fejlett Python keretrendszer MetaTrader 5 kapcsolatokhoz és algoritmus kereskedéshez.

## 🚀 Jellemzők

- **Indikátorok**: RSI, MACD, Bollinger Bands, Stochastic és egyebek
- **Candlestick Minták**: Doji, Hammer, Shooting Star, Engulfing, Star minták
- **Automatikus Mentés**: Konfigurációk automatikus mentése és betöltése
- **Lokalizáció**: Magyar és angol nyelv támogatás
- **Real-time Monitoring**: Élő pattern és signal észlelés
- **GUI Interface**: Intuitív grafikus kezelőfelület

## 📁 Projekt Struktúra

```
MetaTrader.Python.Framework/
├── src/                    # Forráskód
│   ├── core/              # Alapvető komponensek
│   ├── indicators/        # Technikai indikátorok
│   ├── patterns/          # Candlestick minták
│   ├── gui/              # Grafikus felület
│   └── database/         # Adatbázis komponensek
├── docs/                  # Dokumentáció
│   ├── architecture/     # Architektúra dokumentumok
│   ├── development/      # Fejlesztési dokumentumok
│   └── reports/          # Projekt jelentések
└── data/                 # Adatok és konfigurációk mentési helye
 
```

## 📚 Dokumentáció

- [Adatbázis Architektúra](docs/architecture/database-architecture.md)
- [Instrumentumok kezelése a programban](docs/architecture/symbol-management.md)
- [Tick és OHLC adatok lekérdezésének metódikája.](docs/architecture/tick-ohlc-management.md)


## 🌐 Lokalizáció

A rendszer támogatja a magyar és angol nyelvet. A nyelv a GUI beállításokban módosítható.

