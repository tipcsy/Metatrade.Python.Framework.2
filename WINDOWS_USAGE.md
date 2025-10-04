# 🪟 Windows Használati Útmutató

> **FONTOS:** Az MT5 (MetaTrader 5) Terminal csak Windows-on fut, ezért a rendszer natív Windows-on történő futtatása ajánlott!

---

## 3 Lehetőséged Van

| Módszer | Előnyök | Hátrányok | Ajánlott |
|---------|---------|-----------|----------|
| **PowerShell** (.ps1) | Modern, erőteljes, színes output | Execution Policy beállítás kell | ⭐⭐⭐⭐⭐ **LEGJOBB** |
| **Batch** (.bat) | Egyszerű, mindenhol működik | Kevesebb funkció | ⭐⭐⭐⭐ |
| **WSL** (.sh) | Linux parancsok, eredeti | MT5 nem érhető el | ⭐⭐⭐ |

---

## 1️⃣ PowerShell Script-ek (AJÁNLOTT) ⭐

### Telepítés

**PowerShell Execution Policy engedélyezése:**

```powershell
# 1. Nyisd meg PowerShell-t ADMINISZTRÁTORKÉNT
# 2. Futtasd ezt a parancsot:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Válaszd: Y (Yes)
```

### Használat

```powershell
# Navigálj a projekt mappájába
cd C:\Users\YourName\Metatrade.Python.Framework.2

# Service-ek indítása
.\start_all_services.ps1

# Service-ek tesztelése
.\test_all_services.ps1

# Service-ek leállítása
.\stop_all_services.ps1
```

### Mit csinálnak?

**start_all_services.ps1:**
- Elindítja mind a 7 service-t háttérben
- Színes output (zöld/piros/sárga)
- PID mentése minden service-hez
- Portok: 5000-5006

**test_all_services.ps1:**
- HTTP health check minden service-re
- Színes eredmények (✓/✗)
- Sikeres/sikertelen számlálás

**stop_all_services.ps1:**
- Leállítja az összes service-t PID alapján
- Cleanup maradék Python processek

---

## 2️⃣ Batch Script-ek (.bat)

### Telepítés

**Nincs telepítés!** Azonnal működnek.

### Használat

```cmd
REM Nyisd meg a Command Prompt-ot (cmd)
cd C:\Users\YourName\Metatrade.Python.Framework.2

REM Service-ek indítása
start_all_services.bat

REM Service-ek tesztelése
test_all_services.bat

REM Service-ek leállítása
stop_all_services.bat
```

### Követelmények

- **curl** (Windows 10+ beépített)
- **python** (PATH-ban kell legyen)

---

## 3️⃣ WSL-ben Futtatás (.sh)

**Csak akkor használd, ha a projekt WSL-ben van!**

```bash
# WSL indítása
wsl

# Navigálás
cd /home/tipcsy/Metatrade.Python.Framework.2

# Script-ek futtatása
./start_all_services.sh
./test_all_services.sh
python3 test_backtesting_service.py
```

**FIGYELEM:** Az MT5 Terminal WSL-ből NEM érhető el!

---

## 🎯 MT5 Integráció Windows-on

### Ajánlott Struktúra

```
C:\Trading\
├── MetaTrader5\              ← MT5 Terminal telepítési könyvtár
│   ├── terminal64.exe
│   └── ...
├── Metatrade.Python.Framework.2\   ← A Python projekt
│   ├── services\
│   ├── frontend\
│   ├── start_all_services.ps1
│   └── ...
└── Python\                    ← Python telepítés (opcionális)
```

### MT5 Service Konfiguráció

Az MT5 Service-nek tudnia kell, hol van az MT5 Terminal:

```python
# services/mt5-service/config.json
{
  "mt5_path": "C:\\Trading\\MetaTrader5\\terminal64.exe",
  "login": 12345678,
  "password": "your_password",
  "server": "MetaQuotes-Demo"
}
```

---

## 📋 Lépésről-Lépésre Telepítés Windows-on

### 1. Python Telepítése

```powershell
# Letöltés: https://www.python.org/downloads/
# Telepítéskor PIPÁLD BE: "Add Python to PATH"

# Ellenőrzés:
python --version
pip --version
```

### 2. Projekt Leklonozása/Másolása

```powershell
# Git-tel (ha van Git telepítve):
cd C:\Trading
git clone <repository_url>

# VAGY egyszerűen másold át a mappát C:\Trading\-be
```

### 3. Függőségek Telepítése

```powershell
cd C:\Trading\Metatrade.Python.Framework.2

# Minden service-hez (6-7 perc):
foreach ($service in Get-ChildItem services -Directory) {
    Write-Host "Installing dependencies for $($service.Name)..."
    cd "services\$($service.Name)"
    pip install -r requirements.txt
    cd ..\..
}

# Frontend (külön):
cd frontend
npm install
cd ..
```

### 4. MT5 Terminal Telepítése

1. Töltsd le: https://www.metatrader5.com/en/download
2. Telepítsd: `C:\Trading\MetaTrader5\`
3. Indítsd el és jelentkezz be demo account-tal
4. Zárdd be (a service indítja majd)

### 5. Service-ek Indítása

```powershell
# PowerShell (AJÁNLOTT):
.\start_all_services.ps1

# VAGY Batch:
start_all_services.bat
```

### 6. Frontend Indítása

```powershell
cd frontend
npm start

# Böngészőben: http://localhost:4200
```

---

## 🧪 Tesztelés Windows-on

### Gyors Teszt

```powershell
# PowerShell:
.\test_all_services.ps1

# Batch:
test_all_services.bat
```

### Részletes Python Teszt

```powershell
python test_backtesting_service.py
```

### Manuális Teszt (PowerShell)

```powershell
# Health check
Invoke-WebRequest http://localhost:5006/health

# JSON formázva
(Invoke-WebRequest http://localhost:5006/strategies).Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### Manuális Teszt (curl)

```powershell
# Ha curl elérhető (Windows 10+)
curl http://localhost:5006/health
curl http://localhost:5006/strategies
```

---

## 🔧 Troubleshooting Windows-on

### 1. "python: command not found"

**Megoldás:**
```powershell
# Ellenőrizd, hogy Python a PATH-ban van-e:
$env:Path -split ';' | Select-String Python

# Ha nincs, add hozzá:
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Python311", "User")
```

### 2. "Execution Policy" hiba PowerShell-ben

```powershell
# Futtasd Adminisztrátorként:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Port már használatban

```powershell
# Nézd meg, mi használja:
netstat -ano | findstr :5006

# Öld meg a folyamatot:
taskkill /PID <process_id> /F
```

### 4. Service nem indul el

```powershell
# Manuális indítás hibakereséshez:
cd services\backtesting-service
python main.py

# Nézd a hibát a konzolban
```

### 5. Frontend npm hiba

```powershell
cd frontend

# Node/npm verzió ellenőrzés:
node --version    # Kell: v18+
npm --version     # Kell: v9+

# Tiszta újratelepítés:
Remove-Item node_modules -Recurse -Force
Remove-Item package-lock.json -Force
npm install
```

---

## 📁 Elérhető Script-ek

| Fájl | Típus | Platform | Leírás |
|------|-------|----------|--------|
| `start_all_services.ps1` | PowerShell | Windows | Service-ek indítása (színes) |
| `stop_all_services.ps1` | PowerShell | Windows | Service-ek leállítása |
| `test_all_services.ps1` | PowerShell | Windows | Health check (színes) |
| `start_all_services.bat` | Batch | Windows | Service-ek indítása (egyszerű) |
| `stop_all_services.bat` | Batch | Windows | Service-ek leállítása |
| `test_all_services.bat` | Batch | Windows | Health check (egyszerű) |
| `start_all_services.sh` | Bash | WSL/Linux | Service-ek indítása |
| `stop_all_services.sh` | Bash | WSL/Linux | Service-ek leállítása |
| `test_all_services.sh` | Bash | WSL/Linux | Health check |
| `test_backtesting_service.py` | Python | Minden | Részletes teszt |

---

## 🚀 Gyors Start (Windows)

```powershell
# 1. PowerShell megnyitása (ADMIN)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Projekt mappába lépés
cd C:\Trading\Metatrade.Python.Framework.2

# 3. Service-ek indítása
.\start_all_services.ps1

# 4. Tesztelés
.\test_all_services.ps1

# 5. Frontend indítása (új PowerShell ablak)
cd frontend
npm start

# 6. Böngészőben: http://localhost:4200
```

---

## 💡 Tippek

### PowerShell Profil (Opcionális)

Hozz létre aliasokat gyakori parancsokhoz:

```powershell
# Szerkesztd a profilt:
notepad $PROFILE

# Add hozzá:
function Start-Trading {
    Set-Location C:\Trading\Metatrade.Python.Framework.2
    .\start_all_services.ps1
}

function Stop-Trading {
    Set-Location C:\Trading\Metatrade.Python.Framework.2
    .\stop_all_services.ps1
}

function Test-Trading {
    Set-Location C:\Trading\Metatrade.Python.Framework.2
    .\test_all_services.ps1
}

# Most bárhonnan futtathatod:
Start-Trading
Test-Trading
Stop-Trading
```

### Automatikus Indítás Windows Indulásakor

```powershell
# 1. Készíts egy .vbs fájlt (pl: start_trading.vbs):
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "powershell.exe -WindowStyle Hidden -File C:\Trading\Metatrade.Python.Framework.2\start_all_services.ps1", 0, False

# 2. Tedd a Startup mappába:
# %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\
```

---

**Minden Windows-os script elkészült és használatra kész!** 🎉
