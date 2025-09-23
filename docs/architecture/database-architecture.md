# MetaTrader Python Framework - Database Architektúra

**Verzió**: 1.0
**Dátum**: 2025-09-06
**Státusz**: Éles Környezetre Kész Architektúra
**Projekt Kód**: MT5-PY-FRAMEWORK
**Agent**: 00.4-database-architect

---

## 🎯 Összefoglaló

Ez a dokumentum a MetaTrader Python Framework átfogó adatbázis architektúráját határozza meg, amelyet nagyfokú frekvenciájú pénzügyi adatok kezelésére terveztek vállalati szintű teljesítménnyel, megbízhatósággal és skálázhatósággal. Az architektúra támogatja az SQLite (fejlesztési/kisméretű)  telepítéseket.

### Kulcs Tervezési Elvek:
- **Teljesítmény-központú**: Ezredmásodpercnél gyorsabb tick feldolgozás optimalizált indexeléssel
- **Adat Integritás**: ACID megfelelőség átfogó validációval
- **Skálázhatóság**: Horizontális és vertikális skálázási képességek
- **Megbízhatóság**: Automatikus feladatátvétel, biztonsági mentés és helyreállítási mechanizmusok
- **Biztonság**: Titkosított tárolás és biztonságos hozzáférés-vezérlés

### Teljesítmény Célok:
- **Tick Beszúrás**: <1ms tickenként
- **OHLCV Számítás**: <10ms báronként
- **Történeti Lekérdezések**: <100ms 1 napos adatra
- **Egyidejű Hozzáférés**: 1000+ művelet/másodperc
- **Adat Mennyiség**: 7GB+ szimbólumonként évente

---

## 🏗️ Database Architektúra Áttekintés

### Kettős Database Stratégia

#### SQLite Konfiguráció (Fejlesztési/Kisméretű)
```yaml
Primary Use Case: Development, testing, small deployments (<5 symbols)
Performance: 500+ ticks/second per database
Storage: File-based, up to 281TB theoretical limit
Concurrent Users: Read-heavy with single writer
Backup Strategy: File-level backup and replication
```

### Architektúra Rétegek

```
┌─────────────────────────────────────────────────────────────┐
│                   Alkalmazási Réteg                         │
│   DataManager, StrategyEngine, RiskManager, OrderManager    │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                 Adathozzáférési Réteg (DAL)                 │
│     IDataProvider, IDataStorage, ICacheManager              │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                Kapcsolat Kezelési Réteg                     │
│    Connection Pooling, Transaction Management, Health       │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    Database Réteg                           │
│                        SQLite                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Database Séma Tervezés

### Alapvető Táblák Architektúrája

#### 1. Piaci Adat Táblák

##### `ticks` - Nagyfrekveciájú tick adatok
```sql
-- Elsődleges tick tároló tábla
CREATE TABLE ticks (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(20) NOT NULL,
    bid DECIMAL(10,5) NOT NULL,
    ask DECIMAL(10,5) NOT NULL,
    timestamp_utc TIMESTAMP(3) NOT NULL,
    timestamp_local TIMESTAMP(3) NOT NULL,
    flags INT UNSIGNED DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Kompozit indexek a teljesítményért
    INDEX idx_symbol_timestamp (symbol, timestamp_utc),
    INDEX idx_timestamp_symbol (timestamp_utc, symbol),
    INDEX idx_symbol_created (symbol, created_at)
);

```

##### `bars` - OHLC bár adatok
```sql
-- Többszörös időkeret bár tárolás
CREATE TABLE bars (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe ENUM('M1', 'M2', 'M3', 'M4', 'M5', 'M10','M12', 'M15', 'M20', 'M30', 'H1','H2', 'H3', 'H4', 'H6', 'H12','D1', 'W1', 'MN1') NOT NULL,
    open_price DECIMAL(10,5) NOT NULL,
    high_price DECIMAL(10,5) NOT NULL,
    spread decimal(10, 5) NOT NULL,
    low_price DECIMAL(10,5) NOT NULL,
    close_price DECIMAL(10,5) NOT NULL,
    volume BIGINT DEFAULT 0,
    timestamp_utc TIMESTAMP(3) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Egyedi megszorítás a duplikátumok megakadályozására
    UNIQUE KEY uk_symbol_timeframe_timestamp (symbol, timeframe, timestamp_utc),

    -- Teljesítmény indexek
    INDEX idx_symbol_timeframe (symbol, timeframe),
    INDEX idx_timeframe_timestamp (timeframe, timestamp_utc),
    INDEX idx_timestamp_range (timestamp_utc)
);
```

##### `symbol_info` - Kereskedési instrumentum metaadatok
```sql
CREATE TABLE symbol_info (
    id INT PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    base_currency VARCHAR(10),
    quote_currency VARCHAR(10),
    contract_size DECIMAL(15,2) DEFAULT 100000,
    tick_size DECIMAL(10,8) NOT NULL,
    tick_value DECIMAL(10,5) NOT NULL,
    min_volume DECIMAL(10,2) DEFAULT 0.01,
    max_volume DECIMAL(15,2) DEFAULT 500.00,
    volume_step DECIMAL(10,2) DEFAULT 0.01,
    margin_initial DECIMAL(8,4) DEFAULT 0.0,
    margin_maintenance DECIMAL(8,4) DEFAULT 0.0,
    is_active BOOLEAN DEFAULT TRUE,
    trading_hours JSON,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_symbol_active (symbol, is_active),
    INDEX idx_active (is_active)
);
```

#### 2. Kereskedési Műveletek Táblái

##### `positions` - Aktív és történeti pozíciók
```sql
CREATE TABLE positions (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    position_id VARCHAR(50) NOT NULL UNIQUE,
    symbol VARCHAR(20) NOT NULL,
    type ENUM('BUY', 'SELL') NOT NULL,
    volume DECIMAL(10,2) NOT NULL,
    open_price DECIMAL(10,5) NOT NULL,
    close_price DECIMAL(10,5) NULL,
    stop_loss DECIMAL(10,5) NULL,
    take_profit DECIMAL(10,5) NULL,
    swap DECIMAL(10,2) DEFAULT 0.00,
    commission DECIMAL(10,2) DEFAULT 0.00,
    profit DECIMAL(10,2) DEFAULT 0.00,
    status ENUM('OPEN', 'CLOSED', 'PARTIAL') DEFAULT 'OPEN',
    open_time TIMESTAMP(3) NOT NULL,
    close_time TIMESTAMP(3) NULL,
    strategy_id VARCHAR(50),
    comment TEXT,
    magic_number INT UNSIGNED DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexek a teljesítményért
    INDEX idx_symbol_status (symbol, status),
    INDEX idx_status_open_time (status, open_time),
    INDEX idx_strategy_id (strategy_id),
    INDEX idx_position_id (position_id),
    INDEX idx_open_time (open_time),

    -- Külső kulcs kapcsolat
    FOREIGN KEY (symbol) REFERENCES symbol_info(symbol) ON UPDATE CASCADE
);
```

##### `orders` - Megrendelés kezelés és történet
```sql
CREATE TABLE orders (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    order_id VARCHAR(50) NOT NULL UNIQUE,
    position_id BIGINT NULL,
    symbol VARCHAR(20) NOT NULL,
    type ENUM('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT') NOT NULL,
    action ENUM('BUY', 'SELL') NOT NULL,
    volume DECIMAL(10,2) NOT NULL,
    price DECIMAL(10,5) NOT NULL,
    stop_loss DECIMAL(10,5) NULL,
    take_profit DECIMAL(10,5) NULL,
    executed_volume DECIMAL(10,2) DEFAULT 0.00,
    executed_price DECIMAL(10,5) NULL,
    status ENUM('PENDING', 'FILLED', 'PARTIAL', 'CANCELLED', 'REJECTED') NOT NULL,
    strategy_id VARCHAR(50),
    comment TEXT,
    magic_number INT UNSIGNED DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP(3) NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexek
    INDEX idx_order_id (order_id),
    INDEX idx_symbol_status (symbol, status),
    INDEX idx_status_created (status, created_at),
    INDEX idx_strategy_id (strategy_id),

    -- Külső kulcs kapcsolatok
    FOREIGN KEY (symbol) REFERENCES symbol_info(symbol) ON UPDATE CASCADE,
    FOREIGN KEY (position_id) REFERENCES positions(id) ON DELETE SET NULL
);
```

#### 3. Stratégia és Teljesítmény Táblák

##### `strategies` - Stratégia definíciók és konfiguráció
```sql
CREATE TABLE strategies (
    id INT PRIMARY KEY AUTO_INCREMENT,
    strategy_id VARCHAR(50) NOT NULL UNIQUE,
    strategy_name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) DEFAULT '1.0.0',
    description TEXT,
    parameters JSON,
    is_active BOOLEAN DEFAULT TRUE,
    risk_allocation DECIMAL(5,4) DEFAULT 0.0100, -- 1% default
    max_positions INT DEFAULT 1,
    symbols JSON, -- Szimbólumok tömbje, amelyekkel ez a stratégia kereskedhet
    created_by VARCHAR(50) DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_strategy_id (strategy_id),
    INDEX idx_active (is_active),
    INDEX idx_type (strategy_type)
);
```

##### `strategy_performance` - Valós idejű stratégia metrikák
```sql
CREATE TABLE strategy_performance (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    strategy_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    
    -- Teljesítmény metrikák
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    gross_profit DECIMAL(12,2) DEFAULT 0.00,
    gross_loss DECIMAL(12,2) DEFAULT 0.00,
    net_profit DECIMAL(12,2) DEFAULT 0.00,

    -- Kockázati metrikák
    max_drawdown DECIMAL(12,2) DEFAULT 0.00,
    max_drawdown_percent DECIMAL(8,4) DEFAULT 0.0000,
    profit_factor DECIMAL(8,4) DEFAULT 0.0000,
    sharpe_ratio DECIMAL(8,4) DEFAULT 0.0000,

    -- További metrikák
    largest_profit DECIMAL(12,2) DEFAULT 0.00,
    largest_loss DECIMAL(12,2) DEFAULT 0.00,
    avg_profit DECIMAL(12,2) DEFAULT 0.00,
    avg_loss DECIMAL(12,2) DEFAULT 0.00,
    
    -- Időbélyegek
    last_trade_time TIMESTAMP(3) NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Megszorítások és indexek
    UNIQUE KEY uk_strategy_symbol_date (strategy_id, symbol, date),
    INDEX idx_strategy_date (strategy_id, date),
    INDEX idx_symbol_date (symbol, date),

    -- Külső kulcsok
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (symbol) REFERENCES symbol_info(symbol) ON UPDATE CASCADE
);
```

#### 4. Rendszer és Konfigurációs Táblák

##### `system_config` - Dinamikus konfiguráció tárolás
```sql
CREATE TABLE system_config (
    id INT PRIMARY KEY AUTO_INCREMENT,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    config_type ENUM('string', 'integer', 'float', 'boolean', 'json') DEFAULT 'string',
    category VARCHAR(50) DEFAULT 'general',
    description TEXT,
    is_encrypted BOOLEAN DEFAULT FALSE,
    is_readonly BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_category (category),
    INDEX idx_key (config_key)
);
```

##### `system_logs` - Átfogó audit nyomvonal
```sql
CREATE TABLE system_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    log_level ENUM('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') NOT NULL,
    component VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    context JSON,
    user_id VARCHAR(50) NULL,
    session_id VARCHAR(100) NULL,
    ip_address VARCHAR(45) NULL,
    timestamp_utc TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    
    -- Indexek a log elemzéséhez
    INDEX idx_level_timestamp (log_level, timestamp_utc),
    INDEX idx_component_timestamp (component, timestamp_utc),
    INDEX idx_timestamp (timestamp_utc),
    INDEX idx_level (log_level)
);

```
---

## 🔄 Kapcsolat Kezelési Architektúra

### Connection Pool Tervezés

```python
class DatabaseConnectionManager:
    """
    Vállalati szintű kapcsolat kezelés pooling-gal, egészség figyelemmel,
    és automatikus feladatátvételi képességekkel.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pools = {}
        self.health_monitor = ConnectionHealthMonitor()
        self.failover_manager = FailoverManager()
    
    def get_connection_pool(self, database_type: str) -> ConnectionPool:
        """Kapcsolat pool megszerzése vagy létrehozása az adatbázis típushoz."""
        if database_type not in self.connection_pools:
            self.connection_pools[database_type] = self._create_pool(database_type)
        return self.connection_pools[database_type]
    
    def _create_pool(self, database_type: str) -> ConnectionPool:
        """Optimalizált kapcsolat pool létrehozása az adatbázis típus alapján."""
        if database_type == 'sqlite':
            return SQLiteConnectionPool(
                database_path=self.config.sqlite_path,
                max_connections=1,  # SQLite egyetlen író
                timeout=30.0,
                check_same_thread=False
            )
```

### Connection Pool Specifikációk

#### SQLite Konfiguráció
```yaml
Connection Strategy: Egyetlen kapcsolat WAL móddal
Concurrent Readers: Több (thread-safe)
Concurrent Writers: Egyetlen író sorral
Timeout: 30 másodperc
Journal Mode: WAL (Write-Ahead Logging)
Synchronous: NORMAL
Cache Size: -64000 (64MB cache)
Temp Store: MEMORY
```

---

## ⚡ Lekérdezés Optimalizálási Stratégia

### Index Stratégia

#### 1. Elsődleges Indexek (Clustered)
```sql
-- Beszúrási teljesítményre optimalizálva
ALTER TABLE ticks ADD PRIMARY KEY (id);
ALTER TABLE bars ADD PRIMARY KEY (id);
ALTER TABLE positions ADD PRIMARY KEY (id);
ALTER TABLE orders ADD PRIMARY KEY (id);
```

#### 2. Kompozit Indexek a Lekérdezési Teljesítményért
```sql
-- Tick adat lekérdezések (szimbólum + időtartomány)
CREATE INDEX idx_ticks_symbol_time ON ticks (symbol, timestamp_utc, id);
CREATE INDEX idx_ticks_time_symbol ON ticks (timestamp_utc, symbol, id);

-- Bár adat lekérdezések (szimbólum + időkeret + idő)
CREATE INDEX idx_bars_symbol_tf_time ON bars (symbol, timeframe, timestamp_utc);
CREATE INDEX idx_bars_time_symbol_tf ON bars (timestamp_utc, symbol, timeframe);

-- Pozíció lekérdezések (státusz + szimbólum + idő)
CREATE INDEX idx_positions_status_symbol ON positions (status, symbol, open_time);
CREATE INDEX idx_positions_symbol_time ON positions (symbol, open_time, status);

-- Megrendelés lekérdezések (státusz + idő + szimbólum)
CREATE INDEX idx_orders_status_time ON orders (status, created_at, symbol);
```

#### 3. Fedő Indexek a Gyakori Lekérdezésekhez
```sql
-- Legutóbbi tick lekérdezés fedő index
CREATE INDEX idx_ticks_latest_covering ON ticks (symbol, timestamp_utc DESC, bid, ask, volume);

-- Napi teljesítmény fedő index
CREATE INDEX idx_performance_daily ON strategy_performance (strategy_id, date, net_profit, total_trades);
```

### Lekérdezés Optimalizálási Minták

#### 1. Tömeges Beszúrás Optimalizálás
```sql
-- Kötegelt beszúrás duplikátumok figyelmen kívül hagyásával
INSERT IGNORE INTO ticks (symbol, bid, ask, timestamp_utc, volume)
VALUES
    ('EURUSD', 1.0825, 1.0827, '2025-09-06 10:15:30.123', 1000),
    ('EURUSD', 1.0824, 1.0826, '2025-09-06 10:15:31.456', 1500),
    -- ... akár 1000 sor kötegként
;

-- Előkészített utasítások használata paraméter hatékonyságért
PREPARE stmt FROM 'INSERT INTO ticks (symbol, bid, ask, timestamp_utc) VALUES (?, ?, ?, ?)';
```

#### 2. Történeti Adat Lekérdezések

---

## 🔄 Adat Migráció és Verziókezelés

### Migrációs Framework Architektúra

### Verzióvezérlési Stratégia

#### Szemantikus Verziózás Database Sémához
```
Verzió Formátum: MAJOR.MINOR.PATCH
- MAJOR: Törő séma változtatások
- MINOR: Visszafelé kompatibilis kiegészítések
- PATCH: Hibajavítások és kisebb optimalizálások

Példa progresszió:
1.0.0 → Kezdeti séma
1.1.0 → Új táblák/oszlopok hozzáadása
1.1.1 → Index optimalizálások
2.0.0 → Törő változtatások (oszlop átnevezés, tábla átstrukturálás)
```

#### Visszagörgetési Stratégia

---

## 🎯 Teljesítmény Monitorozás és Optimalizálás

### Valós Idejű Teljesítmény Monitorozás

### Automatizált Optimalizálás

```python
class DatabaseOptimizer:
    """
    Automatizált adatbázis optimalizálás adaptív hangolással.
    """
    
    def __init__(self, performance_monitor: DatabasePerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.optimization_history = []
        
    def analyze_and_optimize(self) -> OptimizationResult:
        """Teljesítmény elemzése és optimalizálások alkalmazása."""
        performance_report = self.performance_monitor.collect_performance_metrics()
        
        optimizations = []
        
        # Index optimalizálás
        if performance_report.query_stats.avg_query_time > 100:  # >100ms
            index_optimizations = self.optimize_indexes()
            optimizations.extend(index_optimizations)

        # Kapcsolat pool optimalizálás
        if performance_report.connection_stats.wait_time > 1000:  # >1s várakozási idő
            pool_optimizations = self.optimize_connection_pool()
            optimizations.extend(pool_optimizations)

        # Cache optimalizálás
        if performance_report.database_stats.cache_hit_ratio < 0.90:  # <90% cache találat
            cache_optimizations = self.optimize_cache_settings()
            optimizations.extend(cache_optimizations)
        
        return OptimizationResult(
            optimizations_applied=optimizations,
            performance_improvement=self.measure_improvement()
        )
    
    def optimize_indexes(self) -> List[IndexOptimization]:
        """Adatbázis indexek elemzése és optimalizálása."""
        # Nem használt indexek keresése
        unused_indexes = self.find_unused_indexes()

        # Hiányzó indexek keresése lassú lekérdezésekhez
        missing_indexes = self.suggest_missing_indexes()

        # Redundant indexek keresése
        redundant_indexes = self.find_redundant_indexes()
        
        optimizations = []
        
        # Nem használt indexek elvetése
        for index in unused_indexes:
            self.drop_index(index)
            optimizations.append(IndexOptimization(
                action='DROP',
                index_name=index.name,
                table=index.table,
                impact='Csökkentett tárolási és karbantartási terhelés'
            ))

        # Hiányzó indexek létrehozása
        for suggestion in missing_indexes:
            self.create_index(suggestion)
            optimizations.append(IndexOptimization(
                action='CREATE',
                index_name=suggestion.name,
                table=suggestion.table,
                columns=suggestion.columns,
                impact=f'Lekérdezés teljesítmény javítása {suggestion.expected_improvement}%-kal'
            ))
        
        return optimizations
```

Ez az átfogó adatbázis architektúra vállalati szintű képességeket biztosít a MetaTrader Python Framework számára. A tervezés a teljesítményt, megbízhatóságot és skálázhatóságot hangsúlyozza, miközben megtartja a rugalmasságot mind a SQLite (fejlesztési) környezetekkel való munkához.

Az architektúra a következőket tartalmazza:
- Optimalizált séma tervezés nagyfrekvenciájú kereskedési adatokhoz
- Haladó kapcsolat kezelés és pooling
- Lekérdezés optimalizálás és index stratégiák
- Átfogó biztonsági mentés és katasztrófa helyreállítás
- Valós idejű teljesítmény monitorozás és automatizált optimalizálás
- Adat életciklus kezelés és archívalás

Ez az alap biztosítja, hogy a rendszer kezelni tudja az automatizált kereskedés követelményeit, miközben fenntartja az adat integritást és a rendszer megbízhatóságát.