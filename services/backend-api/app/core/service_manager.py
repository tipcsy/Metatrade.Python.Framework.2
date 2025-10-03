import json
import subprocess
import psutil
import time
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from app.models.service import ServiceConfig, ServiceInfo, ServiceStatus

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Service Manager - Mikroszolgáltatások életciklus kezelése

    Felelősségek:
    - Service konfigurációk betöltése
    - Service indítás/leállítás/újraindítás
    - Service állapot követése
    - Process management
    """

    def __init__(self, config_path: str = "config.json"):
        """
        ServiceManager inicializálás

        Args:
            config_path: Config fájl elérési útja
        """
        self.config_path = config_path
        self.services: Dict[str, ServiceInfo] = {}
        self.base_dir = Path(__file__).parent.parent.parent.parent.resolve()

        logger.info(f"ServiceManager inicializálva, base_dir: {self.base_dir}")

    def load_service_configs(self) -> None:
        """
        Service konfigurációk betöltése a config.json fájlból
        """
        config_file = self.base_dir / "backend-api" / self.config_path

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            services_config = config_data.get('services', [])

            for service_config in services_config:
                config = ServiceConfig(**service_config)

                # ServiceInfo létrehozása
                service_info = ServiceInfo(
                    name=config.name,
                    status=ServiceStatus.OFFLINE,
                    port=config.port,
                    config=config
                )

                self.services[config.name] = service_info
                logger.info(f"Service konfiguráció betöltve: {config.name} (port: {config.port})")

            logger.info(f"Összesen {len(self.services)} service konfiguráció betöltve")

        except FileNotFoundError:
            logger.error(f"Config fájl nem található: {config_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Config fájl nem valid JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Hiba a konfigurációk betöltésekor: {e}")
            raise

    def _is_port_in_use(self, port: int) -> bool:
        """
        Ellenőrzi hogy egy port használatban van-e

        Args:
            port: Port szám

        Returns:
            True ha a port foglalt, False ha szabad
        """
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                return True
        return False

    def _find_process_by_port(self, port: int) -> Optional[psutil.Process]:
        """
        Megkeresi a processt amely az adott portot használja

        Args:
            port: Port szám

        Returns:
            Process object vagy None
        """
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def start_service(self, name: str) -> tuple[bool, str]:
        """
        Service indítása

        Args:
            name: Service neve

        Returns:
            (success, message) tuple
        """
        if name not in self.services:
            return False, f"Service '{name}' nem található"

        service = self.services[name]

        # Ellenőrizzük hogy már fut-e
        if service.status == ServiceStatus.STARTING:
            return False, f"Service '{name}' már indítás alatt van"

        if service.status == ServiceStatus.ONLINE:
            # Double check - valóban fut?
            if service.pid and psutil.pid_exists(service.pid):
                return False, f"Service '{name}' már fut (PID: {service.pid})"
            else:
                # PID nem létezik, de státusz online - reset
                logger.warning(f"Service '{name}' státusza online volt, de a process nem létezik")
                service.status = ServiceStatus.OFFLINE
                service.pid = None

        # Ellenőrizzük hogy a port szabad-e
        if self._is_port_in_use(service.port):
            existing_proc = self._find_process_by_port(service.port)
            if existing_proc:
                logger.warning(f"Port {service.port} már használatban van (PID: {existing_proc.pid})")
                return False, f"Port {service.port} már használatban van"

        try:
            service.status = ServiceStatus.STARTING
            service.error = None

            # Service working directory
            service_dir = self.base_dir / service.config.path

            if not service_dir.exists():
                service.status = ServiceStatus.ERROR
                service.error = f"Service könyvtár nem található: {service_dir}"
                return False, service.error

            # Python futtatható keresése
            python_executable = "python"

            # Parancs előkészítése
            cmd = service.config.command.split()
            cmd[0] = python_executable  # Biztosítjuk hogy python-t használunk

            logger.info(f"Service '{name}' indítása: {' '.join(cmd)} (cwd: {service_dir})")

            # Process indítása
            process = subprocess.Popen(
                cmd,
                cwd=str(service_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Új session hogy ne kapja a SIGINT-et
            )

            service.pid = process.pid
            logger.info(f"Service '{name}' elindítva (PID: {service.pid})")

            # Várunk amíg a service elérhető lesz (max 30 másodperc)
            startup_timeout = 30
            start_time = time.time()

            while time.time() - start_time < startup_timeout:
                # Ellenőrizzük hogy a process még fut-e
                if process.poll() is not None:
                    # Process leállt
                    stdout, stderr = process.communicate()
                    service.status = ServiceStatus.ERROR
                    service.error = f"Service azonnal leállt. Stderr: {stderr.decode('utf-8')[:200]}"
                    service.pid = None
                    logger.error(f"Service '{name}' azonnal leállt: {service.error}")
                    return False, service.error

                # Health check
                if self._check_health(service):
                    service.status = ServiceStatus.ONLINE
                    service.last_check = datetime.now()
                    service.restart_count = 0
                    logger.info(f"Service '{name}' sikeresen elindult és elérhető")
                    return True, f"Service '{name}' sikeresen elindítva"

                time.sleep(1)

            # Timeout
            service.status = ServiceStatus.ERROR
            service.error = "Service nem válaszolt a startup timeout alatt"
            logger.error(f"Service '{name}' nem vált elérhetővé {startup_timeout} másodperc alatt")

            # Megpróbáljuk leállítani
            self._kill_process(service)

            return False, service.error

        except Exception as e:
            service.status = ServiceStatus.ERROR
            service.error = str(e)
            service.pid = None
            logger.error(f"Hiba a service '{name}' indításakor: {e}", exc_info=True)
            return False, f"Hiba a service indításakor: {str(e)}"

    def stop_service(self, name: str, force: bool = False) -> tuple[bool, str]:
        """
        Service leállítása

        Args:
            name: Service neve
            force: Kényszerített leállítás (SIGKILL)

        Returns:
            (success, message) tuple
        """
        if name not in self.services:
            return False, f"Service '{name}' nem található"

        service = self.services[name]

        if service.status == ServiceStatus.OFFLINE:
            return True, f"Service '{name}' már le van állítva"

        if not service.pid:
            service.status = ServiceStatus.OFFLINE
            return True, f"Service '{name}' nem fut (nincs PID)"

        try:
            service.status = ServiceStatus.STOPPING

            if not psutil.pid_exists(service.pid):
                logger.warning(f"Service '{name}' PID-je ({service.pid}) nem létezik")
                service.status = ServiceStatus.OFFLINE
                service.pid = None
                return True, f"Service '{name}' már nem fut"

            process = psutil.Process(service.pid)

            if force:
                # Azonnal kill
                logger.info(f"Service '{name}' (PID: {service.pid}) kényszerített leállítása")
                process.kill()
                service.status = ServiceStatus.OFFLINE
                service.pid = None
                return True, f"Service '{name}' kényszerítetten leállítva"
            else:
                # Graceful shutdown
                logger.info(f"Service '{name}' (PID: {service.pid}) graceful leállítása")
                process.terminate()

                # Várunk max 10 másodpercet
                try:
                    process.wait(timeout=10)
                    service.status = ServiceStatus.OFFLINE
                    service.pid = None
                    logger.info(f"Service '{name}' sikeresen leállt")
                    return True, f"Service '{name}' sikeresen leállítva"
                except psutil.TimeoutExpired:
                    # Ha nem áll le, akkor kill
                    logger.warning(f"Service '{name}' nem állt le gracefully, kényszerített leállítás")
                    process.kill()
                    service.status = ServiceStatus.OFFLINE
                    service.pid = None
                    return True, f"Service '{name}' kényszerítetten leállítva (nem válaszolt)"

        except psutil.NoSuchProcess:
            service.status = ServiceStatus.OFFLINE
            service.pid = None
            return True, f"Service '{name}' már nem fut"
        except Exception as e:
            logger.error(f"Hiba a service '{name}' leállításakor: {e}", exc_info=True)
            return False, f"Hiba a service leállításakor: {str(e)}"

    def restart_service(self, name: str) -> tuple[bool, str]:
        """
        Service újraindítása

        Args:
            name: Service neve

        Returns:
            (success, message) tuple
        """
        if name not in self.services:
            return False, f"Service '{name}' nem található"

        logger.info(f"Service '{name}' újraindítása")

        # Leállítás
        success, message = self.stop_service(name)
        if not success:
            return False, f"Nem sikerült leállítani: {message}"

        # Kis várakozás
        time.sleep(2)

        # Indítás
        success, message = self.start_service(name)
        return success, message

    def get_service_status(self, name: str) -> Optional[ServiceInfo]:
        """
        Egy service állapotának lekérése

        Args:
            name: Service neve

        Returns:
            ServiceInfo vagy None
        """
        return self.services.get(name)

    def get_all_services_status(self) -> List[ServiceInfo]:
        """
        Az összes service állapotának lekérése

        Returns:
            ServiceInfo lista
        """
        return list(self.services.values())

    def _check_health(self, service: ServiceInfo) -> bool:
        """
        Health check végrehajtása egy service-en

        Args:
            service: ServiceInfo object

        Returns:
            True ha elérhető, False ha nem
        """
        try:
            url = f"http://localhost:{service.port}{service.config.health_endpoint}"
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _kill_process(self, service: ServiceInfo) -> None:
        """
        Service process kényszerített leállítása

        Args:
            service: ServiceInfo object
        """
        if service.pid and psutil.pid_exists(service.pid):
            try:
                process = psutil.Process(service.pid)
                process.kill()
                logger.info(f"Service '{service.name}' process (PID: {service.pid}) kényszerítetten leállítva")
            except Exception as e:
                logger.error(f"Hiba a process leállításakor: {e}")

        service.pid = None
        service.status = ServiceStatus.OFFLINE

    def auto_start_services(self) -> None:
        """
        Automatikusan elindítja azokat a service-eket amelyeknél auto_start=True
        """
        logger.info("Auto-start service-ek indítása...")

        for service in self.services.values():
            if service.config.auto_start:
                logger.info(f"Auto-start: {service.name}")
                success, message = self.start_service(service.name)
                if success:
                    logger.info(f"  ✓ {message}")
                else:
                    logger.error(f"  ✗ {message}")

                # Kis várakozás a következő service előtt
                time.sleep(2)

    def shutdown_all_services(self) -> None:
        """
        Leállítja az összes futó service-t
        """
        logger.info("Összes service leállítása...")

        for service in self.services.values():
            if service.status != ServiceStatus.OFFLINE:
                logger.info(f"Leállítás: {service.name}")
                success, message = self.stop_service(service.name)
                if success:
                    logger.info(f"  ✓ {message}")
                else:
                    logger.error(f"  ✗ {message}")
