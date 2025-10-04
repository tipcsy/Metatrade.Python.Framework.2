export interface ServiceStatus {
  name: string;
  status: 'online' | 'offline' | 'error' | 'starting' | 'stopping';
  uptime?: number;
  cpu?: number;
  memory?: number;
  port?: number;
  url?: string;
  metadata?: Record<string, any>;
  lastUpdate?: Date;
}

export interface ServiceControl {
  action: 'start' | 'stop' | 'restart';
  serviceName: string;
}

export interface ServiceLog {
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG';
  service: string;
  message: string;
}
