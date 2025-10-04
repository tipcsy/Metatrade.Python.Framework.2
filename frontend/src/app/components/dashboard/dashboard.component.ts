import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatChipsModule } from '@angular/material/chips';
import { MatGridListModule } from '@angular/material/grid-list';
import { interval, Subject } from 'rxjs';
import { takeUntil, switchMap, startWith } from 'rxjs/operators';
import { BackendApiService } from '../../services/backend-api.service';
import { StrategyService } from '../../services/strategy.service';
import { NotificationService } from '../../services/notification.service';
import { ServiceStatus } from '../../models/service.model';
import { Strategy } from '../../models/strategy.model';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatChipsModule,
    MatGridListModule
  ],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit, OnDestroy {
  services: ServiceStatus[] = [];
  strategies: Strategy[] = [];
  loading = true;
  systemHealth: any = null;

  private destroy$ = new Subject<void>();

  constructor(
    private backendApi: BackendApiService,
    private strategyService: StrategyService,
    private notification: NotificationService
  ) {}

  ngOnInit(): void {
    this.loadData();
    this.startPolling();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private loadData(): void {
    this.loading = true;

    this.backendApi.getAllServices()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (services) => {
          this.services = services;
          this.loading = false;
        },
        error: (error) => {
          this.notification.error('Failed to load services');
          this.loading = false;
        }
      });

    this.strategyService.getAllStrategies()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (strategies) => {
          this.strategies = strategies;
        },
        error: (error) => {
          this.notification.error('Failed to load strategies');
        }
      });

    this.backendApi.getSystemHealth()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (health) => {
          this.systemHealth = health;
        },
        error: () => {
          // Silent fail for health check
        }
      });
  }

  private startPolling(): void {
    interval(environment.pollingInterval)
      .pipe(
        startWith(0),
        switchMap(() => this.backendApi.getAllServices()),
        takeUntil(this.destroy$)
      )
      .subscribe({
        next: (services) => {
          this.services = services;
        },
        error: () => {
          // Silent fail
        }
      });
  }

  getOnlineServicesCount(): number {
    return this.services.filter(s => s.status === 'online').length;
  }

  getTotalServicesCount(): number {
    return this.services.length;
  }

  getRunningStrategiesCount(): number {
    return this.strategies.filter(s => s.status === 'running').length;
  }

  getTotalProfit(): number {
    return this.strategies.reduce((sum, s) => sum + (s.totalProfit || 0), 0);
  }

  getProfitToday(): number {
    return this.strategies.reduce((sum, s) => sum + (s.profitToday || 0), 0);
  }

  getServiceStatusClass(status: string): string {
    switch (status) {
      case 'online':
        return 'status-online';
      case 'offline':
        return 'status-offline';
      case 'error':
        return 'status-error';
      default:
        return 'status-unknown';
    }
  }

  refreshData(): void {
    this.loadData();
    this.notification.info('Dashboard refreshed');
  }

  controlService(serviceName: string, action: 'start' | 'stop' | 'restart'): void {
    this.backendApi.controlService({ serviceName, action })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.notification.success(`Service ${action} command sent`);
          setTimeout(() => this.loadData(), 2000);
        },
        error: (error) => {
          this.notification.error(`Failed to ${action} service`);
        }
      });
  }
}
