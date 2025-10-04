import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTableModule } from '@angular/material/table';
import { MatDialogModule, MatDialog } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatChipsModule } from '@angular/material/chips';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { StrategyService } from '../../services/strategy.service';
import { NotificationService } from '../../services/notification.service';
import { Strategy, Position, CreateStrategyRequest } from '../../models/strategy.model';

@Component({
  selector: 'app-strategies',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatTableModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatChipsModule
  ],
  templateUrl: './strategies.component.html',
  styleUrls: ['./strategies.component.scss']
})
export class StrategiesComponent implements OnInit, OnDestroy {
  strategies: Strategy[] = [];
  positions: Position[] = [];
  loading = true;
  showCreateForm = false;
  strategyForm: FormGroup;
  displayedColumns: string[] = ['name', 'mode', 'symbols', 'status', 'profitToday', 'totalProfit', 'actions'];
  positionColumns: string[] = ['symbol', 'type', 'volume', 'openPrice', 'currentPrice', 'profit', 'actions'];

  private destroy$ = new Subject<void>();

  constructor(
    private strategyService: StrategyService,
    private notification: NotificationService,
    private fb: FormBuilder
  ) {
    this.strategyForm = this.fb.group({
      name: ['', Validators.required],
      mode: ['paper', Validators.required],
      symbols: ['', Validators.required],
      timeframe: ['M15', Validators.required]
    });
  }

  ngOnInit(): void {
    this.loadStrategies();
    this.loadPositions();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadStrategies(): void {
    this.loading = true;
    this.strategyService.getAllStrategies()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (strategies) => {
          this.strategies = strategies;
          this.loading = false;
        },
        error: (error) => {
          this.notification.error('Failed to load strategies');
          this.loading = false;
        }
      });
  }

  loadPositions(): void {
    this.strategyService.getAllPositions()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (positions) => {
          this.positions = positions.filter(p => p.status === 'open');
        },
        error: (error) => {
          this.notification.error('Failed to load positions');
        }
      });
  }

  createStrategy(): void {
    if (this.strategyForm.invalid) {
      this.notification.warning('Please fill all required fields');
      return;
    }

    const formValue = this.strategyForm.value;
    const request: CreateStrategyRequest = {
      name: formValue.name,
      mode: formValue.mode,
      symbols: formValue.symbols.split(',').map((s: string) => s.trim()),
      timeframe: formValue.timeframe
    };

    this.strategyService.createStrategy(request)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (strategy) => {
          this.notification.success('Strategy created successfully');
          this.loadStrategies();
          this.showCreateForm = false;
          this.strategyForm.reset({ mode: 'paper', timeframe: 'M15' });
        },
        error: (error) => {
          this.notification.error('Failed to create strategy');
        }
      });
  }

  startStrategy(id: number): void {
    this.strategyService.startStrategy(id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.notification.success('Strategy started');
          this.loadStrategies();
        },
        error: (error) => {
          this.notification.error('Failed to start strategy');
        }
      });
  }

  stopStrategy(id: number): void {
    this.strategyService.stopStrategy(id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.notification.success('Strategy stopped');
          this.loadStrategies();
        },
        error: (error) => {
          this.notification.error('Failed to stop strategy');
        }
      });
  }

  deleteStrategy(id: number): void {
    if (confirm('Are you sure you want to delete this strategy?')) {
      this.strategyService.deleteStrategy(id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: () => {
            this.notification.success('Strategy deleted');
            this.loadStrategies();
          },
          error: (error) => {
            this.notification.error('Failed to delete strategy');
          }
        });
    }
  }

  closePosition(positionId: number): void {
    if (confirm('Are you sure you want to close this position?')) {
      this.strategyService.closePosition(positionId)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: () => {
            this.notification.success('Position closed');
            this.loadPositions();
          },
          error: (error) => {
            this.notification.error('Failed to close position');
          }
        });
    }
  }

  getTotalProfit(): number {
    return this.positions.reduce((sum, p) => sum + p.profit, 0);
  }
}
