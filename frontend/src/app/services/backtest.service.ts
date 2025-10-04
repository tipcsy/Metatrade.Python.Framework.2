import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { BacktestRequest, BacktestResult } from '../models/backtest.model';

@Injectable({
  providedIn: 'root'
})
export class BacktestService {
  private readonly baseUrl = environment.backtestingServiceUrl;

  constructor(private http: HttpClient) {}

  // Backtest Management
  createBacktest(request: BacktestRequest): Observable<BacktestResult> {
    return this.http.post<BacktestResult>(`${this.baseUrl}/backtests`, request)
      .pipe(
        catchError(this.handleError)
      );
  }

  getAllBacktests(): Observable<BacktestResult[]> {
    return this.http.get<BacktestResult[]>(`${this.baseUrl}/backtests`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  getBacktest(id: number): Observable<BacktestResult> {
    return this.http.get<BacktestResult>(`${this.baseUrl}/backtests/${id}`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  deleteBacktest(id: number): Observable<any> {
    return this.http.delete(`${this.baseUrl}/backtests/${id}`)
      .pipe(
        catchError(this.handleError)
      );
  }

  // Run backtest
  runBacktest(request: BacktestRequest): Observable<BacktestResult> {
    return this.http.post<BacktestResult>(`${this.baseUrl}/backtests/run`, request)
      .pipe(
        catchError(this.handleError)
      );
  }

  // Get backtest status
  getBacktestStatus(id: number): Observable<{ status: string; progress: number }> {
    return this.http.get<{ status: string; progress: number }>(`${this.baseUrl}/backtests/${id}/status`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An error occurred';

    if (error.error instanceof ErrorEvent) {
      errorMessage = `Error: ${error.error.message}`;
    } else {
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }

    console.error(errorMessage);
    return throwError(() => new Error(errorMessage));
  }
}
