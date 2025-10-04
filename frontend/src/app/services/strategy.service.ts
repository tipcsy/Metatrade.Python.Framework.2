import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { Strategy, CreateStrategyRequest, Position, StrategyPerformance } from '../models/strategy.model';

@Injectable({
  providedIn: 'root'
})
export class StrategyService {
  private readonly baseUrl = environment.strategyServiceUrl;

  constructor(private http: HttpClient) {}

  // Strategy Management
  getAllStrategies(): Observable<Strategy[]> {
    return this.http.get<Strategy[]>(`${this.baseUrl}/strategies`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  getStrategy(id: number): Observable<Strategy> {
    return this.http.get<Strategy>(`${this.baseUrl}/strategies/${id}`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  createStrategy(request: CreateStrategyRequest): Observable<Strategy> {
    return this.http.post<Strategy>(`${this.baseUrl}/strategies`, request)
      .pipe(
        catchError(this.handleError)
      );
  }

  updateStrategy(id: number, updates: Partial<Strategy>): Observable<Strategy> {
    return this.http.put<Strategy>(`${this.baseUrl}/strategies/${id}`, updates)
      .pipe(
        catchError(this.handleError)
      );
  }

  deleteStrategy(id: number): Observable<any> {
    return this.http.delete(`${this.baseUrl}/strategies/${id}`)
      .pipe(
        catchError(this.handleError)
      );
  }

  startStrategy(id: number): Observable<any> {
    return this.http.post(`${this.baseUrl}/strategies/${id}/start`, {})
      .pipe(
        catchError(this.handleError)
      );
  }

  stopStrategy(id: number): Observable<any> {
    return this.http.post(`${this.baseUrl}/strategies/${id}/stop`, {})
      .pipe(
        catchError(this.handleError)
      );
  }

  // Positions
  getStrategyPositions(strategyId: number): Observable<Position[]> {
    return this.http.get<Position[]>(`${this.baseUrl}/strategies/${strategyId}/positions`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  getAllPositions(): Observable<Position[]> {
    return this.http.get<Position[]>(`${this.baseUrl}/positions`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  closePosition(positionId: number): Observable<any> {
    return this.http.post(`${this.baseUrl}/positions/${positionId}/close`, {})
      .pipe(
        catchError(this.handleError)
      );
  }

  // Performance
  getStrategyPerformance(strategyId: number): Observable<StrategyPerformance> {
    return this.http.get<StrategyPerformance>(`${this.baseUrl}/strategies/${strategyId}/performance`)
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
