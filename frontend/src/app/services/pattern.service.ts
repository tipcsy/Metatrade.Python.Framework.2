import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import {
  TechnicalIndicator,
  CandlestickPattern,
  PatternScanRequest,
  PatternScanResult,
  ChartData
} from '../models/pattern.model';

@Injectable({
  providedIn: 'root'
})
export class PatternService {
  private readonly baseUrl = environment.patternServiceUrl;

  constructor(private http: HttpClient) {}

  // Pattern Detection
  scanPatterns(request: PatternScanRequest): Observable<PatternScanResult[]> {
    return this.http.post<PatternScanResult[]>(`${this.baseUrl}/patterns/scan`, request)
      .pipe(
        catchError(this.handleError)
      );
  }

  detectPatternsForSymbol(symbol: string, timeframe: string): Observable<CandlestickPattern[]> {
    return this.http.get<CandlestickPattern[]>(`${this.baseUrl}/patterns/${symbol}/${timeframe}`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  // Technical Indicators
  getIndicators(symbol: string, timeframe: string): Observable<TechnicalIndicator[]> {
    return this.http.get<TechnicalIndicator[]>(`${this.baseUrl}/indicators/${symbol}/${timeframe}`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  calculateIndicator(symbol: string, timeframe: string, indicatorName: string, params?: any): Observable<TechnicalIndicator> {
    return this.http.post<TechnicalIndicator>(
      `${this.baseUrl}/indicators/calculate`,
      { symbol, timeframe, indicator: indicatorName, parameters: params }
    ).pipe(
      catchError(this.handleError)
    );
  }

  // Chart Data
  getChartData(symbol: string, timeframe: string, bars: number = 100): Observable<ChartData> {
    return this.http.get<ChartData>(`${this.baseUrl}/chart/${symbol}/${timeframe}?bars=${bars}`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  // Pattern Analysis
  analyzeSymbol(symbol: string, timeframe: string): Observable<PatternScanResult> {
    return this.http.get<PatternScanResult>(`${this.baseUrl}/analysis/${symbol}/${timeframe}`)
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
