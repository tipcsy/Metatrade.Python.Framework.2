import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { Candle } from '../models/pattern.model';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private readonly baseUrl = environment.dataServiceUrl;

  constructor(private http: HttpClient) {}

  // Market Data
  getHistoricalData(symbol: string, timeframe: string, bars: number = 100): Observable<Candle[]> {
    return this.http.get<Candle[]>(`${this.baseUrl}/data/historical/${symbol}/${timeframe}?bars=${bars}`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  getLatestTick(symbol: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/data/tick/${symbol}`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  getSymbols(): Observable<string[]> {
    return this.http.get<string[]>(`${this.baseUrl}/data/symbols`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  // Account Data
  getAccountInfo(): Observable<any> {
    return this.http.get(`${this.baseUrl}/account/info`)
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
