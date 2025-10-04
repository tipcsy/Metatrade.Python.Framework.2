import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { ServiceStatus, ServiceControl } from '../models/service.model';

@Injectable({
  providedIn: 'root'
})
export class BackendApiService {
  private readonly baseUrl = environment.apiBaseUrl;

  constructor(private http: HttpClient) {}

  // Service Management
  getAllServices(): Observable<ServiceStatus[]> {
    return this.http.get<ServiceStatus[]>(`${this.baseUrl}/services/status`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  getServiceStatus(serviceName: string): Observable<ServiceStatus> {
    return this.http.get<ServiceStatus>(`${this.baseUrl}/services/${serviceName}/status`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  controlService(control: ServiceControl): Observable<any> {
    return this.http.post(`${this.baseUrl}/services/control`, control)
      .pipe(
        catchError(this.handleError)
      );
  }

  startService(serviceName: string): Observable<any> {
    return this.controlService({ action: 'start', serviceName });
  }

  stopService(serviceName: string): Observable<any> {
    return this.controlService({ action: 'stop', serviceName });
  }

  restartService(serviceName: string): Observable<any> {
    return this.controlService({ action: 'restart', serviceName });
  }

  // System Health
  getSystemHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl}/health`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An error occurred';

    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }

    console.error(errorMessage);
    return throwError(() => new Error(errorMessage));
  }
}
