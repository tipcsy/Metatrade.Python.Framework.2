import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import {
  AIModel,
  TrainModelRequest,
  PredictionRequest,
  PredictionResult,
  ModelPerformance
} from '../models/ai.model';

@Injectable({
  providedIn: 'root'
})
export class AiService {
  private readonly baseUrl = environment.aiServiceUrl;

  constructor(private http: HttpClient) {}

  // Model Management
  getAllModels(): Observable<AIModel[]> {
    return this.http.get<AIModel[]>(`${this.baseUrl}/models`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  getModel(id: number): Observable<AIModel> {
    return this.http.get<AIModel>(`${this.baseUrl}/models/${id}`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  createModel(request: TrainModelRequest): Observable<AIModel> {
    return this.http.post<AIModel>(`${this.baseUrl}/models`, request)
      .pipe(
        catchError(this.handleError)
      );
  }

  deleteModel(id: number): Observable<any> {
    return this.http.delete(`${this.baseUrl}/models/${id}`)
      .pipe(
        catchError(this.handleError)
      );
  }

  // Training
  trainModel(request: TrainModelRequest): Observable<AIModel> {
    return this.http.post<AIModel>(`${this.baseUrl}/models/train`, request)
      .pipe(
        catchError(this.handleError)
      );
  }

  getTrainingStatus(modelId: number): Observable<{ status: string; progress: number }> {
    return this.http.get<{ status: string; progress: number }>(`${this.baseUrl}/models/${modelId}/training-status`)
      .pipe(
        retry(2),
        catchError(this.handleError)
      );
  }

  stopTraining(modelId: number): Observable<any> {
    return this.http.post(`${this.baseUrl}/models/${modelId}/stop-training`, {})
      .pipe(
        catchError(this.handleError)
      );
  }

  // Predictions
  predict(request: PredictionRequest): Observable<PredictionResult> {
    return this.http.post<PredictionResult>(`${this.baseUrl}/predict`, request)
      .pipe(
        catchError(this.handleError)
      );
  }

  batchPredict(modelId: number, symbols: string[]): Observable<PredictionResult[]> {
    return this.http.post<PredictionResult[]>(`${this.baseUrl}/predict/batch`, { modelId, symbols })
      .pipe(
        catchError(this.handleError)
      );
  }

  // Performance
  getModelPerformance(modelId: number): Observable<ModelPerformance> {
    return this.http.get<ModelPerformance>(`${this.baseUrl}/models/${modelId}/performance`)
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
