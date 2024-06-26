import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  apiUrl = 'http://localhost:5000';

  constructor(
    private http: HttpClient
  ) {}

  getPredictions(cryptoId: string) {
    const url = `${this.apiUrl}/predict/${cryptoId}`;
    return this.http.get<any>(url);
  }

  getPrediction(cryptoId: string){
    const url = `${this.apiUrl}/predict_change/${cryptoId}`;
    return this.http.get<any>(url);
  }
}
