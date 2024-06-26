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

  getPredictions(cryptoId: string, periodo: any) {
    const url = `${this.apiUrl}/predict/${cryptoId}/${periodo}`;
    return this.http.get<any>(url);
  }

  getPrediction(cryptoId: string, periodo: any){
    const url = `${this.apiUrl}/predict_change/${cryptoId}/${periodo}`;
    return this.http.get<any>(url);
  }
}
