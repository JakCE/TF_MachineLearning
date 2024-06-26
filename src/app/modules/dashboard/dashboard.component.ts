import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { TuiDataListWrapperModule, TuiInputModule, TuiSelectModule } from '@taiga-ui/kit';
import { TuiInputPasswordModule } from '@taiga-ui/kit';
import { Router, RouterLink } from '@angular/router';
import { TuiAlertService, TuiButtonModule, TuiDataListModule, TuiLoaderModule, TuiModeModule, TuiNotificationModule } from '@taiga-ui/core';
import { ApiService } from '../../services/api/api.service'
import * as Plotly from 'plotly.js-dist';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [TuiButtonModule, TuiDataListModule, RouterLink, ReactiveFormsModule, 
    TuiInputModule, TuiInputPasswordModule, TuiNotificationModule, TuiModeModule,
    TuiSelectModule,
    TuiDataListModule,
    TuiDataListWrapperModule,
    HttpClientModule,
    TuiLoaderModule
  ],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit{
  dashForm!: FormGroup;
  cryptoId = '';
  resumen: any;
  loader: boolean = false;
  precio_ult_real: any;
  precio_predict: any;
  precio_change: any;

  items: any = [
    'dogecoin', 'shiba-inu', 'the-graph', 'axie-infinity',
    'the-sandbox', 'akash-network', 'pendle', 'singularitynet',
    'aioz-network'
  ];
  periodos: any = ['30 dias'];

  constructor(
    private formBuilder: FormBuilder,
    private apiService: ApiService,
    public router: Router,
    private alerts: TuiAlertService,
  ) {}

  ngOnInit() {
    this.dashForm = this.formBuilder.group({
      idCripto: [null,Validators.required],
      periodosCripto: [null,Validators.required]
    });
  }

  onSelect(){
    //console.log(this.dashForm.value.idCripto);
  }

  onClick(){
    this.loader = true;
    this.apiService.getPrediction(this.dashForm.value.idCripto).subscribe(data => {
      this.resumen = data;
      this.precio_change = parseFloat(data.percentage_change.toFixed(2));
      this.precio_predict = data.first_predicted_price;
      this.precio_ult_real = data.last_real_price;
      console.log(data);
      this.loader = false;
    }, error => {
      console.error('Error fetching predictions', error);
      this.loader = false;
    });

    this.apiService.getPredictions(this.dashForm.value.idCripto).subscribe(data=>{
      //this.createPlot(data);
      data.future_y = data.future_y.map((val: any) => val[0]);
      this.createPlotejm(data);
      //this.createFuturePlot(data);
      console.log(data);
    }, error => {
      console.error('Error fetching predictions', error);
      this.loader = false;
    });
  }

  //////

  /**
   * {
        x: ['2023-07-01', '2023-09-01', '2023-11-01', '2024-01-01', '2024-03-01', '2024-05-01', '2024-07-01'],
        y: [0.067, 0.069, 0.089, 0.11, 0.194, 0.116, 0.098],
        type: 'scatter',
        name: 'Precio Real'
      },
      {
        x: ['2023-07-01', '2023-09-01', '2023-11-01', '2024-01-01', '2024-03-01', '2024-05-01', '2024-07-01'],
        y: [0.067, 0.068, 0.087, 0.108, 0.186, 0.113, 0.095],
        type: 'scatter',
        name: 'Predicción Futura'
      }
   * 
   */

  createPlotejm(dataVal: any) {
    const data = [
      {
        x: dataVal.future_real_x,
        y: dataVal.future_real_y,
        type: 'scatter',
        name: 'Precio Real'
      },
      {
        x: dataVal.future_x,
        y: dataVal.future_y,
        type: 'scatter',
        name: 'Predicción Futura'
      }
    ];

    const layout = {
      title: `Predicción de Precios de ${this.dashForm.value.idCripto} con Modelo GRU`,
      xaxis: { title: 'Fecha' },
      yaxis: { title: 'Precio en USD' }
    };

    Plotly.newPlot('plot', data, layout);
  }

  //////

  createPlot1(data: any): void {
    const trainDates = data.train_data.map((d: any) => d.date);
    const testDates = data.test_data.map((d: any) => d.date);

    const trace1 = {
      x: trainDates,
      y: data.train_data.map((d: any) => d[0]),
      mode: 'lines',
      name: 'Precio Real (Entrenamiento)'
    };
    const trace2 = {
      x: trainDates.slice(30),
      y: data.train_predictions.map((d: any) => d[0]),
      mode: 'lines',
      name: 'Predicción (Entrenamiento)'
    };
    const trace3 = {
      x: testDates,
      y: data.test_data.map((d: any) => d[0]),
      mode: 'lines',
      name: 'Precio Real (Prueba)'
    };
    const trace4 = {
      x: testDates.slice(30),
      y: data.test_predictions.map((d: any) => d[0]),
      mode: 'lines',
      name: 'Predicción (Prueba)'
    };

    const layout = {
      title: `Predicción de Precios de ${this.dashForm.value.idCripto} con Modelo GRU`,
      xaxis: { title: 'Fecha' },
      yaxis: { title: 'Precio en USD' }
    };

    Plotly.newPlot('plot', [trace1, trace2, trace3, trace4], layout);
  }

  createFuturePlot1(data: any): void {
    const futureDates = data.future_dates.map((d: any) => new Date(d));
    const trace1 = {
      x: data.train_data.map((d: any) => new Date(d.date)),
      y: data.train_data.map((d: any) => d[0]),
      mode: 'lines',
      name: 'Precio Real'
    };
    const trace2 = {
      x: futureDates,
      y: data.future_prices.map((d: any) => d[0]),
      mode: 'lines',
      name: 'Predicción Futura'
    };

    const layout = {
      title: `Predicción de Precios de ${this.cryptoId} para el Próximo Mes`,
      xaxis: { title: 'Fecha' },
      yaxis: { title: 'Precio en USD' }
    };

    Plotly.newPlot('futurePlot', [trace1, trace2], layout);
  }

  /* NUEVO */
  createPlot(data: any): void {
    // Convertir fechas de entrenamiento y prueba a objetos Date
    const trainDates = data.train_data.map((d: any) => new Date(d.date)); // Suponiendo que la fecha está en 'date'
    const testDates = data.test_data.map((d: any) => new Date(d.date));   // Suponiendo que la fecha está en 'date'
  
    const trace1 = {
      x: trainDates,
      y: data.train_data.map((d: any) => d[0]), // Ajustar el índice según el precio real
      mode: 'lines',
      name: 'Precio Real (Entrenamiento)'
    };
    const trace2 = {
      x: trainDates.slice(30),
      y: data.train_predictions.map((d: any) => d[0]), // Ajustar el índice según la predicción
      mode: 'lines',
      name: 'Predicción (Entrenamiento)'
    };
    const trace3 = {
      x: testDates,
      y: data.test_data.map((d: any) => d[0]), // Ajustar el índice según el precio real
      mode: 'lines',
      name: 'Precio Real (Prueba)'
    };
    const trace4 = {
      x: testDates.slice(30),
      y: data.test_predictions.map((d: any) => d[0]), // Ajustar el índice según la predicción
      mode: 'lines',
      name: 'Predicción (Prueba)'
    };
  
    const layout = {
      title: `Predicción de Precios de ${this.cryptoId} con Modelo GRU`,
      xaxis: { title: 'Fecha', tickformat: '%b %Y' }, // Formato de fecha como 'Jul 2023'
      yaxis: { title: 'Precio en USD' }
    };
  
    Plotly.newPlot('plot', [trace1, trace2, trace3, trace4], layout);
  }
  
  createFuturePlot(data: any): void {
    // Convertir fechas futuras a objetos Date
    const futureDates = data.future_dates.map((d: any) => new Date(d));
  
    const trace1 = {
      x: data.train_data.map((d: any) => new Date(d.date)), // Suponiendo que la fecha está en 'date'
      y: data.train_data.map((d: any) => d[0]), // Ajustar el índice según el precio real
      mode: 'lines',
      name: 'Precio Real'
    };
    const trace2 = {
      x: futureDates,
      y: data.future_prices.map((d: any) => d[0]), // Ajustar el índice según la predicción futura
      mode: 'lines',
      name: 'Predicción Futura'
    };
  
    const layout = {
      title: `Predicción de Precios de ${this.cryptoId} para el Próximo Mes`,
      xaxis: { title: 'Fecha', tickformat: '%b %Y' }, // Formato de fecha como 'Jul 2023'
      yaxis: { title: 'Precio en USD' }
    };
  
    Plotly.newPlot('futurePlot', [trace1, trace2], layout);
  }  
}
